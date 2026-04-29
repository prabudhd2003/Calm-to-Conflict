import os
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# -------------------------
# Config
# -------------------------
CSV_PATH = "master_dataset.csv"
OUTPUT_DIR = "out/pt_segments"
INDEX_CSV = "out/hubert_embeddings_index.csv"

MODEL_NAME = "facebook/hubert-base-ls960"
TARGET_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv(CSV_PATH)

required_cols = ["sample_id", "wav_path", "start_sec", "end_sec", "label"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df.dropna(subset=["sample_id", "wav_path", "start_sec", "end_sec", "label"]).copy()

# TEMP TEST ONLY
# df = df.head(5).copy()

# -------------------------
# Load HuBERT
# -------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = HubertModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# -------------------------
# Helper: load utterance segment
# -------------------------
def load_audio_segment(wav_path, start_sec, end_sec, target_sr=16000):
    waveform, sr = torchaudio.load(wav_path)  # [channels, time]

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    start_sample = int(float(start_sec) * sr)
    end_sample = int(float(end_sec) * sr)

    start_sample = max(0, start_sample)
    end_sample = min(waveform.shape[1], end_sample)

    if end_sample <= start_sample:
        return None

    segment = waveform[:, start_sample:end_sample]

    if segment.numel() == 0:
        return None

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        segment = resampler(segment)

    return segment.squeeze(0)  # [time]

# -------------------------
# Main extraction loop
# -------------------------
records = []

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = str(row["sample_id"])
        wav_path = str(row["wav_path"])
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        label = int(row["label"])

        out_path = os.path.join(OUTPUT_DIR, f"{sample_id}.pt")

        if os.path.exists(out_path):
            records.append({
                "sample_id": sample_id,
                "pt_path": out_path,
                "label": label
            })
            continue

        try:
            segment = load_audio_segment(wav_path, start_sec, end_sec, TARGET_SR)

            if segment is None or segment.shape[0] < 160:
                print(f"Skipping invalid/too-short sample: {sample_id}")
                continue

            inputs = feature_extractor(
                segment.numpy(),
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs["input_values"].to(DEVICE)
            outputs = model(input_values=input_values)

            hidden_states = outputs.last_hidden_state.squeeze(0).cpu()  # [T, D]
            pooled_embedding = hidden_states.mean(dim=0)

            T = hidden_states.shape[0]
            mid = max(1, T // 2)

            first_half = hidden_states[:mid]
            second_half = hidden_states[mid:] if mid < T else hidden_states[-1:].clone()

            first_half_mean = first_half.mean(dim=0)
            second_half_mean = second_half.mean(dim=0)
            delta_embedding = second_half_mean - first_half_mean

            save_obj = {
                "sample_id": sample_id,
                "label": label,
                "wav_path": wav_path,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": end_sec - start_sec,
                "features": pooled_embedding,
                "first_half_features": first_half_mean,
                "second_half_features": second_half_mean,
                "delta_features": delta_embedding
            }

            torch.save(save_obj, out_path)

            records.append({
                "sample_id": sample_id,
                "pt_path": out_path,
                "label": label
            })

        except Exception as e:
            print(f"Failed on {sample_id}: {e}")

# -------------------------
# Save index CSV
# -------------------------
index_df = pd.DataFrame(records)
index_df.to_csv(INDEX_CSV, index=False)

print(f"Done. Saved {len(index_df)} embeddings.")
print(f"Index written to: {INDEX_CSV}")