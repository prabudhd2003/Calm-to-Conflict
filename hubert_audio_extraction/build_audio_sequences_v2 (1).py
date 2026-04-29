import os
import math
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
import torch.nn.functional as F
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# -------------------------
# Config
# -------------------------
CSV_PATH = "master_dataset.csv"
OUTPUT_PT = "out/audio_sequences_v2.pt"

MODEL_NAME = "superb/hubert-base-superb-er"
TARGET_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["sample_id", "wav_path", "start_sec", "end_sec", "label"]).copy()
df = df.sort_values("sample_id").reset_index(drop=True)

# Match teammate notebook logic
p95_duration = df["duration_sec"].quantile(0.95)
MAX_AUDIO_LEN = math.ceil(p95_duration * 50.0)

print(f"Using MAX_AUDIO_LEN = {MAX_AUDIO_LEN}")
print(f"Running on device: {DEVICE}")

# -------------------------
# Load model
# -------------------------
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = HubertModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# -------------------------
# Helper: load one utterance
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
# Main extraction
# -------------------------
sample_ids = []
labels = []
audio_sequences = []
audio_masks = []

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = str(row["sample_id"])
        wav_path = str(row["wav_path"])
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        label = int(row["label"])

        try:
            segment = load_audio_segment(wav_path, start_sec, end_sec, TARGET_SR)

            if segment is None or segment.shape[0] < 160:
                continue

            inputs = processor(
                segment.numpy(),
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs["input_values"].to(DEVICE)

            outputs = model(input_values=input_values)
            hidden_states = outputs.last_hidden_state.squeeze(0).cpu()  # [T, 768]

            current_len = hidden_states.shape[0]

            if current_len >= MAX_AUDIO_LEN:
                seq = hidden_states[:MAX_AUDIO_LEN, :]
                mask = torch.ones(MAX_AUDIO_LEN, dtype=torch.long)
            else:
                pad_len = MAX_AUDIO_LEN - current_len
                seq = F.pad(hidden_states, (0, 0, 0, pad_len), "constant", 0)
                mask = torch.cat([
                    torch.ones(current_len, dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long)
                ], dim=0)

            sample_ids.append(sample_id)
            labels.append(label)
            audio_sequences.append(seq)
            audio_masks.append(mask)

        except Exception as e:
            print(f"Failed on {sample_id}: {e}")

audio_tensor = torch.stack(audio_sequences)   # [N, T, 768]
mask_tensor = torch.stack(audio_masks)        # [N, T]
label_tensor = torch.tensor(labels, dtype=torch.long)

obj = {
    "sample_ids": sample_ids,
    "labels": label_tensor,
    "audio_sequences": audio_tensor,
    "audio_mask": mask_tensor,
    "max_audio_len": MAX_AUDIO_LEN,
    "feature_dim": audio_tensor.shape[-1],
    "model_name": MODEL_NAME,
}

torch.save(obj, OUTPUT_PT)

print(f"Saved: {OUTPUT_PT}")
print(f"audio_sequences shape: {audio_tensor.shape}")
print(f"audio_mask shape: {mask_tensor.shape}")
print(f"labels shape: {label_tensor.shape}")