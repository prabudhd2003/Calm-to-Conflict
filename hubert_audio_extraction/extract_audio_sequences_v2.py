import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import math
import gc
import pandas as pd
import torch
import librosa
import torch.nn.functional as F

from transformers import HubertModel, Wav2Vec2FeatureExtractor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =========================================================
# Paths
# =========================================================
MASTER_CSV = "master_dataset_dyadic.csv"
OUTPUT_FILE = "out/audio_sequences_v2.pt"

# =========================================================
# Config
# =========================================================
BATCH_SIZE = 4
NUM_WORKERS = 0
MODEL_NAME = "superb/hubert-base-superb-er"

TEST_MODE = False
TEST_N = 100

# =========================================================
# Dataset
# =========================================================
class AudioExtractionDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]

        try:
            waveform_np, _ = librosa.load(
                row["wav_path"],
                sr=16000,
                offset=float(row["start_sec"]),
                duration=float(row["duration_sec"]),
                mono=True,
            )

            if waveform_np is None or len(waveform_np) < 160:
                return sample_id, None

            return sample_id, waveform_np

        except Exception as e:
            print(f"[LOAD ERROR] {sample_id} | {row['wav_path']} | {e}")
            return sample_id, None


def custom_collate(batch):
    batch = [b for b in batch if b[1] is not None]
    if len(batch) == 0:
        return [], []

    sample_ids = [b[0] for b in batch]
    waveforms = [b[1] for b in batch]
    return sample_ids, waveforms


# =========================================================
# Main extraction
# =========================================================
def extract_audio_sequences():
    df = pd.read_csv(MASTER_CSV)
    df = df.dropna(subset=["sample_id", "wav_path", "start_sec", "duration_sec", "label"]).copy()
    df = df.sort_values("sample_id").reset_index(drop=True)

    if TEST_MODE:
        df = df.head(TEST_N).copy()
        print(f"[TEST MODE] Using first {len(df)} rows")

    p95_duration = df["duration_sec"].quantile(0.95)
    MAX_AUDIO_LEN = math.ceil(p95_duration * 50.0)

    print(f"95th Percentile Duration: {p95_duration:.2f}s")
    print(f"MAX_AUDIO_LEN set to: {MAX_AUDIO_LEN} frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running extraction on: {device} | Batch Size: {BATCH_SIZE}")

    # -------------------------
    # Resume / checkpoint load
    # -------------------------
    if os.path.exists(OUTPUT_FILE):
        print(f"Loading existing progress from {OUTPUT_FILE} ...")
        try:
            saved_obj = torch.load(OUTPUT_FILE, map_location="cpu")

            if isinstance(saved_obj, dict) and "audio_sequences" in saved_obj:
                audio_dict = saved_obj["audio_sequences"]
                audio_mask_dict = saved_obj.get("audio_masks", {})
                audio_len_dict = saved_obj.get("audio_lengths", {})
                saved_max_len = saved_obj.get("max_audio_len", None)

                if saved_max_len is not None and saved_max_len != MAX_AUDIO_LEN:
                    print(f"Warning: saved max_audio_len={saved_max_len}, current={MAX_AUDIO_LEN}")
            else:
                # Backward compatibility if an old raw dict exists
                audio_dict = saved_obj
                audio_mask_dict = {}
                audio_len_dict = {}

            processed_ids = set(audio_dict.keys())
            df_to_process = df[~df["sample_id"].isin(processed_ids)].copy()

            print(f"Resuming: {len(processed_ids)} already extracted, {len(df_to_process)} remaining.")

        except Exception as e:
            print(f"Failed to load checkpoint {OUTPUT_FILE}: {e}")
            print("Starting from scratch instead.")
            audio_dict = {}
            audio_mask_dict = {}
            audio_len_dict = {}
            df_to_process = df
    else:
        audio_dict = {}
        audio_mask_dict = {}
        audio_len_dict = {}
        df_to_process = df
        print("No checkpoint found. Starting from scratch.")

    if len(df_to_process) == 0:
        print("All requested audio files are already processed!")
        return

    print(f"Loading Emotion-Tuned HuBERT: {MODEL_NAME}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = HubertModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    dataset = AudioExtractionDataset(df_to_process)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate,
        pin_memory=False,
    )

    print("\nStarting extraction...")
    with torch.no_grad():
        for batch_idx, (sample_ids, waveforms) in enumerate(tqdm(dataloader, desc="Processing Batches")):
            if len(waveforms) == 0:
                continue

            inputs = processor(
                waveforms,
                return_tensors="pt",
                sampling_rate=16000,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            seq_tensors = outputs.last_hidden_state.cpu()  # [B, T, 768]

            for i, s_id in enumerate(sample_ids):
                seq = seq_tensors[i]
                current_len = seq.shape[0]

                if current_len > MAX_AUDIO_LEN:
                    seq = seq[:MAX_AUDIO_LEN, :]
                    mask = torch.ones(MAX_AUDIO_LEN, dtype=torch.long)
                    valid_len = MAX_AUDIO_LEN
                elif current_len < MAX_AUDIO_LEN:
                    pad_amount = MAX_AUDIO_LEN - current_len
                    seq = F.pad(seq, (0, 0, 0, pad_amount), "constant", 0)
                    mask = torch.cat([
                        torch.ones(current_len, dtype=torch.long),
                        torch.zeros(pad_amount, dtype=torch.long)
                    ], dim=0)
                    valid_len = current_len
                else:
                    mask = torch.ones(MAX_AUDIO_LEN, dtype=torch.long)
                    valid_len = current_len

                audio_dict[s_id] = seq
                audio_mask_dict[s_id] = mask
                audio_len_dict[s_id] = valid_len

            # Save checkpoint every 50 batches
            if (batch_idx + 1) % 50 == 0:
                torch.save(
                    {
                        "audio_sequences": audio_dict,
                        "audio_masks": audio_mask_dict,
                        "audio_lengths": audio_len_dict,
                        "max_audio_len": MAX_AUDIO_LEN,
                        "feature_dim": 768,
                        "model_name": MODEL_NAME,
                        "batch_size": BATCH_SIZE,
                    },
                    OUTPUT_FILE,
                )

                del inputs, outputs, seq_tensors
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Final save
    torch.save(
        {
            "audio_sequences": audio_dict,
            "audio_masks": audio_mask_dict,
            "audio_lengths": audio_len_dict,
            "max_audio_len": MAX_AUDIO_LEN,
            "feature_dim": 768,
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
        },
        OUTPUT_FILE,
    )

    print(f"\nSUCCESS! Saved audio sequences to: {OUTPUT_FILE}")
    print(f"Number of samples saved: {len(audio_dict)}")
    print(f"Per-sample sequence shape: ({MAX_AUDIO_LEN}, 768)")
    print(f"If fully stacked, equivalent tensor shape would be: ({len(audio_dict)}, {MAX_AUDIO_LEN}, 768)")


if __name__ == "__main__":
    extract_audio_sequences()