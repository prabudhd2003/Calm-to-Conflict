import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

from model import EarlyFusionLSTM

# -------------------------
# Paths
# -------------------------
AUDIO_PATH = "embeddings/audio_sequences_v2.pt"
TEXT_PATH = "embeddings/text_sequences_v2.pt"
VIDEO_PATH = "embeddings/video_self_sequences_v1.pt"
LABEL_PATH = "out/master_dataset_dyadic.csv"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Config
# -------------------------
SEED = 42
T_TARGET = 45
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-3
HIDDEN_DIM = 256
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resize_sequence(seq: torch.Tensor, target_len: int) -> torch.Tensor:
    # seq: [T, D]
    T, D = seq.shape

    if T == target_len:
        return seq

    if T > target_len:
        idx = torch.linspace(0, T - 1, steps=target_len).long()
        return seq[idx]

    pad_len = target_len - T
    pad = torch.zeros(pad_len, D, dtype=seq.dtype)
    return torch.cat([seq, pad], dim=0)


class EarlyFusionDataset(Dataset):
    def __init__(self, sample_ids, audio_dict, text_dict, video_dict, label_map, t_target=45):
        self.sample_ids = sample_ids
        self.audio_dict = audio_dict
        self.text_dict = text_dict
        self.video_dict = video_dict
        self.label_map = label_map
        self.t_target = t_target

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]

        audio_seq = self.audio_dict[sid].float()   # [647, 768]
        text_seq = self.text_dict[sid].float()     # [45, 768]
        video_seq = self.video_dict[sid].float()   # [65, 92]

        audio_seq = resize_sequence(audio_seq, self.t_target)
        text_seq = resize_sequence(text_seq, self.t_target)
        video_seq = resize_sequence(video_seq, self.t_target)

        fused_seq = torch.cat([audio_seq, text_seq, video_seq], dim=-1)  # [45, 1628]
        label = self.label_map[sid]

        return fused_seq, torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, macro_f1


def main():
    set_seed(SEED)

    print("Loading labels...")
    labels_df = pd.read_csv(LABEL_PATH)

    print("Loading text...")
    text_dict = torch.load(TEXT_PATH, map_location="cpu")

    print("Loading video...")
    video_dict = torch.load(VIDEO_PATH, map_location="cpu")

    print("Loading audio...")
    audio_obj = torch.load(AUDIO_PATH, map_location="cpu")
    audio_dict = audio_obj["audio_sequences"]

    label_map = dict(zip(labels_df["sample_id"].astype(str), labels_df["label"].astype(int)))
    file_id_map = dict(zip(labels_df["sample_id"].astype(str), labels_df["file_id"].astype(str)))

    audio_ids = set(audio_dict.keys())
    text_ids = set(text_dict.keys())
    video_ids = set(video_dict.keys())
    label_ids = set(label_map.keys())

    common_ids = sorted(audio_ids & text_ids & video_ids & label_ids)
    print(f"Common samples: {len(common_ids)}")

    # Group split by file_id to avoid leakage
    groups = np.array([file_id_map[sid] for sid in common_ids])
    sample_ids_np = np.array(common_ids)

    gss_1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_val_idx, test_idx = next(gss_1.split(sample_ids_np, groups=groups))

    train_val_ids = sample_ids_np[train_val_idx]
    test_ids = sample_ids_np[test_idx]

    train_val_groups = np.array([file_id_map[sid] for sid in train_val_ids])

    gss_2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
    train_idx, val_idx = next(gss_2.split(train_val_ids, groups=train_val_groups))

    train_ids = train_val_ids[train_idx].tolist()
    val_ids = train_val_ids[val_idx].tolist()
    test_ids = test_ids.tolist()

    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")

    train_ds = EarlyFusionDataset(train_ids, audio_dict, text_dict, video_dict, label_map, T_TARGET)
    val_ds = EarlyFusionDataset(val_ids, audio_dict, text_dict, video_dict, label_map, T_TARGET)
    test_ds = EarlyFusionDataset(test_ids, audio_dict, text_dict, video_dict, label_map, T_TARGET)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    input_dim = 768 + 768 + 92
    model = EarlyFusionLSTM(input_dim=input_dim, hidden_dim=HIDDEN_DIM).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_path = os.path.join(OUTPUT_DIR, "best_early_fusion_lstm.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_acc, val_f1 = evaluate(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best model to {best_path}")

    print("\nLoading best model for final test...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    test_acc, test_f1 = evaluate(model, test_loader, DEVICE)
    print(f"\nFINAL TEST RESULTS | Acc: {test_acc:.4f} | Macro F1: {test_f1:.4f}")

    with open(os.path.join(OUTPUT_DIR, "early_fusion_lstm_results.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.6f}\n")
        f.write(f"Test Macro F1: {test_f1:.6f}\n")


if __name__ == "__main__":
    main()