import torch
import pandas as pd

AUDIO_PATH = "embeddings/audio_sequences_v2.pt"
TEXT_PATH = "embeddings/text_sequences_v2.pt"
VIDEO_PATH = "embeddings/video_self_sequences_v1.pt"
LABEL_PATH = "out/master_dataset_dyadic.csv"

OUTPUT_PATH = "out/early_fusion_dataset.pt"

# Chosen target sequence length for baseline
T_TARGET = 45


def resize_sequence(seq: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Resize sequence [T, D] to [target_len, D]
    by uniform downsampling or zero-padding.
    """
    T, D = seq.shape

    if T == target_len:
        return seq

    if T > target_len:
        idx = torch.linspace(0, T - 1, steps=target_len).long()
        return seq[idx]

    pad_len = target_len - T
    pad = torch.zeros(pad_len, D, dtype=seq.dtype)
    return torch.cat([seq, pad], dim=0)


def main():
    print("Loading text...")
    text_dict = torch.load(TEXT_PATH, map_location="cpu")

    print("Loading video...")
    video_dict = torch.load(VIDEO_PATH, map_location="cpu")

    print("Loading labels...")
    labels_df = pd.read_csv(LABEL_PATH)

    print("Loading audio...")
    audio_obj = torch.load(AUDIO_PATH, map_location="cpu")
    audio_dict = audio_obj["audio_sequences"]

    label_map = dict(zip(labels_df["sample_id"].astype(str), labels_df["label"].astype(int)))

    audio_ids = set(audio_dict.keys())
    text_ids = set(text_dict.keys())
    video_ids = set(video_dict.keys())
    label_ids = set(label_map.keys())

    common_ids = sorted(audio_ids & text_ids & video_ids & label_ids)

    print(f"audio ids: {len(audio_ids)}")
    print(f"text ids: {len(text_ids)}")
    print(f"video ids: {len(video_ids)}")
    print(f"label ids: {len(label_ids)}")
    print(f"common ids: {len(common_ids)}")

    fused_sequences = []
    labels = []
    sample_ids = []

    for sid in common_ids:
        audio_seq = audio_dict[sid].float()   # [647, 768]
        text_seq = text_dict[sid].float()     # [45, 768]
        video_seq = video_dict[sid].float()   # [65, 92]

        audio_seq = resize_sequence(audio_seq, T_TARGET)
        text_seq = resize_sequence(text_seq, T_TARGET)
        video_seq = resize_sequence(video_seq, T_TARGET)

        fused_seq = torch.cat([audio_seq, text_seq, video_seq], dim=-1)  # [45, 1628]

        fused_sequences.append(fused_seq)
        labels.append(label_map[sid])
        sample_ids.append(sid)

    X = torch.stack(fused_sequences)  # [N, 45, 1628]
    y = torch.tensor(labels, dtype=torch.long)

    out = {
        "sample_ids": sample_ids,
        "X": X,
        "y": y,
        "t_target": T_TARGET,
        "feature_dim": X.shape[-1],
        "num_samples": X.shape[0],
        "modal_dims": {
            "audio": 768,
            "text": 768,
            "video": 92,
        },
    }

    torch.save(out, OUTPUT_PATH)

    print("\nSaved:", OUTPUT_PATH)
    print("X shape:", tuple(X.shape))
    print("y shape:", tuple(y.shape))
    print("feature dim:", X.shape[-1])


if __name__ == "__main__":
    main()