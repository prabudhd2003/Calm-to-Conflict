import torch
import pandas as pd

TEXT_PATH = "embeddings/text_sequences_v2.pt"
VIDEO_PATH = "embeddings/video_self_sequences_v1.pt"
LABEL_PATH = "out/master_dataset_dyadic.csv"

# Known from finished audio extraction log
AUDIO_NUM_SAMPLES = 7859
AUDIO_SEQ_SHAPE = (647, 768)


def main():
    text = torch.load(TEXT_PATH, map_location="cpu")
    video = torch.load(VIDEO_PATH, map_location="cpu")
    labels_df = pd.read_csv(LABEL_PATH)

    print("=== AUDIO (from extraction log) ===")
    print("audio num samples:", AUDIO_NUM_SAMPLES)
    print("audio per-sample shape:", AUDIO_SEQ_SHAPE)

    print("\n=== TEXT ===")
    print("type:", type(text))
    if isinstance(text, dict):
        print("text top-level keys:", list(text.keys())[:10])

    print("\n=== VIDEO ===")
    print("type:", type(video))
    if isinstance(video, dict):
        print("video top-level keys:", list(video.keys())[:10])

    print("\n=== LABELS ===")
    print("labels shape:", labels_df.shape)
    print("label columns:", labels_df.columns.tolist())
    print(labels_df[["sample_id", "label"]].head())

    # -------------------------
    # Text
    # -------------------------
    text_dict = text
    print("\n[TEXT]")
    print("text dict length:", len(text_dict))
    first_text_id = next(iter(text_dict))
    print("first text sample_id:", first_text_id)
    print("first text shape:", tuple(text_dict[first_text_id].shape))

    # -------------------------
    # Video
    # -------------------------
    video_dict = video
    print("\n[VIDEO]")
    print("video dict length:", len(video_dict))
    first_video_id = next(iter(video_dict))
    print("first video sample_id:", first_video_id)
    print("first video shape:", tuple(video_dict[first_video_id].shape))

    # -------------------------
    # Common IDs
    # -------------------------
    text_ids = set(text_dict.keys())
    video_ids = set(video_dict.keys())
    label_ids = set(labels_df["sample_id"].astype(str))

    common_ids = text_ids & video_ids & label_ids

    print("\n=== ID OVERLAP ===")
    print("text ids:", len(text_ids))
    print("video ids:", len(video_ids))
    print("label ids:", len(label_ids))
    print("common ids (text ∩ video ∩ labels):", len(common_ids))


if __name__ == "__main__":
    main()