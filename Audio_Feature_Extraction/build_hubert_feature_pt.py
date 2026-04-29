import pandas as pd
import torch

INPUT_CSV = "out/hubert_feature_table.csv"
OUTPUT_PT = "out/hubert_feature_table.pt"

df = pd.read_csv(INPUT_CSV)
df = df.sort_values("sample_id").reset_index(drop=True)

sample_ids = df["sample_id"].tolist()
labels = torch.tensor(df["label"].tolist(), dtype=torch.long)

feature_cols = [c for c in df.columns if c.startswith("hubert_")]
features = torch.tensor(df[feature_cols].values, dtype=torch.float32)

obj = {
    "sample_ids": sample_ids,
    "labels": labels,
    "features": features,
}

torch.save(obj, OUTPUT_PT)

print("Saved:", OUTPUT_PT)
print("features shape:", features.shape)
print("labels shape:", labels.shape)
print("num sample_ids:", len(sample_ids))