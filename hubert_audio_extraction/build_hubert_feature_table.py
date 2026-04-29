import os
import pandas as pd
import torch
from tqdm import tqdm

INDEX_CSV = "out/hubert_embeddings_index_rebuilt.csv"
OUTPUT_CSV = "out/hubert_feature_table.csv"

index_df = pd.read_csv(INDEX_CSV)

rows = []

for _, row in tqdm(index_df.iterrows(), total=len(index_df)):
    path = row["pt_path"]
    obj = torch.load(path, map_location="cpu")

    emb = obj["features"]  # [768]
    if emb.ndim != 1:
        raise ValueError(f"Expected 1D embedding for {obj['sample_id']}, got shape {tuple(emb.shape)}")

    out_row = {
        "sample_id": obj["sample_id"],
        "label": int(obj["label"])
    }

    for i, val in enumerate(emb.tolist()):
        out_row[f"hubert_{i}"] = float(val)

    rows.append(out_row)

feature_df = pd.DataFrame(rows)
feature_df = feature_df.sort_values("sample_id").reset_index(drop=True)
feature_df.to_csv(OUTPUT_CSV, index=False)

print("Saved feature table to:", OUTPUT_CSV)
print("Shape:", feature_df.shape)
print(feature_df.head())