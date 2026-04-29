import os
import pandas as pd
import torch
from tqdm import tqdm

PT_DIR = "out/pt_segments"
OUTPUT_CSV = "out/hubert_embeddings_index_rebuilt.csv"

files = sorted([f for f in os.listdir(PT_DIR) if f.endswith(".pt")])

rows = []

for fname in tqdm(files):
    path = os.path.join(PT_DIR, fname)
    obj = torch.load(path, map_location="cpu")

    rows.append({
        "sample_id": obj["sample_id"],
        "pt_path": path,
        "label": int(obj["label"]),
        "duration_sec": float(obj["duration_sec"])
    })

df = pd.DataFrame(rows)
df = df.sort_values("sample_id").reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)

print("Saved rebuilt index to:", OUTPUT_CSV)
print("Shape:", df.shape)
print(df.head())