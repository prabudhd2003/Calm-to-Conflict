import pandas as pd

master = pd.read_csv("master_dataset.csv")
index_df = pd.read_csv("out/hubert_embeddings_index_rebuilt.csv")

# Keep only valid master rows used by extraction
master = master.dropna(subset=["sample_id", "wav_path", "start_sec", "end_sec", "label"]).copy()

master_ids = set(master["sample_id"].astype(str))
index_ids = set(index_df["sample_id"].astype(str))

print("Master shape:", master.shape)
print("Index shape:", index_df.shape)

print("Unique sample_ids in master:", len(master_ids))
print("Unique sample_ids in index:", len(index_ids))

missing_in_index = sorted(master_ids - index_ids)
extra_in_index = sorted(index_ids - master_ids)

print("Missing in index:", len(missing_in_index))
print("Extra in index:", len(extra_in_index))

print("Duplicate sample_ids in master:", master["sample_id"].duplicated().sum())
print("Duplicate sample_ids in index:", index_df["sample_id"].duplicated().sum())

if missing_in_index:
    print("\nFirst 10 missing in index:")
    print(missing_in_index[:10])

if extra_in_index:
    print("\nFirst 10 extra in index:")
    print(extra_in_index[:10])

if (
    len(missing_in_index) == 0
    and len(extra_in_index) == 0
    and master["sample_id"].duplicated().sum() == 0
    and index_df["sample_id"].duplicated().sum() == 0
):
    print("\nVALIDATION PASSED: rebuilt HuBERT index matches master dataset exactly.")
else:
    print("\nVALIDATION FAILED: check missing/extra/duplicate sample_ids.")