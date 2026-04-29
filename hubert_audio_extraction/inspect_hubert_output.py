import os
import torch

PT_DIR = "out/pt_segments"

files = sorted(os.listdir(PT_DIR))
if not files:
    raise ValueError("No .pt files found in out/pt_segments")

path = os.path.join(PT_DIR, files[0])
obj = torch.load(path, map_location="cpu")

print("Loaded file:", path)
print("Keys:", list(obj.keys()))
print("sample_id:", obj["sample_id"])
print("label:", obj["label"])
print("wav_path:", obj["wav_path"])
print("start_sec:", obj["start_sec"])
print("end_sec:", obj["end_sec"])
print("duration_sec:", obj["duration_sec"])
print("features shape:", obj["features"].shape)
print("first_half_features shape:", obj["first_half_features"].shape)
print("second_half_features shape:", obj["second_half_features"].shape)
print("delta_features shape:", obj["delta_features"].shape)