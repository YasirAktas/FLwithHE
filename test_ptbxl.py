"""PTB-XL dataset yükleme ve boyut testi."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
from src.data.ptbxl_dataset import PTBXLDataset

DATA_DIR = "./data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

print("=" * 55)
print("PTB-XL Dataset Testi")
print("=" * 55)

train_ds = PTBXLDataset(data_dir=DATA_DIR, sampling_rate=100, split="train")
test_ds  = PTBXLDataset(data_dir=DATA_DIR, sampling_rate=100, split="test")

loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
x, y = next(iter(loader))

print()
print(f"x shape  : {x.shape}")       # (32, 1000, 12)
print(f"y shape  : {y.shape}")       # (32,)
print(f"x dtype  : {x.dtype}")
print(f"Sınıflar : {sorted(y.unique().tolist())}")
print(f"x min/max: {x.min():.3f} / {x.max():.3f}")
print()
print("TEST BAŞARILI")
