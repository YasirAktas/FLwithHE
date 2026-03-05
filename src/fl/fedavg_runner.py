import argparse
import ast
import random
import time
from typing import List
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.models.mnist_cnn import SimpleCNN
from src.models.cifar_resnet18 import ResNetCIFAR10
from src.models.ptbxl_cnn_medium import PTBXL_CNN_Medium
from src.models.ptbxl_cnn_large import PTBXL_CNN_Large
from src.fl.partitions import iid_partitions, dirichlet_partitions
from src.fl.client import Client
from src.fl.aggregator import Aggregator
from src.he.encryption import PlainContext, HomomorphicContext, PaillierContext

import os
import subprocess
import zipfile

def ensure_ptbxl_downloaded(ptbxl_root: str):
    """
    Download + unzip PTB-XL from Kaggle if required files are missing.
    Requires: kaggle CLI configured (~/.kaggle/kaggle.json).
    """
    root = Path(ptbxl_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Your dataset wrapper searches recursively for these, so just ensure they exist somewhere under root.
    required = ["ptbxl_database.csv", "scp_statements.csv"]
    if all(list(root.rglob(name)) for name in required):
        return  # already present

    # Download zip into root
    zip_path = root / "ptb-xl-dataset.zip"
    if not zip_path.exists():
        cmd = [
            "kaggle", "datasets", "download",
            "-d", "khyeh0719/ptb-xl-dataset",
            "-p", str(root),
            "--force",
        ]
        print("[PTB-XL] Downloading from Kaggle:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # Kaggle names the file after the dataset slug typically; find the newest zip if name differs
        zips = sorted(root.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if zips and zips[0] != zip_path:
            zips[0].rename(zip_path)

    # Unzip
    print(f"[PTB-XL] Extracting {zip_path} -> {root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    # Final sanity check
    if not all(list(root.rglob(name)) for name in required):
        raise RuntimeError(
            f"[PTB-XL] Download/extract completed but required files not found under {root}. "
            f"Expected at least: {required}"
        )


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * y.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total, total_loss / total


def _find_file_under_root(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists():
        return direct
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find '{filename}' under {root}")
    return matches[0]


class PTBXLKaggleDataset(Dataset):
    """PTB-XL dataset wrapper for Kaggle export (single-label, 5 superclass setup)."""

    CLASS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]

    def __init__(self, root: str, split: str = "train", sampling_rate: int = 100):
        try:
            import pandas as pd
            import wfdb
        except ImportError as exc:
            raise ImportError(
                "PTB-XL support requires `pandas` and `wfdb`. "
                "Install with: pip install pandas wfdb"
            ) from exc

        self.wfdb = wfdb
        root_path = Path(root).expanduser().resolve()
        db_csv = _find_file_under_root(root_path, "ptbxl_database.csv")
        scp_csv = _find_file_under_root(root_path, "scp_statements.csv")
        base_dir = db_csv.parent

        y_df = pd.read_csv(db_csv, index_col="ecg_id")
        agg_df = pd.read_csv(scp_csv, index_col=0)
        diag_map = agg_df[agg_df.diagnostic == 1].diagnostic_class.to_dict()

        def parse_superclasses(scp_codes_raw: str) -> List[str]:
            scp_codes = ast.literal_eval(scp_codes_raw)
            classes = {diag_map[code] for code in scp_codes.keys() if code in diag_map}
            return sorted(classes)

        y_df["labels"] = y_df.scp_codes.apply(parse_superclasses)
        y_df = y_df[y_df.labels.map(len) > 0].copy()

        if split == "train":
            y_df = y_df[y_df.strat_fold <= 8]
        elif split == "test":
            y_df = y_df[y_df.strat_fold == 10]
        else:
            raise ValueError(f"Unsupported split: {split}")

        class_to_idx = {label: idx for idx, label in enumerate(self.CLASS_ORDER)}

        def pick_single_label(labels: List[str]) -> int:
            for label in self.CLASS_ORDER:
                if label in labels:
                    return class_to_idx[label]
            return -1

        y_df["target"] = y_df.labels.apply(pick_single_label)
        y_df = y_df[y_df.target >= 0].copy()

        path_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        self.record_paths = [(base_dir / rel_path).as_posix() for rel_path in y_df[path_col].tolist()]
        self.targets = y_df["target"].astype(int).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        signal, _ = self.wfdb.rdsamp(self.record_paths[idx])
        x = torch.tensor(signal, dtype=torch.float32)  # [time, leads] => [1000, 12] at 100Hz
        # Per-lead normalization improves optimization stability across clients.
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y


def build_loaders(batch_size: int, dataset: str, use_aug: bool = False, ptbxl_root: str = "./data"):
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        if use_aug:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    elif dataset == "ptbxl":
        ensure_ptbxl_downloaded(ptbxl_root)
        train_ds = PTBXLKaggleDataset(root=ptbxl_root, split="train", sampling_rate=100)
        test_ds  = PTBXLKaggleDataset(root=ptbxl_root, split="test", sampling_rate=100)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    return train_ds, test_loader


def run(config):
    set_seed(config.seed)
    device = torch.device("cuda" if (not config.no_cuda and torch.cuda.is_available()) else "cpu")
    train_ds, test_loader = build_loaders(
        config.batch_size,
        config.dataset,
        use_aug=config.use_aug,
        ptbxl_root=config.ptbxl_root,
    )
    start_time = time.time()
    round_times = []

    if config.partition == "iid":
        partitions = iid_partitions(train_ds, config.num_clients)
    else:
        partitions = dirichlet_partitions(train_ds, config.num_clients, alpha=config.dirichlet_alpha)

    # Select model and (optional) encryption scheme
    if config.dataset == "mnist":
        global_model = SimpleCNN().to(device)
    elif config.dataset == "cifar10":
        global_model = ResNetCIFAR10().to(device)
    elif config.ptbxl_model == "large":
        global_model = PTBXL_CNN_Large().to(device)
    else:
        # "small" currently maps to medium, since no dedicated small PTB-XL CNN is defined.
        global_model = PTBXL_CNN_Medium().to(device)

    if config.use_encryption:
        scheme = getattr(config, "encryption_scheme", "ckks")
        if scheme == "paillier":
            encryption_ctx = PaillierContext()
        elif scheme == "ckks":
            encryption_ctx = HomomorphicContext()
        else:
            raise ValueError(f"Unknown encryption_scheme: {scheme}")
    else:
        encryption_ctx = None
    aggregator = Aggregator(encryption_context=encryption_ctx)
    if encryption_ctx is not None:
        scheme = getattr(config, "encryption_scheme", "ckks")
        print(f"[HE] Encryption: ACTIVE (dataset={config.dataset}, scheme={scheme})")
    else:
        print("[HE] Encryption: DISABLED for this run")

    for rnd in range(1, config.rounds + 1):
        round_start = time.time()
        client_updates: List = []
        for cid, idxs in enumerate(partitions):
            subset = torch.utils.data.Subset(train_ds, idxs)
            loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)
            client = Client(cid, loader, device, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay, scheduler=config.scheduler, encryption_context=encryption_ctx)
            update = client.train(global_model, epochs=config.local_epochs)
            client_updates.append(update)
        aggregator.federated_average(client_updates, global_model)
        acc, loss = evaluate(global_model, test_loader, device)
        round_time = time.time() - round_start
        round_times.append(round_time)
        elapsed = time.time() - start_time
        print(f"Round {rnd:02d}: Acc={acc*100:.2f}% Loss={loss:.4f} Time={round_time:.2f}s Elapsed={elapsed:.2f}s")
    return global_model


def parse_args():
    p = argparse.ArgumentParser(description="Modular FedAvg Runner")
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", choices=["mnist","cifar10","ptbxl"], default="mnist")
    p.add_argument("--ptbxl_model",choices=["small","medium","large"], default="small")
    p.add_argument("--ptbxl_root", type=str, default="./data/ptbxl",
                   help="Root folder containing PTB-XL files (ptbxl_database.csv, scp_statements.csv, records*/).")
    p.add_argument("--use_aug", action="store_true")
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--scheduler", choices=["none", "step", "cosine"], default="none")
    p.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    p.add_argument("--dirichlet_alpha", type=float, default=0.5)
    p.add_argument("--use_encryption", action="store_true")
    p.add_argument("--encryption_scheme", choices=["ckks", "paillier"], default="ckks",
                   help="Which HE scheme to use when --use_encryption is set.")
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
