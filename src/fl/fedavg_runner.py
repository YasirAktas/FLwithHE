import argparse
import random
import time
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.mnist_cnn import SimpleCNN
from src.models.cifar_resnet18 import ResNetCIFAR10
from src.models.ptbxl_cnn_large import PTBXL_CNN_Large
from src.models.ptbxl_cnn_medium import PTBXL_CNN_Medium
from src.models.ptbxl_logistic import PTBXL_Logistic
from src.fl.partitions import iid_partitions, dirichlet_partitions
from src.fl.client import Client
from src.fl.aggregator import Aggregator
from src.he.encryption import PlainContext, HomomorphicContext, PaillierContext


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


def build_loaders(batch_size: int, dataset: str, use_aug: bool = False, ptbxl_data_dir: str = None):
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
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
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    elif dataset == "ptbxl":
        from src.data.ptbxl_dataset import PTBXLDataset
        data_dir = ptbxl_data_dir or "./data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
        train_ds = PTBXLDataset(data_dir=data_dir, split="train")
        test_ds  = PTBXLDataset(data_dir=data_dir, split="test")
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_ds, test_loader


def run(config):
    set_seed(config.seed)
    device = torch.device("cuda" if (not config.no_cuda and torch.cuda.is_available()) else "cpu")
    train_ds, test_loader = build_loaders(config.batch_size, config.dataset, use_aug=config.use_aug, ptbxl_data_dir=getattr(config, "ptbxl_data_dir", None))
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
    elif config.dataset == "ptbxl":
        ptbxl_model = getattr(config, "ptbxl_model", "cnn_medium")
        if ptbxl_model == "cnn_large":
            global_model = PTBXL_CNN_Large().to(device)
        elif ptbxl_model == "cnn_medium":
            global_model = PTBXL_CNN_Medium().to(device)
        elif ptbxl_model == "logistic":
            global_model = PTBXL_Logistic().to(device)
        else:
            raise ValueError(f"Unknown ptbxl_model: {ptbxl_model}")
        print(f"[PTB-XL] Model: {ptbxl_model}")
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

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
        total_train_time  = sum(u.train_time   for u in client_updates)
        total_encrypt_time = sum(u.encrypt_time for u in client_updates)
        agg_start = time.time()
        aggregator.federated_average(client_updates, global_model)
        agg_time = time.time() - agg_start
        acc, loss = evaluate(global_model, test_loader, device)
        round_time = time.time() - round_start
        round_times.append(round_time)
        elapsed = time.time() - start_time
        print(
            f"Round {rnd:02d}: Acc={acc*100:.2f}% Loss={loss:.4f} "
            f"| Train={total_train_time:.2f}s Encrypt={total_encrypt_time:.2f}s "
            f"Agg={agg_time:.2f}s | Total={round_time:.2f}s Elapsed={elapsed:.2f}s"
        )
    return global_model


def parse_args():
    p = argparse.ArgumentParser(description="Modular FedAvg Runner")
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", choices=["mnist", "cifar10", "ptbxl"], default="mnist")
    p.add_argument("--ptbxl_model", choices=["cnn_large", "cnn_medium", "logistic"], default="cnn_medium",
                   help="PTB-XL model seçimi (yalnızca --dataset ptbxl ile geçerli)")
    p.add_argument("--ptbxl_data_dir", type=str,
                   default="./data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
                   help="PTB-XL veri seti klasör yolu")
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
