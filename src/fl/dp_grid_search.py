"""
DP Grid Search: sigma (noise_multiplier) ve clip_norm kombinasyonlarını
tarayarak CIFAR-10 üzerinde en iyi doğruluk/epsilon dengesini bulur.

Kullanım:
    python -m src.fl.dp_grid_search --rounds 20 --local_epochs 2
"""

import argparse
import itertools
import random
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.cifar_resnet18 import DPResNetCIFAR10
from src.fl.partitions import iid_partitions
from src.fl.client import Client
from src.fl.aggregator import Aggregator
from src.fl.fedavg_runner import evaluate, set_seed
from src.privacy.dp_utils import compute_epsilon

SIGMA_GRID = [0.5, 0.8, 1.0, 1.2, 1.5]
CLIP_NORM_GRID = [0.5, 0.8, 1.0, 1.5, 2.0]


def run_single(sigma: float, clip_norm: float, config) -> dict:
    """Run one DP-FL experiment and return metrics."""
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
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

    partitions = iid_partitions(train_ds, config.num_clients)
    global_model = DPResNetCIFAR10().to(device)

    aggregator = Aggregator(
        encryption_context=None,
    )

    best_acc = 0.0
    for rnd in range(1, config.rounds + 1):
        client_updates = []
        for cid, idxs in enumerate(partitions):
            subset = torch.utils.data.Subset(train_ds, idxs)
            loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)
            client = Client(
                cid, loader, device,
                lr=config.lr, momentum=0.9, weight_decay=config.weight_decay,
                scheduler="none", encryption_context=None,
                dp_clip_norm=clip_norm,
                dp_noise_multiplier=sigma,
                dp_mechanism="gaussian",
            )
            update = client.train(global_model, epochs=config.local_epochs)
            client_updates.append(update)
        aggregator.federated_average(client_updates, global_model)
        acc, loss = evaluate(global_model, test_loader, device)
        best_acc = max(best_acc, acc)

    eps = compute_epsilon(
        noise_multiplier=sigma,
        num_rounds=config.rounds,
        target_delta=config.dp_target_delta,
    )
    return {"sigma": sigma, "clip_norm": clip_norm, "best_acc": best_acc,
            "final_acc": acc, "final_loss": loss, "epsilon": eps}


def main():
    p = argparse.ArgumentParser(description="DP Grid Search for CIFAR-10")
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dp_target_delta", type=float, default=1e-5)
    p.add_argument("--sigma_values", type=float, nargs="+", default=SIGMA_GRID,
                   help="Noise multiplier values to search")
    p.add_argument("--clip_values", type=float, nargs="+", default=CLIP_NORM_GRID,
                   help="Clip norm values to search")
    p.add_argument("--target_epsilon", type=float, default=3.0,
                   help="Only report configs with epsilon <= this value")
    config = p.parse_args()

    combos = list(itertools.product(config.sigma_values, config.clip_values))
    print(f"DP Grid Search: {len(combos)} combinations")
    print(f"  sigma  : {config.sigma_values}")
    print(f"  clip   : {config.clip_values}")
    print(f"  rounds : {config.rounds}, local_epochs: {config.local_epochs}")
    print(f"  target : epsilon <= {config.target_epsilon}\n")

    results = []
    for i, (sigma, clip_norm) in enumerate(combos, 1):
        eps_est = compute_epsilon(sigma, config.rounds, config.dp_target_delta)
        print(f"[{i}/{len(combos)}] sigma={sigma}, clip={clip_norm}, est_eps={eps_est:.2f} ... ", end="", flush=True)
        t0 = time.time()
        r = run_single(sigma, clip_norm, config)
        elapsed = time.time() - t0
        results.append(r)
        status = "OK" if r["epsilon"] <= config.target_epsilon else "OVER"
        print(f"acc={r['final_acc']*100:.2f}% eps={r['epsilon']:.2f} [{status}] ({elapsed:.0f}s)")

    print(f"\n{'='*70}")
    print(f"{'sigma':>8} {'clip':>8} {'acc%':>8} {'best%':>8} {'epsilon':>10} {'status':>8}")
    print(f"{'-'*70}")
    valid = [r for r in results if r["epsilon"] <= config.target_epsilon]
    for r in sorted(results, key=lambda x: -x["final_acc"]):
        status = "<= eps" if r["epsilon"] <= config.target_epsilon else "> eps"
        print(f"{r['sigma']:>8.2f} {r['clip_norm']:>8.2f} {r['final_acc']*100:>7.2f}% "
              f"{r['best_acc']*100:>7.2f}% {r['epsilon']:>10.4f} {status:>8}")

    if valid:
        best = max(valid, key=lambda x: x["final_acc"])
        print(f"\nBest config (epsilon <= {config.target_epsilon}):")
        print(f"  sigma={best['sigma']}, clip_norm={best['clip_norm']}")
        print(f"  accuracy={best['final_acc']*100:.2f}%, epsilon={best['epsilon']:.4f}")
    else:
        print(f"\nNo config achieved epsilon <= {config.target_epsilon} in {config.rounds} rounds.")
        print("Try: increase sigma, reduce rounds, or relax target_epsilon.")


if __name__ == "__main__":
    main()
