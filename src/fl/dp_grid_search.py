"""
DP Grid Search: epsilon, mechanism ve clip_norm kombinasyonlarini
tarayarak CIFAR-10 uzerinde dogruluk/noise-scale dengesini bulur.

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
from src.privacy.dp_utils import gaussian_noise_scale, laplace_noise_scale

EPSILON_GRID = [0.5, 1.0, 2.0, 5.0]
CLIP_NORM_GRID = [0.5, 0.8, 1.0, 1.5, 2.0]
MECHANISMS = ["gaussian", "laplace"]


def run_single(mechanism: str, epsilon: float, clip_norm: float, config) -> dict:
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
                dp_mechanism=mechanism,
                dp_epsilon=epsilon,
                dp_delta=config.dp_target_delta,
            )
            update = client.train(global_model, epochs=config.local_epochs)
            client_updates.append(update)
        aggregator.federated_average(client_updates, global_model)
        acc, loss = evaluate(global_model, test_loader, device)
        best_acc = max(best_acc, acc)

    epsilon_total = epsilon * config.rounds
    if mechanism == "gaussian":
        noise_scale = gaussian_noise_scale(epsilon, config.dp_target_delta, clip_norm)
    else:
        model_dim = sum(p.numel() for p in global_model.state_dict().values() if p.is_floating_point())
        noise_scale = laplace_noise_scale(epsilon, clip_norm, model_dim)
    return {"mechanism": mechanism, "epsilon_per_round": epsilon, "clip_norm": clip_norm,
            "best_acc": best_acc, "final_acc": acc, "final_loss": loss,
            "epsilon": epsilon_total, "noise_scale": noise_scale}


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
    p.add_argument("--epsilon_values", type=float, nargs="+", default=EPSILON_GRID,
                   help="Per-round epsilon values to search")
    p.add_argument("--clip_values", type=float, nargs="+", default=CLIP_NORM_GRID,
                   help="Clip norm values to search")
    p.add_argument("--mechanisms", choices=["gaussian", "laplace"], nargs="+", default=MECHANISMS,
                   help="DP mechanisms to compare")
    p.add_argument("--target_epsilon", type=float, default=3.0,
                   help="Only report configs with epsilon <= this value")
    config = p.parse_args()

    combos = list(itertools.product(config.mechanisms, config.epsilon_values, config.clip_values))
    print(f"DP Grid Search: {len(combos)} combinations")
    print(f"  mechanisms : {config.mechanisms}")
    print(f"  epsilon    : {config.epsilon_values}")
    print(f"  clip       : {config.clip_values}")
    print(f"  rounds     : {config.rounds}, local_epochs: {config.local_epochs}")
    print(f"  target     : epsilon_total <= {config.target_epsilon}\n")

    results = []
    for i, (mechanism, epsilon, clip_norm) in enumerate(combos, 1):
        eps_est = epsilon * config.rounds
        print(f"[{i}/{len(combos)}] mechanism={mechanism}, eps/round={epsilon}, clip={clip_norm}, eps_total={eps_est:.2f} ... ", end="", flush=True)
        t0 = time.time()
        r = run_single(mechanism, epsilon, clip_norm, config)
        elapsed = time.time() - t0
        results.append(r)
        status = "OK" if r["epsilon"] <= config.target_epsilon else "OVER"
        print(f"acc={r['final_acc']*100:.2f}% eps={r['epsilon']:.2f} [{status}] ({elapsed:.0f}s)")

    print(f"\n{'='*70}")
    print(f"{'mechanism':>10} {'eps/r':>8} {'clip':>8} {'noise':>10} {'acc%':>8} {'best%':>8} {'epsilon':>10} {'status':>8}")
    print(f"{'-'*70}")
    valid = [r for r in results if r["epsilon"] <= config.target_epsilon]
    for r in sorted(results, key=lambda x: -x["final_acc"]):
        status = "<= eps" if r["epsilon"] <= config.target_epsilon else "> eps"
        print(f"{r['mechanism']:>10} {r['epsilon_per_round']:>8.2f} {r['clip_norm']:>8.2f} "
              f"{r['noise_scale']:>10.4f} {r['final_acc']*100:>7.2f}% "
              f"{r['best_acc']*100:>7.2f}% {r['epsilon']:>10.4f} {status:>8}")

    if valid:
        best = max(valid, key=lambda x: x["final_acc"])
        print(f"\nBest config (epsilon <= {config.target_epsilon}):")
        print(f"  mechanism={best['mechanism']}, epsilon_per_round={best['epsilon_per_round']}, clip_norm={best['clip_norm']}")
        print(f"  accuracy={best['final_acc']*100:.2f}%, epsilon_total={best['epsilon']:.4f}, noise_scale={best['noise_scale']:.4f}")
    else:
        print(f"\nNo config achieved epsilon <= {config.target_epsilon} in {config.rounds} rounds.")
        print("Try: increase sigma, reduce rounds, or relax target_epsilon.")


if __name__ == "__main__":
    main()
