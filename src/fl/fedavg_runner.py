import argparse
import csv
import copy
import math
import os
import random
import time
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.mnist_cnn import SimpleCNN
from src.models.cifar_resnet18 import ResNetCIFAR10, DPResNetCIFAR10, EMAModel
from src.models.ptbxl_cnn_large import PTBXL_CNN_Large
from src.models.ptbxl_cnn_medium import PTBXL_CNN_Medium
from src.models.ptbxl_logistic import PTBXL_Logistic
from src.models.ptbxl_lstm import PTBXL_LSTM
from src.fl.partitions import iid_partitions, dirichlet_partitions
from src.fl.client import Client
from src.fl.aggregator import Aggregator
from src.he.encryption import PlainContext, HomomorphicContext, PaillierContext
from src.privacy.dp_utils import compute_laplace_epsilon, gaussian_noise_scale


def _append_csv_row(csv_path: str, row: Dict[str, object]):
    fieldnames = [
        "timestamp",
        "round",
        "dataset",
        "model",
        "num_clients",
        "scheme",
        "payload_mode",
        "training_time",
        "encrypt_time",
        "aggregate_time",
        "decrypt_time",
        "he_total_time",
        "total_round_time",
        "ciphertext_count",
        "encrypted_values",
        "payload_nbytes",
        "accuracy",
        "loss",
        "mean_abs_error",
        "max_abs_error",
        "analytics_reference",
        "analytics_decrypted",
        "integer_reference",
        "integer_decrypted",
    ]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def warmup_cosine_lr(base_lr: float, current_round: int, total_rounds: int, warmup_rounds: int) -> float:
    """Linear warmup for warmup_rounds, then cosine decay to 0."""
    if current_round <= warmup_rounds:
        return base_lr * current_round / max(1, warmup_rounds)
    progress = (current_round - warmup_rounds) / max(1, total_rounds - warmup_rounds)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


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


def build_loaders(batch_size: int, dataset: str, use_aug: bool = False,
                   autoaugment: bool = False, ptbxl_data_dir: str = None):
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
            aug_list = []
            if autoaugment:
                aug_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
            aug_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            train_transform = transforms.Compose(aug_list)
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


def _run_fl_rounds(global_model, partitions, train_ds, test_loader, device, config,
                    aggregator, encryption_ctx, dp_clip_norm, use_dp,
                    num_rounds, lr, label="Round", ema=None):
    """Shared FL round loop used by both baseline/pretrain and main DP phases."""
    dp_mode = getattr(config, "dp_mode", "dp_sgd")
    dp_mechanism = getattr(config, "dp_mechanism", "gaussian")
    dp_clip_strategy = getattr(config, "dp_clip_strategy", "adaptive")
    dp_noise_multiplier = getattr(config, "dp_noise_multiplier", 0.0)
    dp_epsilon = getattr(config, "dp_epsilon", 1.0)
    dp_laplace_epsilon = getattr(config, "dp_laplace_epsilon", 5.0)
    # Laplace epsilon is static per round (no automatic division by num_rounds).
    dp_laplace_epsilon_per_round = dp_epsilon if dp_epsilon > 0 else dp_laplace_epsilon
    dp_delta = getattr(config, "dp_target_delta", 1e-5)
    dp_debug = getattr(config, "dp_debug", False)
    results = []
    for rnd in range(1, num_rounds + 1):
        round_start = time.time()
        client_updates: List = []
        for cid, idxs in enumerate(partitions):
            subset = torch.utils.data.Subset(train_ds, idxs)
            loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)
            client = Client(
                cid, loader, device,
                lr=lr, momentum=0.9, weight_decay=config.weight_decay,
                scheduler=config.scheduler,
                encryption_context=encryption_ctx,
                dp_clip_norm=dp_clip_norm if use_dp else None,
                dp_mechanism=dp_mechanism if use_dp else "gaussian",
                dp_epsilon=dp_epsilon if use_dp else 0.0,
                dp_delta=dp_delta,
                dp_debug=dp_debug and use_dp,
                dp_clip_strategy=dp_clip_strategy if use_dp else "fixed",
                dp_clip_quantile=getattr(config, "dp_clip_quantile", 50.0),
                dp_clip_alpha=getattr(config, "dp_clip_alpha", 0.9),
                dp_clip_min=getattr(config, "dp_clip_min", 0.1),
                dp_clip_max=getattr(config, "dp_clip_max", 10.0),
            )
            update = client.train(global_model, epochs=config.local_epochs)
            client_updates.append(update)
        aggregator.federated_average(client_updates, global_model)
        if ema is not None:
            ema.update(global_model)
        acc, loss = evaluate(global_model, test_loader, device)
        if math.isnan(loss):
            print(f"[DP DEBUG] LOSS IS NAN after {label} {rnd}")
            raise SystemExit(1)
        if loss > 1e5:
            print(f"[DP DEBUG] WARNING: loss explosion after {label} {rnd}: loss={loss:.6f}")
        round_time = time.time() - round_start
        results.append((rnd, acc, loss, round_time))
        print(f"  {label} {rnd:02d}: Acc={acc*100:.2f}% Loss={loss:.4f} | Time={round_time:.2f}s")
    return results


def run(config):
    set_seed(config.seed)
    device = torch.device("cuda" if (not config.no_cuda and torch.cuda.is_available()) else "cpu")
    use_dp = getattr(config, "use_dp", False)
    payload_mode = getattr(config, "payload_mode", "full_model")

    # Auto-enable augmentation for DP CIFAR-10
    use_aug = config.use_aug
    autoaugment = getattr(config, "autoaugment", False)
    if use_dp and config.dataset == "cifar10" and not use_aug:
        print("[DP] Augmentation auto-enabled for DP training")
        use_aug = True

    train_ds, test_loader = build_loaders(
        config.batch_size, config.dataset, use_aug=use_aug,
        autoaugment=autoaugment,
        ptbxl_data_dir=getattr(config, "ptbxl_data_dir", None),
    )
    start_time = time.time()
    round_times = []

    if config.partition == "iid":
        partitions = iid_partitions(train_ds, config.num_clients)
    else:
        partitions = dirichlet_partitions(train_ds, config.num_clients, alpha=config.dirichlet_alpha)

    # Select model — use DP-friendly variant (GroupNorm) when DP is active on CIFAR-10
    if config.dataset == "mnist":
        global_model = SimpleCNN().to(device)
    elif config.dataset == "cifar10":
        if use_dp:
            global_model = DPResNetCIFAR10().to(device)
            print("[DP] Using DPResNetCIFAR10 (GroupNorm instead of BatchNorm)")
        else:
            global_model = ResNetCIFAR10().to(device)
    elif config.dataset == "ptbxl":
        ptbxl_model = getattr(config, "ptbxl_model", "cnn_medium")
        if ptbxl_model == "cnn_large":
            global_model = PTBXL_CNN_Large().to(device)
        elif ptbxl_model == "cnn_medium":
            global_model = PTBXL_CNN_Medium().to(device)
        elif ptbxl_model == "logistic":
            global_model = PTBXL_Logistic().to(device)
        elif ptbxl_model == "lstm":
            global_model = PTBXL_LSTM().to(device)
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

    dp_clip_norm = getattr(config, "dp_clip_norm", 1.0)
    dp_noise_multiplier = getattr(config, "dp_noise_multiplier", 0.0)
    dp_epsilon = getattr(config, "dp_epsilon", 1.0)
    dp_target_delta = getattr(config, "dp_target_delta", 1e-5)
    dp_mode = getattr(config, "dp_mode", "dp_sgd")
    dp_mechanism = getattr(config, "dp_mechanism", "gaussian")
    dp_clip_strategy = getattr(config, "dp_clip_strategy", "adaptive")
    dp_laplace_epsilon = getattr(config, "dp_laplace_epsilon", 5.0)
    warmup_rounds = getattr(config, "warmup_rounds", 0)
    use_ema = getattr(config, "use_ema", False)
    ema_decay = getattr(config, "ema_decay", 0.999)
    pretrain_rounds = getattr(config, "pretrain_rounds", 0)
    baseline_compare = getattr(config, "baseline_compare", False)
    dp_debug = getattr(config, "dp_debug", False)

    if use_dp and dp_mode != "dp_sgd":
        raise ValueError(f"Unsupported dp_mode: {dp_mode}")

    if use_dp and dp_mechanism not in {"gaussian", "laplace"}:
        raise ValueError(f"Unsupported dp_mechanism: {dp_mechanism}")

    if use_dp and dp_epsilon <= 0:
        raise ValueError("For DP, --dp_epsilon must be > 0.")

    if use_dp and dp_mechanism == "laplace" and dp_laplace_epsilon <= 0:
        raise ValueError("For Laplace DP, --dp_laplace_epsilon must be > 0.")

    dp_laplace_epsilon_per_round = 0.0
    if use_dp and dp_mechanism == "laplace":
        # Laplace epsilon is interpreted as per-round epsilon.
        dp_laplace_epsilon_per_round = dp_epsilon if dp_epsilon > 0 else dp_laplace_epsilon

    aggregator = Aggregator(
        encryption_context=encryption_ctx,
    )

    if use_dp and pretrain_rounds > 0:
        raise ValueError(
            "pretrain_rounds > 0 is incompatible with a valid DP claim. "
            "Run pretraining as a separate non-DP experiment, then run DP with --pretrain_rounds 0."
        )

    if use_dp and config.local_epochs > 3:
        print(
            f"[DP WARNING] local_epochs={config.local_epochs} applies multiple local passes before one DP release. "
            "Keep local_epochs <= 3 for meaningful, stable round-level DP comparisons."
        )

    if config.use_encryption and use_dp:
        mode = "HE+DP"
    elif config.use_encryption:
        mode = "HE"
    elif use_dp:
        mode = "DP"
    else:
        mode = "BASELINE"
    print(f"[MODE] {mode}")

    if encryption_ctx is not None:
        scheme = getattr(config, "encryption_scheme", "ckks")
        print(f"[HE] Encryption: ACTIVE (dataset={config.dataset}, scheme={scheme})")
    else:
        print("[HE] Encryption: DISABLED for this run")

    if use_dp:
        cumulative_epsilon = dp_epsilon * config.rounds
        if dp_mechanism == "gaussian":
            noise_scale = gaussian_noise_scale(
                epsilon=dp_epsilon,
                delta=dp_target_delta,
                clip_norm=dp_clip_norm,
            )
            cumulative_delta = dp_target_delta * config.rounds
            print(
                f"[DP] enabled: mode={dp_mode} mechanism=gaussian "
                f"eps/round={dp_epsilon:.4f} delta/round={dp_target_delta:.0e} "
                f"clip={dp_clip_norm:.4f} noise_scale={noise_scale:.4f}"
            )
            print(
                f"[DP] planned privacy budget after {config.rounds} rounds: "
                f"epsilon~{cumulative_epsilon:.4f} delta~{cumulative_delta:.0e}"
            )
        else:
            epsilon_total = compute_laplace_epsilon(
                epsilon_per_round=dp_laplace_epsilon_per_round,
                num_rounds=config.rounds,
            )
            noise_scale = dp_clip_norm / dp_laplace_epsilon_per_round
            print(
                f"[DP] enabled: mode={dp_mode} mechanism=laplace "
                f"eps/round={dp_laplace_epsilon_per_round:.4f} delta=0 "
                f"clip={dp_clip_norm:.4f} noise_scale={noise_scale:.4f}"
            )
            print(
                f"[DP] planned privacy budget after {config.rounds} rounds: "
                f"epsilon={epsilon_total:.4f} delta=0"
            )
        print(
            f"[DP] clipping: strategy={dp_clip_strategy} "
            f"quantile={getattr(config, 'dp_clip_quantile', 50.0)} "
            f"alpha={getattr(config, 'dp_clip_alpha', 0.9)} "
            f"range=[{getattr(config, 'dp_clip_min', 0.1)}, {getattr(config, 'dp_clip_max', 10.0)}]"
        )
        if warmup_rounds > 0:
            print(f"[DP] LR schedule: {warmup_rounds} warmup rounds + cosine decay over {config.rounds} rounds")
    else:
        print("[DP] Differential Privacy: DISABLED for this run")

    # ----------------------------------------------------------------
    # Phase 0 (optional): Non-private baseline comparison
    # ----------------------------------------------------------------
    if baseline_compare and use_dp:
        print(f"\n{'='*60}")
        print(f"Phase 0: Non-private baseline ({config.rounds} rounds, no DP)")
        print(f"{'='*60}")
        rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        baseline_model = type(global_model)().to(device)
        baseline_model.load_state_dict(global_model.state_dict())
        baseline_agg = Aggregator(
            encryption_context=encryption_ctx,
        )
        baseline_results = _run_fl_rounds(
            baseline_model, partitions, train_ds, test_loader, device, config,
            baseline_agg, encryption_ctx, dp_clip_norm=None, use_dp=False,
            num_rounds=config.rounds, lr=config.lr, label="Baseline",
        )
        baseline_acc = baseline_results[-1][1]
        print(f"\n  >>> Non-private baseline final accuracy: {baseline_acc*100:.2f}%")
        del baseline_model, baseline_agg
        torch.set_rng_state(rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    # ----------------------------------------------------------------
    # EMA initialization
    # ----------------------------------------------------------------
    ema = None
    if use_ema and use_dp:
        ema = EMAModel(global_model, decay=ema_decay)
        print(f"[EMA] Enabled with decay={ema_decay}")

    for rnd in range(1, config.rounds + 1):
        round_start = time.time()

        # Warmup + cosine LR across rounds
        if use_dp and warmup_rounds > 0:
            current_lr = warmup_cosine_lr(config.lr, rnd, config.rounds, warmup_rounds)
        else:
            current_lr = config.lr

        client_updates: List = []
        for cid, idxs in enumerate(partitions):
            subset = torch.utils.data.Subset(train_ds, idxs)
            loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)
            client = Client(
                cid, loader, device,
                lr=current_lr, momentum=0.9, weight_decay=config.weight_decay,
                scheduler=config.scheduler,
                encryption_context=encryption_ctx,
                dp_clip_norm=dp_clip_norm if use_dp else None,
                dp_mechanism=dp_mechanism if use_dp else "gaussian",
                dp_epsilon=dp_epsilon if use_dp else 0.0,
                dp_delta=dp_target_delta,
                dp_debug=dp_debug and use_dp,
                dp_clip_strategy=dp_clip_strategy if use_dp else "fixed",
                dp_clip_quantile=getattr(config, "dp_clip_quantile", 50.0),
                dp_clip_alpha=getattr(config, "dp_clip_alpha", 0.9),
                dp_clip_min=getattr(config, "dp_clip_min", 0.1),
                dp_clip_max=getattr(config, "dp_clip_max", 10.0),
            )
            update = client.train(global_model, epochs=config.local_epochs)
            client_updates.append(update)

        total_train_time = sum(u.train_time for u in client_updates)
        total_encrypt_time = sum(u.encrypt_time for u in client_updates)

        agg_start = time.time()
        aggregator.federated_average(client_updates, global_model)
        agg_time = time.time() - agg_start

        dp_stats_str = ""
        if use_dp and dp_mechanism in {"gaussian", "laplace"}:
            raw_norm_avg = sum(u.raw_update_norm for u in client_updates) / len(client_updates)
            clipped_norm_avg = sum(u.clipped_update_norm for u in client_updates) / len(client_updates)
            clip_factor_avg = sum(u.clipping_factor for u in client_updates) / len(client_updates)
            noise_norm_avg = sum(u.noise_norm for u in client_updates) / len(client_updates)
            signal_noise_ratio_avg = sum(u.signal_noise_ratio for u in client_updates) / len(client_updates)
            noise_scale = client_updates[0].noise_scale if client_updates else 0.0
            dp_stats_str = (
                f" | DP(raw={raw_norm_avg:.3f} clip={clipped_norm_avg:.3f} factor={clip_factor_avg:.3f} "
                f"noise_scale={noise_scale:.3f} noise_norm={noise_norm_avg:.3f} n/s={signal_noise_ratio_avg:.3f})"
            )
            if signal_noise_ratio_avg > 10.0:
                dp_stats_str += " [warn: noise dominates]"

        if ema is not None:
            ema.update(global_model)

        # Evaluate — prefer EMA model when available
        ema_str = ""
        if ema is not None:
            ema_model_eval = type(global_model)().to(device)
            ema.apply_to(ema_model_eval)
            acc, loss = evaluate(ema_model_eval, test_loader, device)
            raw_acc, _ = evaluate(global_model, test_loader, device)
            ema_str = f" (raw={raw_acc*100:.2f}%)"
            del ema_model_eval
        else:
            acc, loss = evaluate(global_model, test_loader, device)

        if math.isnan(loss):
            print(f"[DP DEBUG][round={rnd}] LOSS IS NAN")
            raise SystemExit(1)
        if loss > 1e5:
            print(f"[DP DEBUG][round={rnd}] WARNING: loss explosion loss={loss:.6f}")

        round_time = time.time() - round_start
        round_times.append(round_time)
        elapsed = time.time() - start_time

        # Per-round privacy budget
        eps_str = ""
        if use_dp:
            if dp_mechanism == "gaussian":
                round_eps = dp_epsilon * rnd
            else:
                round_eps = compute_laplace_epsilon(
                    epsilon_per_round=dp_laplace_epsilon_per_round,
                    num_rounds=rnd,
                )
            eps_str = f" eps={round_eps:.4f}"

        lr_str = f" LR={current_lr:.6f}" if use_dp and warmup_rounds > 0 else ""

        print(
            f"Round {rnd:02d}: Acc={acc*100:.2f}%{ema_str} Loss={loss:.4f}{eps_str}{lr_str} "
            f"| Train={total_train_time:.2f}s Encrypt={total_encrypt_time:.2f}s "
            f"Agg={agg_time:.2f}s{dp_stats_str} | Total={round_time:.2f}s Elapsed={elapsed:.2f}s"
        )

        if getattr(config, "save_metrics_csv", None):
            model_name = getattr(config, "ptbxl_model", "-") if config.dataset == "ptbxl" else type(global_model).__name__
            if use_dp and config.use_encryption:
                scheme_name = f"{getattr(config, 'encryption_scheme', 'he')}+dp_{dp_mechanism}"
            elif use_dp:
                scheme_name = f"dp_{dp_mechanism}"
            elif config.use_encryption:
                scheme_name = getattr(config, "encryption_scheme", "none")
            else:
                scheme_name = "none"
            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "round": rnd,
                "dataset": config.dataset,
                "model": model_name,
                "num_clients": config.num_clients,
                "scheme": scheme_name,
                "payload_mode": payload_mode,
                "training_time": round(total_train_time, 6),
                "encrypt_time": round(total_encrypt_time, 6),
                "aggregate_time": round(agg_time, 6),
                "decrypt_time": 0.0,
                "he_total_time": round(total_encrypt_time + agg_time, 6),
                "total_round_time": round(round_time, 6),
                "ciphertext_count": 0,
                "encrypted_values": 0,
                "payload_nbytes": 0,
                "accuracy": round(acc, 6),
                "loss": round(loss, 6),
                "mean_abs_error": 0.0,
                "max_abs_error": 0.0,
                "analytics_reference": "",
                "analytics_decrypted": "",
                "integer_reference": "",
                "integer_decrypted": "",
            }
            _append_csv_row(config.save_metrics_csv, row)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    if use_dp:
        if dp_mechanism == "gaussian":
            final_eps = dp_epsilon * config.rounds
            final_delta = dp_target_delta * config.rounds
            print(f"\n[DP Summary] mode={dp_mode} mechanism=gaussian Final epsilon~{final_eps:.4f}, delta~{final_delta:.0e}")
        else:
            final_eps = compute_laplace_epsilon(
                epsilon_per_round=dp_laplace_epsilon_per_round,
                num_rounds=config.rounds,
            )
            print(f"\n[DP Summary] mode={dp_mode} mechanism=laplace Final epsilon={final_eps:.4f}, delta=0")
        if ema is not None:
            print(f"[EMA] Final EMA accuracy: {acc*100:.2f}%")

    if ema is not None:
        ema.apply_to(global_model)

    return global_model


def run_dp_mechanism_comparison(config, mechanisms=("gaussian", "laplace")):
    """Run the same DP-FedAvg configuration once per mechanism."""
    results = {}
    for mechanism in mechanisms:
        cfg = copy.deepcopy(config)
        cfg.use_dp = True
        cfg.dp_mechanism = mechanism
        cfg.compare_dp_mechanisms = False
        print(f"\n{'='*72}")
        print(f"DP mechanism comparison: {mechanism}")
        print(f"{'='*72}")
        results[mechanism] = run(cfg)
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Modular FedAvg Runner")
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", choices=["mnist", "cifar10", "ptbxl"], default="mnist")
    p.add_argument("--ptbxl_model", choices=["cnn_large", "cnn_medium", "logistic", "lstm"], default="cnn_medium",
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
    p.add_argument("--autoaugment", action="store_true",
                   help="CIFAR-10 AutoAugment policy (use_aug ile birlikte).")
    # Differential Privacy
    p.add_argument("--use_dp", action="store_true",
                   help="Differential Privacy'yi etkinleştir (DP-FedAvg).")
    p.add_argument("--dp_clip_norm", type=float, default=1.0,
                   help="DP için L2 clip normu (varsayılan: 1.0).")
    p.add_argument("--dp_clip_strategy", choices=["fixed", "quantile", "adaptive"], default="adaptive",
                   help="DP-SGD clipping strategy. Uses per-example gradient norms.")
    p.add_argument("--dp_clip_quantile", type=float, default=50.0,
                   help="Norm percentile for quantile/adaptive clipping.")
    p.add_argument("--dp_clip_alpha", type=float, default=0.9,
                   help="Moving-average smoothing factor for adaptive clipping.")
    p.add_argument("--dp_clip_min", type=float, default=0.1,
                   help="Minimum clamp for quantile/adaptive clip norm.")
    p.add_argument("--dp_clip_max", type=float, default=10.0,
                   help="Maximum clamp for quantile/adaptive clip norm.")
    p.add_argument("--dp_mode", choices=["dp_sgd"], default="dp_sgd",
                   help="DP mode. Only dp_sgd is supported.")
    p.add_argument("--dp_mechanism", choices=["gaussian", "laplace"], default="gaussian",
                   help="DP mekanizmasi: gaussian veya laplace.")
    p.add_argument("--dp_epsilon", type=float, default=1.0,
                   help="Round-basina epsilon. Gaussian ve Laplace mekanizmalari icin ortak karsilastirma parametresi.")
    p.add_argument("--dp_noise_multiplier", type=float, default=0.01,
                   help="Deprecated compatibility option; Gaussian noise is calibrated from --dp_epsilon and --dp_target_delta.")
    p.add_argument("--dp_laplace_epsilon", type=float, default=5.0,
                   help="Laplace mekanizmasi icin round-basina sabit epsilon (delta=0).")
    p.add_argument("--dp_target_delta", type=float, default=1e-5,
                   help="Hedef delta değeri (varsayılan: 1e-5).")
    # DP accuracy improvements
    p.add_argument("--warmup_rounds", type=int, default=0,
                   help="Linear LR warmup round sayısı; ardından cosine decay uygulanır.")
    p.add_argument("--use_ema", action="store_true",
                   help="EMA (Exponential Moving Average) ile DP gürültüsünü yumuşat.")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay oranı (varsayılan: 0.999).")
    p.add_argument("--pretrain_rounds", type=int, default=0,
                   help="Non-private pretraining rounds (do not use in DP mode).")
    p.add_argument("--baseline_compare", action="store_true",
                   help="DP çalıştırmadan önce non-private baseline ölç.")
    p.add_argument("--compare_dp_mechanisms", action="store_true",
                   help="Ayni ayarlarla gaussian ve laplace DP mekanizmalarini sirayla calistir.")
    p.add_argument("--dp_debug", action="store_true",
                   help="Print step-by-step DP tensor diagnostics and stop on NaN/Inf.")
    p.add_argument("--payload_mode", choices=["full_model", "analytics", "integer_stats"], default="full_model",
                   help="CSV/plot compatibility field. The current runner uses full-model FL updates.")
    p.add_argument("--save_metrics_csv", type=str, default=None,
                   help="Optional CSV path for per-round metrics export.")
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.compare_dp_mechanisms:
        run_dp_mechanism_comparison(args)
    else:
        run(args)
