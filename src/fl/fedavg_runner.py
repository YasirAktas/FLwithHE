import argparse
import copy
import math
import random
import time
from typing import List

import numpy as np
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
from src.fl.client import ClientUpdate
from src.fl.aggregator import Aggregator
from src.he.encryption import PlainContext, HomomorphicContext, PaillierContext
from src.privacy.dp_utils import (
    add_gaussian_noise,
    add_laplace_noise,
    clip_delta,
    compute_laplace_epsilon,
    flatten_state_update,
    gaussian_noise_scale,
    laplace_noise_scale,
    privatize_update_delta,
)


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


def _client_update_norm(update: ClientUpdate) -> float:
    flat_update, _metadata = flatten_state_update(update.state_dict)
    if flat_update.numel() == 0:
        return 0.0
    return flat_update.norm(p=2).item()


def _compute_adaptive_clip_norm(config, norms: List[float], default_clip_norm: float) -> float:
    """
    Clipping defines the sensitivity used by the DP mechanism.

    Instead of using one fixed manual value, quantile clipping tracks the actual
    client update norm distribution. Adaptive smoothing then follows training
    dynamics across rounds, which prevents persistent over-clipping and reduces
    manual tuning. This improves signal strength but does not remove
    high-dimensional DP noise, so accuracy gains are limited by epsilon and d.
    """
    if not norms:
        return default_clip_norm

    q = float(getattr(config, "dp_clip_quantile", 50.0))
    alpha = float(getattr(config, "dp_clip_alpha", 0.9))
    min_clip = float(getattr(config, "dp_clip_min", 0.1))
    max_clip = float(getattr(config, "dp_clip_max", 10.0))

    clip_norm_current = float(np.percentile(np.asarray(norms, dtype=np.float64), q))
    strategy = getattr(config, "dp_clip_strategy", "adaptive")
    if strategy == "adaptive":
        if not hasattr(config, "clip_norm_running"):
            config.clip_norm_running = clip_norm_current
        config.clip_norm_running = alpha * float(config.clip_norm_running) + (1.0 - alpha) * clip_norm_current
        clip_norm = float(config.clip_norm_running)
    else:
        config.clip_norm_running = clip_norm_current
        clip_norm = clip_norm_current

    clip_norm = max(min_clip, min(clip_norm, max_clip))
    config.clip_norm_running = clip_norm

    return clip_norm


def _apply_client_level_dp_after_adaptive_clip(
    client_updates: List[ClientUpdate],
    mechanism: str,
    epsilon: float,
    delta: float,
    clip_norm: float,
    debug: bool,
) -> None:
    for update in client_updates:
        privatized, raw_norm, clipped_norm, clip_factor, noise_scale, noise_norm, ratio = privatize_update_delta(
            update.state_dict,
            mechanism=mechanism,
            epsilon=epsilon,
            delta_value=delta,
            clip_norm=clip_norm,
            debug=debug,
            debug_prefix="[DP DEBUG][adaptive client-level] ",
        )
        update.state_dict = privatized
        update.is_model_delta = True
        update.raw_update_norm = raw_norm
        update.clipped_update_norm = clipped_norm
        update.clipping_factor = clip_factor
        update.noise_scale = noise_scale
        update.noise_norm = noise_norm
        update.signal_noise_ratio = ratio
        if mechanism == "gaussian":
            update.gaussian_std = noise_scale
        else:
            update.laplace_scale = noise_scale
            dim = sum(v.numel() for v in privatized.values() if v.is_floating_point())
            update.laplace_expected_noise_l2 = math.sqrt(2.0 * float(dim)) * noise_scale if dim > 0 else 0.0


def _apply_server_level_dp(
    client_updates: List[ClientUpdate],
    global_model: torch.nn.Module,
    mechanism: str,
    epsilon: float,
    delta: float,
    clip_norm: float,
    clip_strategy: str,
    config,
    debug: bool,
):
    """
    Server-side DP mode:
      1) collect client deltas (no local DP),
      2) clip each client delta at server,
      3) aggregate clipped deltas,
      4) add DP noise once on aggregated delta,
      5) apply noisy aggregate to global model.
    """
    if not client_updates:
        return {
            "raw_norm": 0.0,
            "clipped_norm": 0.0,
            "clip_factor": 1.0,
            "noise_scale": 0.0,
            "noise_norm": 0.0,
            "signal_noise_ratio": 0.0,
            "clip_norm_eff": float(clip_norm),
        }

    mechanism = mechanism.lower()
    if mechanism not in {"gaussian", "laplace"}:
        raise ValueError(f"Unsupported dp_mechanism for server_level: {mechanism}")

    base_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
    total_samples = max(1, sum(u.num_samples for u in client_updates))

    deltas = []
    raw_norms = []
    for update in client_updates:
        if update.is_model_delta:
            delta_state = {
                k: v.detach().cpu().clone()
                for k, v in update.state_dict.items()
                if isinstance(v, torch.Tensor) and v.is_floating_point()
            }
        else:
            delta_state = {
                k: update.state_dict[k].detach().cpu() - base_state[k]
                for k in base_state
                if base_state[k].is_floating_point() and k in update.state_dict
            }
        flat_delta, _ = flatten_state_update(delta_state)
        raw_norms.append(flat_delta.norm(p=2).item() if flat_delta.numel() > 0 else 0.0)
        deltas.append(delta_state)

    clip_norm_eff = float(clip_norm)
    if clip_strategy in {"quantile", "adaptive"}:
        clip_norm_eff = _compute_adaptive_clip_norm(config, raw_norms, float(clip_norm))

    clipped_deltas = []
    clipped_norms = []
    clip_factors = []
    for idx, delta_state in enumerate(deltas):
        clipped_state = clip_delta(delta_state, clip_norm=clip_norm_eff)
        flat_clipped, _ = flatten_state_update(clipped_state)
        clipped_norm = flat_clipped.norm(p=2).item() if flat_clipped.numel() > 0 else 0.0
        raw_norm = raw_norms[idx]
        clip_factor = min(1.0, clip_norm_eff / (raw_norm + 1e-12))
        clipped_norms.append(clipped_norm)
        clip_factors.append(clip_factor)
        clipped_deltas.append(clipped_state)

    agg_delta = {}
    delta_keys = set()
    for delta_state in clipped_deltas:
        delta_keys.update(delta_state.keys())
    for key in delta_keys:
        acc = None
        for update, clipped_state in zip(client_updates, clipped_deltas):
            if key not in clipped_state:
                continue
            part = clipped_state[key] * (update.num_samples / total_samples)
            acc = part if acc is None else acc + part
        if acc is not None:
            agg_delta[key] = acc

    signal_flat, _ = flatten_state_update(agg_delta)
    signal_norm = signal_flat.norm(p=2).item() if signal_flat.numel() > 0 else 0.0
    sensitivity_scale = max((u.num_samples / total_samples) for u in client_updates)
    sensitivity = clip_norm_eff * sensitivity_scale

    if mechanism == "gaussian":
        # sigma = sensitivity * sqrt(2 log(1.25/delta)) / epsilon
        noise_multiplier = math.sqrt(2.0 * math.log(1.25 / float(delta))) / float(epsilon)
        noisy_agg_delta = add_gaussian_noise(
            agg_delta,
            clip_norm=clip_norm_eff,
            noise_multiplier=noise_multiplier,
            num_clients=len(client_updates),
            sensitivity_scale=sensitivity_scale,
        )
        noise_scale = gaussian_noise_scale(epsilon=epsilon, delta=delta, clip_norm=sensitivity)
    else:
        noisy_agg_delta = add_laplace_noise(
            agg_delta,
            clip_norm=clip_norm_eff,
            epsilon_per_round=epsilon,
            sensitivity_scale=sensitivity_scale,
        )
        dim = sum(v.numel() for v in agg_delta.values() if v.is_floating_point())
        noise_scale = (math.sqrt(float(dim)) * sensitivity) / float(epsilon) if dim > 0 else 0.0

    noise_delta = {
        k: noisy_agg_delta[k] - agg_delta[k]
        for k in agg_delta.keys()
        if k in noisy_agg_delta
    }
    noise_flat, _ = flatten_state_update(noise_delta)
    noise_norm = noise_flat.norm(p=2).item() if noise_flat.numel() > 0 else 0.0
    signal_noise_ratio = noise_norm / (signal_norm + 1e-6)

    updated_state = {}
    for key, value in base_state.items():
        if value.is_floating_point() and key in noisy_agg_delta:
            updated_state[key] = value + noisy_agg_delta[key].to(dtype=value.dtype)
        else:
            updated_state[key] = value
    global_model.load_state_dict(updated_state)

    raw_norm_avg = sum(raw_norms) / max(len(raw_norms), 1)
    clipped_norm_avg = sum(clipped_norms) / max(len(clipped_norms), 1)
    clip_factor_avg = sum(clip_factors) / max(len(clip_factors), 1)

    if debug:
        print("[DP DEBUG][server_level]")
        print("  clip_norm_eff:", clip_norm_eff)
        print("  sensitivity_scale(max weight):", sensitivity_scale)
        print("  raw_norm_avg:", raw_norm_avg)
        print("  clipped_norm_avg:", clipped_norm_avg)
        print("  clip_factor_avg:", clip_factor_avg)
        print("  noise_scale:", noise_scale)
        print("  signal_norm:", signal_norm)
        print("  noise_norm:", noise_norm)
        print("  n/s:", signal_noise_ratio)

    return {
        "raw_norm": raw_norm_avg,
        "clipped_norm": clipped_norm_avg,
        "clip_factor": clip_factor_avg,
        "noise_scale": noise_scale,
        "noise_norm": noise_norm,
        "signal_noise_ratio": signal_noise_ratio,
        "clip_norm_eff": clip_norm_eff,
    }


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
    dp_mode = getattr(config, "dp_mode", "client_level")
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
        adaptive_client_clip = use_dp and dp_mode == "client_level" and dp_clip_strategy in {"quantile", "adaptive"}
        for cid, idxs in enumerate(partitions):
            subset = torch.utils.data.Subset(train_ds, idxs)
            loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)
            client = Client(
                cid, loader, device,
                lr=lr, momentum=0.9, weight_decay=config.weight_decay,
                scheduler=config.scheduler,
                encryption_context=encryption_ctx,
                dp_clip_norm=None if adaptive_client_clip else (dp_clip_norm if use_dp else None),
                dp_noise_multiplier=dp_noise_multiplier if (use_dp and dp_mechanism == "gaussian") else 0.0,
                dp_mode=dp_mode if use_dp else "client_level",
                dp_mechanism=dp_mechanism if use_dp else "gaussian",
                dp_laplace_epsilon_per_round=dp_laplace_epsilon_per_round if (use_dp and dp_mechanism == "laplace") else 0.0,
                dp_epsilon=dp_epsilon if use_dp else 0.0,
                dp_delta=dp_delta,
                dp_debug=dp_debug and use_dp,
                return_model_delta=adaptive_client_clip,
                dp_clip_strategy=dp_clip_strategy if use_dp else "fixed",
                dp_clip_quantile=getattr(config, "dp_clip_quantile", 50.0),
                dp_clip_alpha=getattr(config, "dp_clip_alpha", 0.9),
                dp_clip_min=getattr(config, "dp_clip_min", 0.1),
                dp_clip_max=getattr(config, "dp_clip_max", 10.0),
            )
            update = client.train(global_model, epochs=config.local_epochs)
            client_updates.append(update)
        if adaptive_client_clip:
            norms = [_client_update_norm(update) for update in client_updates]
            clip_norm_current = _compute_adaptive_clip_norm(config, norms, dp_clip_norm)
            _apply_client_level_dp_after_adaptive_clip(
                client_updates,
                mechanism=dp_mechanism,
                epsilon=dp_epsilon,
                delta=dp_delta,
                clip_norm=clip_norm_current,
                debug=dp_debug,
            )
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
    dp_mode = getattr(config, "dp_mode", "client_level")
    dp_mechanism = getattr(config, "dp_mechanism", "gaussian")
    dp_clip_strategy = getattr(config, "dp_clip_strategy", "adaptive")
    dp_laplace_epsilon = getattr(config, "dp_laplace_epsilon", 5.0)
    warmup_rounds = getattr(config, "warmup_rounds", 0)
    use_ema = getattr(config, "use_ema", False)
    ema_decay = getattr(config, "ema_decay", 0.999)
    pretrain_rounds = getattr(config, "pretrain_rounds", 0)
    baseline_compare = getattr(config, "baseline_compare", False)
    dp_debug = getattr(config, "dp_debug", False)

    if use_dp and dp_mode not in {"dp_sgd", "client_level", "server_level"}:
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

    if use_dp and dp_mode in {"client_level", "server_level"} and dp_clip_strategy in {"quantile", "adaptive"} and encryption_ctx is not None:
        raise ValueError(
            "Client/server-level quantile/adaptive clipping needs plaintext client deltas before DP. "
            "Use --dp_clip_strategy fixed with encryption, or run DP without HE."
        )

    if use_dp and dp_mode == "server_level" and encryption_ctx is not None:
        raise ValueError(
            "server_level DP currently expects plaintext client deltas at the server. "
            "Disable encryption or use a client-side DP mode."
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
        if dp_mode == "server_level":
            total_part_samples = sum(len(idxs) for idxs in partitions)
            sensitivity_scale_preview = max((len(idxs) / max(total_part_samples, 1)) for idxs in partitions)
        else:
            sensitivity_scale_preview = 1.0
        if dp_mechanism == "gaussian":
            noise_scale = gaussian_noise_scale(
                epsilon=dp_epsilon,
                delta=dp_target_delta,
                clip_norm=dp_clip_norm * sensitivity_scale_preview,
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
            if dp_mode == "dp_sgd":
                noise_scale = dp_clip_norm / dp_laplace_epsilon_per_round
            else:
                model_dim = sum(p.numel() for p in global_model.state_dict().values() if p.is_floating_point())
                noise_scale = laplace_noise_scale(
                    epsilon=dp_laplace_epsilon_per_round,
                    clip_norm=dp_clip_norm * sensitivity_scale_preview,
                    dimension=model_dim,
                )
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
        adaptive_client_clip = use_dp and dp_mode == "client_level" and dp_clip_strategy in {"quantile", "adaptive"}
        server_level_dp = use_dp and dp_mode == "server_level"
        for cid, idxs in enumerate(partitions):
            subset = torch.utils.data.Subset(train_ds, idxs)
            loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)
            client = Client(
                cid, loader, device,
                lr=current_lr, momentum=0.9, weight_decay=config.weight_decay,
                scheduler=config.scheduler,
                encryption_context=encryption_ctx,
                dp_clip_norm=None if (adaptive_client_clip or server_level_dp) else (dp_clip_norm if use_dp else None),
                dp_noise_multiplier=dp_noise_multiplier if (use_dp and dp_mechanism == "gaussian") else 0.0,
                dp_mode=dp_mode if use_dp else "client_level",
                dp_mechanism=dp_mechanism if use_dp else "gaussian",
                dp_laplace_epsilon_per_round=dp_laplace_epsilon_per_round if (use_dp and dp_mechanism == "laplace") else 0.0,
                dp_epsilon=dp_epsilon if use_dp else 0.0,
                dp_delta=dp_target_delta,
                dp_debug=dp_debug and use_dp,
                return_model_delta=(adaptive_client_clip or server_level_dp),
                dp_clip_strategy=dp_clip_strategy if use_dp else "fixed",
                dp_clip_quantile=getattr(config, "dp_clip_quantile", 50.0),
                dp_clip_alpha=getattr(config, "dp_clip_alpha", 0.9),
                dp_clip_min=getattr(config, "dp_clip_min", 0.1),
                dp_clip_max=getattr(config, "dp_clip_max", 10.0),
            )
            update = client.train(global_model, epochs=config.local_epochs)
            client_updates.append(update)

        if adaptive_client_clip:
            norms = [_client_update_norm(update) for update in client_updates]
            clip_norm_current = _compute_adaptive_clip_norm(config, norms, dp_clip_norm)
            _apply_client_level_dp_after_adaptive_clip(
                client_updates,
                mechanism=dp_mechanism,
                epsilon=dp_epsilon,
                delta=dp_target_delta,
                clip_norm=clip_norm_current,
                debug=dp_debug,
            )

        total_train_time = sum(u.train_time for u in client_updates)
        total_encrypt_time = sum(u.encrypt_time for u in client_updates)
        server_dp_stats = None

        agg_start = time.time()
        if server_level_dp:
            server_dp_stats = _apply_server_level_dp(
                client_updates,
                global_model=global_model,
                mechanism=dp_mechanism,
                epsilon=dp_epsilon if dp_mechanism == "gaussian" else dp_laplace_epsilon_per_round,
                delta=dp_target_delta,
                clip_norm=dp_clip_norm,
                clip_strategy=dp_clip_strategy,
                config=config,
                debug=dp_debug,
            )
        else:
            aggregator.federated_average(client_updates, global_model)
        agg_time = time.time() - agg_start

        dp_stats_str = ""
        if use_dp and dp_mechanism in {"gaussian", "laplace"}:
            if server_level_dp and server_dp_stats is not None:
                raw_norm_avg = server_dp_stats["raw_norm"]
                clipped_norm_avg = server_dp_stats["clipped_norm"]
                clip_factor_avg = server_dp_stats["clip_factor"]
                noise_norm_avg = server_dp_stats["noise_norm"]
                signal_noise_ratio_avg = server_dp_stats["signal_noise_ratio"]
                noise_scale = server_dp_stats["noise_scale"]
            else:
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
            if (adaptive_client_clip or server_level_dp) and hasattr(config, "clip_norm_running"):
                dp_stats_str += f" clip_now={float(config.clip_norm_running):.3f}"
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
                   help="DP clipping strategy. DP-SGD uses per-example gradient norms; client_level uses client update norms.")
    p.add_argument("--dp_clip_quantile", type=float, default=50.0,
                   help="Norm percentile for quantile/adaptive clipping.")
    p.add_argument("--dp_clip_alpha", type=float, default=0.9,
                   help="Moving-average smoothing factor for adaptive clipping.")
    p.add_argument("--dp_clip_min", type=float, default=0.1,
                   help="Minimum clamp for quantile/adaptive clip norm.")
    p.add_argument("--dp_clip_max", type=float, default=10.0,
                   help="Maximum clamp for quantile/adaptive clip norm.")
    p.add_argument("--dp_mode", choices=["dp_sgd", "client_level", "server_level"], default="client_level",
                   help="DP modu: dp_sgd gradyan seviyesinde, client_level istemci deltası seviyesinde, server_level ise sunucuda aggregate delta seviyesinde DP uygular.")
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
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.compare_dp_mechanisms:
        run_dp_mechanism_comparison(args)
    else:
        run(args)
