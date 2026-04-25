"""Differential Privacy utilities for Federated Learning."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Dict, List, Optional, Tuple, Union

import torch

try:
    import opendp.prelude as dp
except ImportError:
    dp = None


# ---------------------------------------------------------------------------
# 1. Full-vector update helpers
# ---------------------------------------------------------------------------

def flatten_state_update(delta: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[str, torch.Size, torch.dtype]]]:
    """Flatten all floating-point tensors in a state update into one vector."""
    flat_chunks: List[torch.Tensor] = []
    metadata: List[Tuple[str, torch.Size, torch.dtype]] = []
    for key, value in delta.items():
        if not value.is_floating_point():
            continue
        flat_value = value.reshape(-1)
        flat_chunks.append(flat_value)
        metadata.append((key, value.shape, value.dtype))

    if not flat_chunks:
        return torch.empty(0, dtype=torch.float32), metadata
    return torch.cat(flat_chunks), metadata


def unflatten_state_update(
    flat_delta: torch.Tensor,
    metadata: List[Tuple[str, torch.Size, torch.dtype]],
    reference_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Restore a flattened floating-point update back into a state-dict shaped delta."""
    restored: Dict[str, torch.Tensor] = {}
    offset = 0
    for key, shape, dtype in metadata:
        numel = math.prod(shape)
        chunk = flat_delta[offset:offset + numel].reshape(shape)
        restored[key] = chunk.to(dtype=dtype)
        offset += numel

    for key, value in reference_state.items():
        if key not in restored and value.is_floating_point():
            restored[key] = torch.zeros_like(value)
    return restored


def debug_tensor(name: str, tensor: torch.Tensor) -> None:
    """Print numerical diagnostics for a tensor without changing it."""
    print(f"{name}:")
    if tensor.numel() == 0:
        print("  empty: True")
        print("  has_nan:", False)
        print("  has_inf:", False)
        return

    finite_tensor = tensor.detach()
    print("  min:", finite_tensor.min().item())
    print("  max:", finite_tensor.max().item())
    print("  mean:", finite_tensor.mean().item())
    print("  norm:", torch.norm(finite_tensor).item())
    print("  has_nan:", torch.isnan(finite_tensor).any().item())
    print("  has_inf:", torch.isinf(finite_tensor).any().item())


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    """Fail fast when a tensor contains NaN or Inf."""
    if torch.isnan(tensor).any().item():
        raise FloatingPointError(f"{name} contains NaN")
    if torch.isinf(tensor).any().item():
        raise FloatingPointError(f"{name} contains Inf")


def clip_update(
    update: torch.Tensor,
    clip_norm: float,
    norm_type: Union[float, int, str],
) -> Tuple[torch.Tensor, float, float, float]:
    """Clip one flattened update vector with a single global norm bound."""
    if clip_norm <= 0:
        raise ValueError("clip_norm must be > 0")
    if update.numel() == 0:
        return update.clone(), 0.0, 0.0, 1.0

    p = 1 if str(norm_type).lower() in {"1", "l1"} else 2
    raw_norm = update.norm(p=p).item()
    clipping_factor = min(1.0, float(clip_norm) / (raw_norm + 1e-12))
    clipped = update * clipping_factor
    clipped_norm = clipped.norm(p=p).item()
    return clipped, raw_norm, clipped_norm, clipping_factor


def gaussian_noise_scale(epsilon: float, delta: float, clip_norm: float) -> float:
    """Return Gaussian sigma calibrated to L2 sensitivity clip_norm."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 for Gaussian mechanism")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1) for Gaussian mechanism")
    if clip_norm <= 0:
        raise ValueError("clip_norm must be > 0")
    return float(clip_norm) * math.sqrt(2.0 * math.log(1.25 / float(delta))) / float(epsilon)


def laplace_noise_scale(epsilon: float, clip_norm: float, dimension: int) -> float:
    """Return Laplace b using L1 sensitivity sqrt(d) * clip_norm."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 for Laplace mechanism")
    if clip_norm <= 0:
        raise ValueError("clip_norm must be > 0")
    if dimension <= 0:
        return 0.0
    l1_sensitivity = float(clip_norm) * math.sqrt(float(dimension))
    return l1_sensitivity / float(epsilon)


def laplace_l1_noise_scale(epsilon: float, l1_sensitivity: float) -> float:
    """Return Laplace b when the vector was clipped directly in L1."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 for Laplace mechanism")
    if l1_sensitivity <= 0:
        raise ValueError("l1_sensitivity must be > 0 for Laplace mechanism")
    return float(l1_sensitivity) / float(epsilon)


def add_dp_noise(
    update: torch.Tensor,
    mechanism: str,
    epsilon: float,
    delta: float,
    clip_norm: float,
    debug: bool = False,
    debug_prefix: str = "",
) -> Tuple[torch.Tensor, float]:
    """
    Add DP noise to one already-clipped full update vector.

    Gaussian uses L2 sensitivity clip_norm:
        sigma = clip_norm * sqrt(2 * log(1.25 / delta)) / epsilon

    Laplace uses the L1 bound implied by L2 clipping in d dimensions:
        b = clip_norm * sqrt(d) / epsilon
    """
    mechanism = mechanism.lower()
    if update.numel() == 0:
        return update.clone(), 0.0

    if mechanism == "gaussian":
        scale = gaussian_noise_scale(epsilon=epsilon, delta=delta, clip_norm=clip_norm)
        noise = torch.normal(
            mean=0.0,
            std=scale,
            size=update.shape,
            dtype=update.dtype,
            device=update.device,
        )
    elif mechanism == "laplace":
        scale = laplace_noise_scale(epsilon=epsilon, clip_norm=clip_norm, dimension=update.numel())
        dist = torch.distributions.Laplace(
            loc=torch.tensor(0.0, device=update.device, dtype=update.dtype),
            scale=torch.tensor(scale, device=update.device, dtype=update.dtype),
        )
        noise = dist.sample(update.shape)
    else:
        raise ValueError(f"Unsupported DP mechanism: {mechanism}")

    noisy_raw = update + noise
    if debug:
        print(f"{debug_prefix}Noise stats:")
        print("  scale:", scale)
        print("  std:", noise.std(unbiased=False).item())
        print("  mean:", noise.mean().item())
        print("  max:", noise.max().item())
        print("  min:", noise.min().item())
        print("  signal_norm:", torch.norm(update).item())
        print("  noise_norm:", torch.norm(noise).item())
        signal_norm = torch.norm(update).item()
        noise_norm = torch.norm(noise).item()
        if signal_norm > 0 and noise_norm > 10.0 * signal_norm:
            print(f"{debug_prefix}WARNING: noise_norm is more than 10x signal_norm")
        print(f"{debug_prefix}After noise:")
        print("  norm:", torch.norm(noisy_raw).item())
        print("  has_nan:", torch.isnan(noisy_raw).any().item())
        print("  has_inf:", torch.isinf(noisy_raw).any().item())
        if torch.isnan(noisy_raw).any().item():
            print(f"{debug_prefix}ERROR: NaN in update before sending")
            raise SystemExit(1)
        if torch.isinf(noisy_raw).any().item():
            print(f"{debug_prefix}ERROR: Inf in update before sending")
            raise SystemExit(1)

    noisy = torch.nan_to_num(noisy_raw, nan=0.0, posinf=1e6, neginf=-1e6)
    return noisy, scale


def _parameter_grad_vector(model: torch.nn.Module) -> Tuple[torch.Tensor, List[Tuple[torch.nn.Parameter, torch.Size, torch.dtype]]]:
    """Flatten all existing floating-point gradients into one vector."""
    chunks: List[torch.Tensor] = []
    metadata: List[Tuple[torch.nn.Parameter, torch.Size, torch.dtype]] = []
    for param in model.parameters():
        if param.grad is None or not param.grad.is_floating_point():
            continue
        grad = param.grad.detach()
        chunks.append(grad.reshape(-1))
        metadata.append((param, grad.shape, grad.dtype))
    if not chunks:
        return torch.empty(0), metadata
    return torch.cat(chunks), metadata


def _write_grad_vector(flat_grad: torch.Tensor, metadata: List[Tuple[torch.nn.Parameter, torch.Size, torch.dtype]]) -> None:
    """Write a flattened gradient vector back into model parameter gradients."""
    offset = 0
    for param, shape, dtype in metadata:
        numel = math.prod(shape)
        piece = flat_grad[offset:offset + numel].reshape(shape).to(dtype=dtype, device=param.device)
        param.grad = piece.clone()
        offset += numel


def _sample_noise_like(update: torch.Tensor, mechanism: str, scale: float) -> torch.Tensor:
    if mechanism == "gaussian":
        return torch.normal(
            mean=0.0,
            std=scale,
            size=update.shape,
            dtype=update.dtype,
            device=update.device,
        )
    if mechanism == "laplace":
        dist = torch.distributions.Laplace(
            loc=torch.tensor(0.0, device=update.device, dtype=update.dtype),
            scale=torch.tensor(scale, device=update.device, dtype=update.dtype),
        )
        return dist.sample(update.shape)
    raise ValueError(f"Unsupported DP mechanism: {mechanism}")


def _print_signal_noise_stats(debug_prefix: str, signal: torch.Tensor, noise: torch.Tensor, scale: float) -> None:
    signal_norm = torch.norm(signal).item()
    noise_norm = torch.norm(noise).item()
    print(f"{debug_prefix}Noise stats:")
    print("  scale:", scale)
    print("  std:", noise.std(unbiased=False).item())
    print("  mean:", noise.mean().item())
    print("  max:", noise.max().item())
    print("  min:", noise.min().item())
    print("  signal_norm:", signal_norm)
    print("  noise_norm:", noise_norm)
    ratio = noise_norm / (signal_norm + 1e-12)
    print("  signal_vs_noise_ratio:", ratio)
    # High-dimensional vectors amplify aggregate noise magnitude:
    # for iid Gaussian coordinates, noise_norm is approximately sqrt(d) * sigma.
    if signal_norm > 0 and noise_norm > 10.0 * signal_norm:
        print(f"{debug_prefix}WARNING: noise_norm is more than 10x signal_norm")


def apply_dp_sgd(
    model: torch.nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    mechanism: str,
    clip_norm: float,
    epsilon: float,
    delta: float,
    criterion: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    debug: bool = False,
    debug_prefix: str = "",
    clip_strategy: str = "fixed",
    clip_quantile: float = 50.0,
    clip_alpha: float = 0.9,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
    clip_state: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Apply per-example DP-SGD during local training.

    Expected behavior:
      - DP-SGD + Gaussian is the standard, usually most stable option.
      - DP-SGD + Laplace is noisier because it must match L1 sensitivity.

    Pipeline:
        for x_i, y_i in batch:
            compute grad_i
            clip grad_i
        stack clipped grads
        average
        add calibrated noise to the averaged gradient
        optimizer.step()

    This is intentionally simple and exact, but slow. For large models/datasets,
    Opacus or a vmap/functorch implementation is much faster.
    """
    mechanism = mechanism.lower()
    if mechanism not in {"gaussian", "laplace"}:
        raise ValueError(f"Unsupported DP mechanism: {mechanism}")
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    if device is None:
        device = next(model.parameters()).device

    model.train()
    total_loss = 0.0
    total_samples = 0
    grad_norm_sum = 0.0
    clipped_norm_sum = 0.0
    noise_norm_sum = 0.0
    noise_scale_sum = 0.0
    clip_norm_sum = 0.0
    clip_factor_sum = 0.0
    steps = 0

    norm_type = 2 if mechanism == "gaussian" else 1
    for batch_idx, (inputs, targets) in enumerate(data_loader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = targets.size(0)
        flat_grads: List[torch.Tensor] = []
        clipped_grads: List[torch.Tensor] = []
        raw_norms: List[float] = []
        clipped_norms: List[float] = []
        clip_factors: List[float] = []
        metadata = None
        batch_loss_sum = 0.0

        for sample_idx in range(batch_size):
            optimizer.zero_grad()
            sample_x = inputs[sample_idx:sample_idx + 1]
            sample_y = targets[sample_idx:sample_idx + 1]
            output = model(sample_x)
            loss = criterion(output, sample_y)
            if torch.isnan(loss).item():
                raise FloatingPointError(f"{debug_prefix}LOSS IS NAN before per-example DP-SGD step")
            if torch.isinf(loss).item():
                raise FloatingPointError(f"{debug_prefix}LOSS IS INF before per-example DP-SGD step")
            if loss.item() > 1e5:
                print(f"{debug_prefix}WARNING: loss explosion loss={loss.item():.6f}")
            loss.backward()

            flat_grad, sample_metadata = _parameter_grad_vector(model)
            if flat_grad.numel() == 0:
                continue
            if metadata is None:
                metadata = sample_metadata
            assert_finite_tensor(f"{debug_prefix}grad_i_before_clipping", flat_grad)
            raw_norm = flat_grad.norm(p=norm_type).item()
            flat_grads.append(flat_grad)
            raw_norms.append(raw_norm)
            batch_loss_sum += loss.item()

        if not flat_grads or metadata is None:
            optimizer.zero_grad()
            continue

        clip_norm_eff = float(clip_norm)
        if clip_strategy in {"quantile", "adaptive"}:
            # Clipping defines sensitivity in DP. Instead of a fixed value,
            # quantile clipping tracks the actual per-example gradient norm
            # distribution; adaptive smoothing follows training dynamics and
            # reduces manual tuning without changing the DP mechanism.
            norms_t = torch.tensor(raw_norms, dtype=torch.float64)
            q = max(0.0, min(float(clip_quantile), 100.0)) / 100.0
            clip_norm_current = float(torch.quantile(norms_t, q).item())
            if clip_strategy == "adaptive":
                if clip_state is None:
                    clip_state = {}
                if "running" not in clip_state:
                    clip_state["running"] = clip_norm_current
                clip_state["running"] = (
                    float(clip_alpha) * float(clip_state["running"]) +
                    (1.0 - float(clip_alpha)) * clip_norm_current
                )
                clip_norm_eff = float(clip_state["running"])
            else:
                clip_norm_eff = clip_norm_current
                if clip_state is not None:
                    clip_state["running"] = clip_norm_eff
            clip_norm_eff = max(float(clip_min), min(clip_norm_eff, float(clip_max)))
            if clip_state is not None:
                clip_state["running"] = clip_norm_eff
            if debug:
                print("[ADAPTIVE CLIP][DP-SGD]")
                print("  quantile clip:", clip_norm_current)
                print("  running clip:", clip_norm_eff)
                print("  min norm:", min(raw_norms))
                print("  max norm:", max(raw_norms))

        for flat_grad in flat_grads:
            clipped_grad, _raw_norm, clipped_norm, clip_factor = clip_update(
                flat_grad,
                clip_norm=clip_norm_eff,
                norm_type=norm_type,
            )
            assert_finite_tensor(f"{debug_prefix}grad_i_after_clipping", clipped_grad)
            clipped_grads.append(clipped_grad)
            clipped_norms.append(clipped_norm)
            clip_factors.append(clip_factor)

        stacked_grads = torch.stack(clipped_grads, dim=0)
        averaged_grad = stacked_grads.mean(dim=0)
        assert_finite_tensor(f"{debug_prefix}averaged_clipped_gradient", averaged_grad)

        # Noise is added after averaging, so the sensitivity of the averaged
        # clipped gradient is clip_norm / batch_size. For Gaussian this is L2
        # sensitivity; for Laplace the per-example gradients were clipped in L1.
        averaged_sensitivity = float(clip_norm_eff) / float(batch_size)
        if mechanism == "gaussian":
            noise_scale = gaussian_noise_scale(epsilon=epsilon, delta=delta, clip_norm=averaged_sensitivity)
        else:
            # Laplace requires L1 sensitivity. Here each grad_i is L1 clipped,
            # then averaged, so sensitivity_L1 = clip_norm / batch_size.
            noise_scale = laplace_l1_noise_scale(epsilon=epsilon, l1_sensitivity=averaged_sensitivity)
        noise = _sample_noise_like(averaged_grad, mechanism, noise_scale)
        assert_finite_tensor(f"{debug_prefix}gradient_noise", noise)
        noisy_grad = averaged_grad + noise
        assert_finite_tensor(f"{debug_prefix}gradient_after_noise", noisy_grad)

        if debug:
            step_prefix = f"{debug_prefix}[batch={batch_idx}] "
            print(f"{step_prefix}Per-example DP-SGD gradient stats:")
            print("  norm_type:", "L2" if norm_type == 2 else "L1")
            print("  batch_size:", batch_size)
            print("  clip_norm_eff:", clip_norm_eff)
            print("  raw_grad_norm_mean:", sum(raw_norms) / len(raw_norms))
            print("  raw_grad_norm_max:", max(raw_norms))
            print("  clipped_grad_norm_mean:", sum(clipped_norms) / len(clipped_norms))
            print("  clipped_grad_norm_max:", max(clipped_norms))
            print("  clip_factor_mean:", sum(clip_factors) / len(clip_factors))
            print("  averaged_sensitivity:", averaged_sensitivity)
            _print_signal_noise_stats(step_prefix, averaged_grad, noise, noise_scale)

        _write_grad_vector(noisy_grad, metadata)
        optimizer.step()

        total_loss += batch_loss_sum
        total_samples += batch_size
        grad_norm_sum += sum(raw_norms) / len(raw_norms)
        clipped_norm_sum += sum(clipped_norms) / len(clipped_norms)
        clip_factor_sum += sum(clip_factors) / len(clip_factors)
        noise_norm_sum += torch.norm(noise).item()
        noise_scale_sum += noise_scale
        clip_norm_sum += clip_norm_eff
        steps += 1

    return {
        "loss": total_loss / max(total_samples, 1),
        "raw_norm": grad_norm_sum / max(steps, 1),
        "clipped_norm": clipped_norm_sum / max(steps, 1),
        "noise_norm": noise_norm_sum / max(steps, 1),
        "noise_scale": noise_scale_sum / max(steps, 1),
        "clip_norm": clip_norm_sum / max(steps, 1),
        "clip_factor": clip_factor_sum / max(steps, 1),
        "steps": float(steps),
    }


def apply_client_dp(
    update: torch.Tensor,
    mechanism: str,
    clip_norm: float,
    epsilon: float,
    delta: float,
    debug: bool = False,
    debug_prefix: str = "",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Apply client-level DP to a flattened model update.

    Expected behavior:
      - Client-level + Gaussian can be unstable for small epsilon because the
        full model update is high-dimensional.
      - Client-level + Laplace is usually worst-case because L2 clipping implies
        an L1 bound of sqrt(d) * clip_norm, producing very large noise.
    """
    mechanism = mechanism.lower()
    if mechanism not in {"gaussian", "laplace"}:
        raise ValueError(f"Unsupported DP mechanism: {mechanism}")
    if update.numel() == 0:
        return update.clone(), {
            "raw_norm": 0.0,
            "clipped_norm": 0.0,
            "clip_factor": 1.0,
            "noise_scale": 0.0,
            "signal_norm": 0.0,
            "noise_norm": 0.0,
            "signal_noise_ratio": 0.0,
        }

    if debug:
        print(f"{debug_prefix}Before clipping:")
        debug_tensor(f"{debug_prefix}update", update)
        if update.numel() > 0 and update.max().item() > 1e3:
            print(f"{debug_prefix}WARNING: update.max() > 1e3 before clipping")
    assert_finite_tensor(f"{debug_prefix}client_update_before_clipping", update)

    d = update.numel()
    sqrt_d = math.sqrt(float(d))

    # Scale the clipping radius with sqrt(d) to restore signal magnitude
    # while keeping the full update vector and sensitivity explicit.
    clip_norm_eff = float(clip_norm) * sqrt_d
    clipped_update, raw_norm, clipped_norm, clip_factor = clip_update(
        update,
        clip_norm=clip_norm_eff,
        norm_type=2,
    )
    assert_finite_tensor(f"{debug_prefix}client_update_after_clipping", clipped_update)
    if debug:
        print(f"{debug_prefix}After clipping:")
        print("  norm:", torch.norm(clipped_update).item())
        print("  clip_factor:", clip_factor)
        print("  clip_norm_eff:", clip_norm_eff)
        print("  has_nan:", torch.isnan(clipped_update).any().item())
        print("  has_inf:", torch.isinf(clipped_update).any().item())
        if torch.norm(clipped_update).item() > clip_norm_eff + 1e-6:
            print(f"{debug_prefix}WARNING: clipped norm exceeds clip_norm_eff")

    if mechanism == "gaussian":
        # Keep the Gaussian mechanism formula intact, using the dimension-aware
        # effective L2 sensitivity.
        noise_scale = gaussian_noise_scale(epsilon=epsilon, delta=delta, clip_norm=clip_norm_eff)
    else:
        # Laplace requires L1 sensitivity. With a full d-dimensional vector
        # clipped in L2 at clip_norm_eff, the conservative L1 bound is
        # sqrt(d) * clip_norm_eff.
        noise_scale = laplace_noise_scale(epsilon=epsilon, clip_norm=clip_norm_eff, dimension=d)
    noise = _sample_noise_like(clipped_update, mechanism, noise_scale)
    assert_finite_tensor(f"{debug_prefix}client_update_noise", noise)
    signal_norm = torch.norm(clipped_update).item()
    noise_norm = torch.norm(noise).item()
    signal_noise_ratio = noise_norm / (signal_norm + 1e-6)

    noisy_update = clipped_update + noise
    assert_finite_tensor(f"{debug_prefix}client_update_after_noise", noisy_update)

    # Post-noise normalization is DP post-processing. It keeps the full update
    # structure but prevents exploding client deltas from destabilizing FedAvg.
    noisy_update = noisy_update / (torch.norm(noisy_update) + 1e-6)
    noisy_update = noisy_update * float(clip_norm)
    assert_finite_tensor(f"{debug_prefix}client_update_after_post_noise_normalization", noisy_update)

    if debug:
        _print_signal_noise_stats(debug_prefix, clipped_update, noise, noise_scale)
        print(f"{debug_prefix}After noise:")
        print("  norm:", torch.norm(noisy_update).item())
        print("  has_nan:", torch.isnan(noisy_update).any().item())
        print("  has_inf:", torch.isinf(noisy_update).any().item())
        print(f"{debug_prefix}Before sending to server:")
        debug_tensor(f"{debug_prefix}update_noisy", noisy_update)

    return torch.nan_to_num(noisy_update, nan=0.0, posinf=1e6, neginf=-1e6), {
        "raw_norm": raw_norm,
        "clipped_norm": clipped_norm,
        "clip_factor": clip_factor,
        "noise_scale": noise_scale,
        "signal_norm": signal_norm,
        "noise_norm": noise_norm,
        "signal_noise_ratio": signal_noise_ratio,
    }


def privatize_update_delta(
    delta: Dict[str, torch.Tensor],
    mechanism: str,
    epsilon: float,
    delta_value: float,
    clip_norm: float,
    debug: bool = False,
    debug_prefix: str = "",
) -> Tuple[Dict[str, torch.Tensor], float, float, float, float, float]:
    """Flatten a client delta, clip once, add noise once, and unflatten it."""
    mechanism = mechanism.lower()
    flat_delta, metadata = flatten_state_update(delta)
    if flat_delta.numel() == 0:
        empty_delta = {
            key: torch.zeros_like(value)
            for key, value in delta.items()
            if value.is_floating_point()
        }
        return empty_delta, 0.0, 0.0, 1.0, 0.0, 0.0

    noisy_flat, stats = apply_client_dp(
        flat_delta,
        mechanism=mechanism,
        clip_norm=clip_norm,
        epsilon=epsilon,
        delta=delta_value,
        debug=debug,
        debug_prefix=debug_prefix,
    )
    noisy_delta = unflatten_state_update(noisy_flat, metadata, delta)
    return (
        noisy_delta,
        stats["raw_norm"],
        stats["clipped_norm"],
        stats["clip_factor"],
        stats["noise_scale"],
        stats["noise_norm"],
        stats["signal_noise_ratio"],
    )


def clip_delta(
    delta: Dict[str, torch.Tensor],
    clip_norm: float,
) -> Dict[str, torch.Tensor]:
    """
    Bir model güncellemesini (delta) L2 normu clip_norm ile kırp.
    Tamsayılı tensörler (ör. BatchNorm num_batches_tracked) atlanır.
    """
    flat_delta, metadata = flatten_state_update(delta)
    if flat_delta.numel() == 0:
        return {k: torch.zeros_like(v) for k, v in delta.items() if v.is_floating_point()}

    total_norm = flat_delta.norm(p=2).item()
    scale = min(1.0, clip_norm / (total_norm + 1e-12))
    clipped_flat = flat_delta * scale
    return unflatten_state_update(clipped_flat, metadata, delta)


def clip_and_gaussian_noise_delta(
    delta: Dict[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
) -> Tuple[Dict[str, torch.Tensor], float, float, float, float]:
    """
    Apply full-vector L2 clipping, then iid Gaussian noise to each coordinate.

    Returns:
        noisy_delta, raw_norm, clipped_norm, clipping_factor, gaussian_std
    """
    flat_delta, metadata = flatten_state_update(delta)
    if flat_delta.numel() == 0:
        empty_delta = {
            key: torch.zeros_like(value)
            for key, value in delta.items()
            if value.is_floating_point()
        }
        return empty_delta, 0.0, 0.0, 1.0, noise_multiplier * clip_norm

    clipped_flat, raw_norm, clipped_norm, clipping_factor = clip_update(flat_delta, clip_norm, norm_type=2)
    gaussian_std = noise_multiplier * clip_norm

    if gaussian_std > 0:
        noise = torch.normal(
            mean=0.0,
            std=gaussian_std,
            size=clipped_flat.shape,
            dtype=clipped_flat.dtype,
            device=clipped_flat.device,
        )
        clipped_flat = clipped_flat + noise

    clipped_flat = torch.nan_to_num(clipped_flat, nan=0.0, posinf=1e6, neginf=-1e6)
    noisy_delta = unflatten_state_update(clipped_flat, metadata, delta)
    return noisy_delta, raw_norm, clipped_norm, clipping_factor, gaussian_std


def clip_and_laplace_noise_delta(
    delta: Dict[str, torch.Tensor],
    clip_norm: float,
    epsilon: float,
) -> Tuple[Dict[str, torch.Tensor], float, float, float, float]:
    """
    Apply full-vector L2 clipping, then iid Laplace noise to each coordinate.

    Laplace scale is set to:
        b = sqrt(d) * clip_norm / epsilon
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 for Laplace mechanism")

    flat_delta, metadata = flatten_state_update(delta)
    if flat_delta.numel() == 0:
        empty_delta = {
            key: torch.zeros_like(value)
            for key, value in delta.items()
            if value.is_floating_point()
        }
        return empty_delta, 0.0, 0.0, 1.0, 0.0

    clipped_flat, raw_norm, clipped_norm, clipping_factor = clip_update(flat_delta, clip_norm, norm_type=2)

    laplace_scale = laplace_noise_scale(epsilon, clip_norm, flat_delta.numel())
    if laplace_scale > 0:
        dist = torch.distributions.Laplace(
            loc=torch.tensor(0.0, device=clipped_flat.device, dtype=clipped_flat.dtype),
            scale=torch.tensor(laplace_scale, device=clipped_flat.device, dtype=clipped_flat.dtype),
        )
        clipped_flat = clipped_flat + dist.sample(clipped_flat.shape)

    clipped_flat = torch.nan_to_num(clipped_flat, nan=0.0, posinf=1e6, neginf=-1e6)
    noisy_delta = unflatten_state_update(clipped_flat, metadata, delta)
    return noisy_delta, raw_norm, clipped_norm, clipping_factor, laplace_scale


# ---------------------------------------------------------------------------
# 2.  Gaussian / Laplace gürültüsü
# ---------------------------------------------------------------------------


@lru_cache(maxsize=128)
def _make_opendp_gaussian_measurement(size: int, scale: float):
    """Construct an OpenDP Gaussian measurement over a fixed-size float vector."""
    if dp is None:
        raise ImportError("opendp is not installed in the active Python environment")

    dp.enable_features("contrib")
    input_domain = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=size)
    input_metric = dp.l2_distance(T=float)
    return dp.m.make_gaussian(input_domain, input_metric, scale=scale)


def _apply_opendp_gaussian_measurement(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Privatize a tensor by flattening it into a vector and invoking OpenDP's Gaussian measurement."""
    measurement = _make_opendp_gaussian_measurement(tensor.numel(), scale)
    flat = tensor.detach().cpu().to(torch.float64).reshape(-1).tolist()
    privatized = measurement(flat)
    noisy = torch.tensor(privatized, dtype=torch.float64).reshape(tensor.shape)
    return noisy.to(dtype=tensor.dtype, device=tensor.device)


def privatize_aggregate_with_opendp(
    state: Dict[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    num_clients: int,
    sensitivity_scale: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate model state'ini tek bir OpenDP Gaussian measurement üzerinden privatize et.

    Tüm floating-point parametreler tek bir vektöre düzleştirilir, measurement bir kez
    uygulanır ve sonuç tekrar orijinal parametre şekillerine dağıtılır.
    """
    if sensitivity_scale is None:
        sensitivity_scale = 1.0 / max(num_clients, 1)
    sigma = noise_multiplier * clip_norm * float(sensitivity_scale)

    float_meta = []
    flat_chunks = []
    for key, tensor in state.items():
        if tensor.is_floating_point():
            t_cpu = tensor.detach().cpu().to(torch.float64)
            float_meta.append((key, tensor.shape, tensor.dtype, tensor.device, t_cpu.numel()))
            flat_chunks.append(t_cpu.reshape(-1))

    # Floating tensor yoksa state aynen döner.
    if not flat_chunks:
        return dict(state)

    flat_all = torch.cat(flat_chunks)
    measurement = _make_opendp_gaussian_measurement(int(flat_all.numel()), sigma)
    privatized_all = measurement(flat_all.tolist())
    privatized_all_t = torch.tensor(privatized_all, dtype=torch.float64)

    noisy: Dict[str, torch.Tensor] = {}
    offset = 0
    for key, shape, dtype, device, numel in float_meta:
        piece = privatized_all_t[offset:offset + numel].reshape(shape)
        noisy[key] = torch.nan_to_num(piece.to(dtype=dtype, device=device), nan=0.0, posinf=1e6, neginf=-1e6)
        offset += numel

    for key, tensor in state.items():
        if key not in noisy:
            noisy[key] = tensor

    return noisy

def add_gaussian_noise(
    state: Dict[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    num_clients: int,
    sensitivity_scale: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate edilmiş model durumuna Gaussian gürültüsü ekle.

    Formül:
        sigma = noise_multiplier * clip_norm * sensitivity_scale

    Yani her parametrenin her öğesine N(0, sigma²) gürültüsü eklenir.
    Tamsayılı tensörler değiştirilmez.
    """
    if sensitivity_scale is None:
        sensitivity_scale = 1.0 / max(num_clients, 1)
    sigma = noise_multiplier * clip_norm * float(sensitivity_scale)
    try:
        return privatize_aggregate_with_opendp(
            state,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            num_clients=num_clients,
            sensitivity_scale=sensitivity_scale,
        )
    except Exception:
        noisy: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if v.is_floating_point():
                noise = torch.normal(
                    mean=0.0, std=sigma,
                    size=v.shape,
                    dtype=v.dtype,
                    device=v.device,
                )
                noisy[k] = torch.nan_to_num(v + noise, nan=0.0, posinf=1e6, neginf=-1e6)
            else:
                noisy[k] = v
        return noisy


def add_laplace_noise(
    state: Dict[str, torch.Tensor],
    clip_norm: float,
    epsilon_per_round: float,
    sensitivity_scale: float,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate model durumuna Laplace gürültüsü ekle.

    Not:
      - Client update'leri L2 ile clip edildiği için, L1 duyarlilik bound'u
        muhafazakar olarak sqrt(d) * clip_norm kullanilir (d: toplam float boyut).
      - Weighted FedAvg duyarlilik carpani sensitivity_scale (genelde max_i w_i) ile carpilir.

    Sonuc:
        b = (sqrt(d) * clip_norm * sensitivity_scale) / epsilon_per_round
        noise ~ Laplace(0, b)  (her koordinat icin bagimsiz)
    """
    if epsilon_per_round <= 0:
        raise ValueError("epsilon_per_round must be > 0 for Laplace mechanism")
    if sensitivity_scale <= 0:
        raise ValueError("sensitivity_scale must be > 0 for Laplace mechanism")

    dim = sum(v.numel() for v in state.values() if v.is_floating_point())
    if dim <= 0:
        return dict(state)

    l1_sensitivity = math.sqrt(float(dim)) * clip_norm * float(sensitivity_scale)
    scale_b = l1_sensitivity / float(epsilon_per_round)

    noisy: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if v.is_floating_point():
            dist = torch.distributions.Laplace(
                loc=torch.tensor(0.0, device=v.device, dtype=v.dtype),
                scale=torch.tensor(scale_b, device=v.device, dtype=v.dtype),
            )
            noise = dist.sample(v.shape)
            noisy[k] = torch.nan_to_num(v + noise, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            noisy[k] = v
    return noisy


# ---------------------------------------------------------------------------
# 3.  Privacy budget hesabı (OpenDP + RDP fallback)
# ---------------------------------------------------------------------------

def compute_epsilon(
    noise_multiplier: float,
    num_rounds: int,
    target_delta: float = 1e-5,
    num_clients: int = 1,
    sample_rate: float = 1.0,
) -> float:
    """
    Estimate the Gaussian (epsilon, delta)-DP budget with simple composition.

    Gaussian mekanizması için RDP:
        eps_rdp(alpha) = alpha / (2 * sigma²)  [tek adım, sensitivity=1]

    k adım sonrası:
        eps_rdp_total = k * alpha / (2 * sigma²)

    RDP → (eps, delta)-DP dönüşümü:
        epsilon = eps_rdp_total + log(1/delta) / (alpha - 1)

    Optimal alpha seçimi için [2, 512] aralığında grid arama yapılır.

    Parameters
    ----------
    noise_multiplier : float
        Gürültü çarpanı (sigma / sensitivity). Büyüdükçe epsilon küçülür.
    num_rounds       : int
        Toplam FL round sayısı (kompozisyon sayısı).
    target_delta     : float
        Hedef delta değeri (ör. 1e-5).
    num_clients      : int
        Currently unused by this accountant. Reserved for future client sampling logic.
    sample_rate      : float
        Currently unused by this accountant. Reserved for future subsampling logic.

    Returns
    -------
    float
        Estimated epsilon value. This is not a tight accountant for sampled FL.
    """
    # Önce OpenDP ile dene
    try:
        return _compute_epsilon_opendp(noise_multiplier, num_rounds, target_delta)
    except Exception:
        pass

    # Fallback: RDP analitik formül
    return _compute_epsilon_rdp(noise_multiplier, num_rounds, target_delta)


def compute_laplace_epsilon(
    epsilon_per_round: float,
    num_rounds: int,
) -> float:
    """
    Laplace mekanizmasi icin (epsilon, 0)-DP basic composition:
        epsilon_total = num_rounds * epsilon_per_round
    """
    if epsilon_per_round <= 0:
        raise ValueError("epsilon_per_round must be > 0")
    if num_rounds <= 0:
        return 0.0
    return float(epsilon_per_round) * float(num_rounds)


def _compute_epsilon_opendp(
    noise_multiplier: float,
    num_rounds: int,
    target_delta: float,
) -> float:
    """
    Epsilon hesabını OpenDP accounting combinator zinciriyle yap.

    Adımlar:
      1) Gaussian measurement (zCDP)
      2) zCDP -> approxDP dönüşümü: make_zCDP_to_approxDP
      3) sabit delta'da epsilon: make_fix_delta

    FL round kompozisyonu zCDP'de toplamsal olduğundan, k adımı tek-adım eşdeğeri
    Gaussian scale ile temsil ediyoruz:
        sigma_eff = sigma / sqrt(k)

    Ardından OpenDP'nin approxDP accounting modelinden epsilon çekiyoruz.
    """
    if dp is None:
        raise ImportError("opendp is not installed in the active Python environment")
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be > 0")
    if num_rounds <= 0:
        return 0.0

    dp.enable_features("contrib")

    sigma_eff = noise_multiplier / math.sqrt(float(num_rounds))
    measurement = _make_opendp_gaussian_measurement(size=1, scale=sigma_eff)

    # OpenDP accounting: zCDP -> approxDP -> fix delta
    approx_measurement = dp.c.make_zCDP_to_approxDP(measurement)
    fixed_delta_measurement = dp.c.make_fix_delta(approx_measurement, float(target_delta))
    eps, _delta = fixed_delta_measurement.map(1.0)
    return float(eps)


def _compute_epsilon_rdp(
    noise_multiplier: float,
    num_rounds: int,
    target_delta: float,
) -> float:
    """RDP analitik formülü (Mironov 2017) ile epsilon hesabı."""
    sigma = noise_multiplier
    k = num_rounds
    best_eps = float("inf")
    for alpha_int in range(2, 512):
        alpha = float(alpha_int)
        rdp = k * alpha / (2.0 * sigma ** 2)          # k adım RDP
        eps = rdp + math.log(1.0 / target_delta) / (alpha - 1.0)
        if eps < best_eps:
            best_eps = eps
    return best_eps
