"""Differential Privacy utilities for Federated Learning."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Dict, List, Optional, Tuple

import torch

try:
    import opendp.prelude as dp
except ImportError:
    dp = None


# ---------------------------------------------------------------------------
# 1.  L2 clipping
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
            key: torch.nan_to_num(value.clone(), nan=0.0, posinf=1e6, neginf=-1e6)
            for key, value in delta.items()
            if value.is_floating_point()
        }
        return empty_delta, 0.0, 0.0, 1.0, noise_multiplier * clip_norm

    raw_norm = flat_delta.norm(p=2).item()
    clipping_factor = min(1.0, clip_norm / (raw_norm + 1e-12))
    clipped_flat = flat_delta * clipping_factor
    clipped_norm = clipped_flat.norm(p=2).item()
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
        b = clip_norm / epsilon
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 for Laplace mechanism")

    flat_delta, metadata = flatten_state_update(delta)
    if flat_delta.numel() == 0:
        empty_delta = {
            key: torch.nan_to_num(value.clone(), nan=0.0, posinf=1e6, neginf=-1e6)
            for key, value in delta.items()
            if value.is_floating_point()
        }
        return empty_delta, 0.0, 0.0, 1.0, 0.0

    raw_norm = flat_delta.norm(p=2).item()
    clipping_factor = min(1.0, clip_norm / (raw_norm + 1e-12))
    clipped_flat = flat_delta * clipping_factor
    clipped_norm = clipped_flat.norm(p=2).item()

    laplace_scale = clip_norm / float(epsilon)
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
