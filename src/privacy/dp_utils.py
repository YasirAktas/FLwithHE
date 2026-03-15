"""
Differential Privacy utilities for Federated Learning.

Akış (FL + HE + DP):
  1. Client: delta = local_weights - global_weights
  2. Client: clipped_delta = L2-clip(delta, clip_norm)      ← sensitivity kontrolü
  3. Client: local_weights = global_weights + clipped_delta
  4. Client: HE ile şifrele (opsiyonel)
  5. Server: HE ile aggregate et, decrypt et
  6. Server: noisy_avg = avg + Gaussian(0, sigma²)          ← DP gürültüsü
         sigma = noise_multiplier * clip_norm / num_clients
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Dict

import torch

try:
    import opendp.prelude as dp
except ImportError:
    dp = None


# ---------------------------------------------------------------------------
# 1.  L2 clipping
# ---------------------------------------------------------------------------

def clip_delta(
    delta: Dict[str, torch.Tensor],
    clip_norm: float,
) -> Dict[str, torch.Tensor]:
    """
    Bir model güncellemesini (delta) L2 normu clip_norm ile kırp.
    Tamsayılı tensörler (ör. BatchNorm num_batches_tracked) atlanır.
    """
    total_sq = sum(
        v.double().pow(2).sum().item()
        for v in delta.values()
        if v.is_floating_point()
    )
    total_norm = math.sqrt(total_sq) + 1e-12
    scale = min(1.0, clip_norm / total_norm)
    return {
        k: (v * scale if v.is_floating_point() else v)
        for k, v in delta.items()
    }


# ---------------------------------------------------------------------------
# 2.  Gaussian gürültüsü
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
) -> Dict[str, torch.Tensor]:
    """
    Aggregate model state'ini tek bir OpenDP Gaussian measurement üzerinden privatize et.

    Tüm floating-point parametreler tek bir vektöre düzleştirilir, measurement bir kez
    uygulanır ve sonuç tekrar orijinal parametre şekillerine dağıtılır.
    """
    sigma = noise_multiplier * clip_norm / max(num_clients, 1)

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
) -> Dict[str, torch.Tensor]:
    """
    Aggregate edilmiş model durumuna Gaussian gürültüsü ekle.

    Formül (uniform ortalama varsayımı):
        sigma = noise_multiplier * clip_norm / num_clients

    Yani her parametrenin her öğesine N(0, sigma²) gürültüsü eklenir.
    Tamsayılı tensörler değiştirilmez.
    """
    sigma = noise_multiplier * clip_norm / max(num_clients, 1)
    try:
        return privatize_aggregate_with_opendp(
            state,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            num_clients=num_clients,
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
    (epsilon, delta)-DP bütçesini Renyi DP (RDP) kompozisyonu ile hesapla.

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
        Kullanılmaz (ileride Poisson örnekleme için rezerve edilmiş).
    sample_rate      : float
        Kullanılmaz (ileride subsampling için rezerve edilmiş).

    Returns
    -------
    float
        Tahmini epsilon değeri.
    """
    # Önce OpenDP ile dene
    try:
        return _compute_epsilon_opendp(noise_multiplier, num_rounds, target_delta)
    except Exception:
        pass

    # Fallback: RDP analitik formül
    return _compute_epsilon_rdp(noise_multiplier, num_rounds, target_delta)


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
