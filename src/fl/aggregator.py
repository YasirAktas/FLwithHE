from typing import List, Dict
import torch

from .client import ClientUpdate
from src.privacy.dp_utils import privatize_aggregate_with_opendp


class Aggregator:
    def __init__(self, encryption_context=None, dp_clip_norm: float = 1.0, dp_noise_multiplier: float = 0.0):
        # encryption_context can be:
        # - None (no encryption)
        # - HomomorphicContext (CKKS, supports float scalars)
        # - PaillierContext (additive HE, integer scalars only)
        self.encryption_context = encryption_context
        # DP: dp_noise_multiplier > 0 → DP aktif
        self.dp_clip_norm = dp_clip_norm
        self.dp_noise_multiplier = dp_noise_multiplier

    def federated_average(self, updates: List[ClientUpdate], global_model: torch.nn.Module):
        if not updates:
            return
        total_samples = sum(u.num_samples for u in updates)
        new_state: Dict[str, torch.Tensor] = {}
        for key in updates[0].state_dict.keys():
            if self.encryption_context:
                # Mixed mode support: some parameters may be encrypted (e.g.,
                # final layer with Paillier), while others remain plaintext.
                first_val = updates[0].state_dict[key]

                # Paillier (integer-scalar) context: only aggregate
                # homomorphically when the parameter is actually encrypted;
                # otherwise fall back to standard plaintext FedAvg.
                if getattr(self.encryption_context, "scalar_mode", None) == "int":
                    enc_type = getattr(self.encryption_context, "EncryptedTensor", None)
                    is_encrypted = enc_type is not None and isinstance(first_val, enc_type)

                    if is_encrypted:
                        acc = None
                        for u in updates:
                            part = self.encryption_context.mul_scalar(u.state_dict[key], u.num_samples)
                            acc = part if acc is None else self.encryption_context.add(acc, part)
                        decrypted_sum = self.encryption_context.decrypt(acc)
                        new_state[key] = decrypted_sum / float(total_samples)
                    else:
                        weighted = sum(u.state_dict[key] * (u.num_samples / total_samples) for u in updates)
                        new_state[key] = weighted
                else:
                    # CKKS / full HE: assume all parameters are encrypted.
                    acc = None
                    for u in updates:
                        w = u.num_samples / total_samples
                        # Assume client updates are already encrypted; apply scalar weight then add.
                        part = self.encryption_context.mul_scalar(u.state_dict[key], w)
                        acc = part if acc is None else self.encryption_context.add(acc, part)
                    new_state[key] = self.encryption_context.decrypt(acc)
            else:
                weighted = sum(u.state_dict[key] * (u.num_samples / total_samples) for u in updates)
                new_state[key] = weighted
        # DP: aggregate sonucu OpenDP Gaussian measurement'dan geçirilir.
        # sigma = noise_multiplier * clip_norm / num_clients
        if self.dp_noise_multiplier > 0.0:
            new_state = privatize_aggregate_with_opendp(
                new_state,
                clip_norm=self.dp_clip_norm,
                noise_multiplier=self.dp_noise_multiplier,
                num_clients=len(updates),
            )
        global_model.load_state_dict(new_state)
