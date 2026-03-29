from typing import List, Dict
import torch

from .client import ClientUpdate

class Aggregator:
    def __init__(
        self,
        encryption_context=None,
    ):
        # encryption_context can be:
        # - None (no encryption)
        # - HomomorphicContext (CKKS, supports float scalars)
        # - PaillierContext (additive HE, integer scalars only)
        self.encryption_context = encryption_context

    def federated_average(self, updates: List[ClientUpdate], global_model: torch.nn.Module):
        if not updates:
            return
        is_model_delta = all(u.is_model_delta for u in updates)
        total_samples = sum(u.num_samples for u in updates)
        base_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
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
        if is_model_delta:
            updated_state: Dict[str, torch.Tensor] = {}
            for key, value in base_state.items():
                if key in new_state and value.is_floating_point():
                    updated_state[key] = value + new_state[key]
                else:
                    updated_state[key] = value
            new_state = updated_state
        global_model.load_state_dict(new_state)
