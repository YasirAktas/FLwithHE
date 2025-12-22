from typing import List, Dict
import torch

from .client import ClientUpdate

class Aggregator:
    def __init__(self, encryption_context=None):
        self.encryption_context = encryption_context  # can be None or object with encrypt/decrypt/add

    def federated_average(self, updates: List[ClientUpdate], global_model: torch.nn.Module):
        if not updates:
            return
        total_samples = sum(u.num_samples for u in updates)
        new_state: Dict[str, torch.Tensor] = {}

         # 1) Plain parametreler: normal FedAvg
        # Tüm client'larda aynı plain key set'i olduğunu varsayıyoruz
        plain_keys = list(updates[0].plain_state.keys())
        for key in plain_keys:
            weighted = sum(u.plain_state[key] * (u.num_samples / total_samples) for u in updates)
            new_state[key] = weighted
        
        if self.encryption_context is not None:
            enc_keys = list(updates[0].enc_state.keys())
            for key in enc_keys:
                # 2.1 ciphertext sum (ağırlıksız topluyoruz)
                csum = None
                for u in updates:
                    c = u.enc_state[key]
                    csum = c if csum is None else self.encryption_context.add(csum, c)

                # 2.2 decrypt aggregate
                summed_plain = self.encryption_context.decrypt(csum)

                # 2.3 weighted average (plaintextte)
                # Eğer tüm client sample sayıları eşitse bu: summed_plain / K
                # Genel durumda: ağırlıklandırma için client başına c * n_i gerekir.
                # Şimdilik eşit varsayım yapalım veya aşağıdaki gibi normalize et:
                avg = summed_plain / len(updates)

                new_state[key] = avg
        else:
            # encryption_context yoksa, enc_state zaten boş olmalı; bir şey yapma
            pass

        global_model.load_state_dict(new_state)

