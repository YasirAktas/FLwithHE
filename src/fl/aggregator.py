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
        for key in updates[0].state_dict.keys():
            if self.encryption_context:
                # Example path for encrypted aggregation (stub: just plain for now)
                weighted = sum(u.state_dict[key] * (u.num_samples / total_samples) for u in updates)
            else:
                weighted = sum(u.state_dict[key] * (u.num_samples / total_samples) for u in updates)
            new_state[key] = weighted
        global_model.load_state_dict(new_state)
