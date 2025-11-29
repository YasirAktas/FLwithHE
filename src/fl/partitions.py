import random
from typing import List, Dict

from torch.utils.data import Dataset
import numpy as np

__all__ = ["iid_partitions", "dirichlet_partitions"]

def iid_partitions(dataset: Dataset, num_clients: int) -> List[List[int]]:
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    part_size = len(indices) // num_clients
    parts = [indices[i * part_size:(i + 1) * part_size] for i in range(num_clients)]
    remainder_start = num_clients * part_size
    if remainder_start < len(indices):
        parts[-1].extend(indices[remainder_start:])
    return parts


def dirichlet_partitions(dataset: Dataset, num_clients: int, alpha: float = 0.5) -> List[List[int]]:
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = labels.max() + 1
    class_indices: Dict[int, List[int]] = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
    for c in class_indices:
        random.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]
    # For each class, sample a proportion vector from Dirichlet
    for c in range(num_classes):
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        c_indices = class_indices[c]
        split_points = (np.cumsum(proportions) * len(c_indices)).astype(int)[:-1]
        splits = np.split(c_indices, split_points)
        for cid, split in enumerate(splits):
            client_indices[cid].extend(split.tolist())
    for cid in range(num_clients):
        random.shuffle(client_indices[cid])
    return client_indices
