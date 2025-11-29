from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

@dataclass
class ClientUpdate:
    state_dict: Dict[str, torch.Tensor]
    num_samples: int

class Client:
    def __init__(self, client_id: int, dataloader: DataLoader, device: torch.device, lr: float, momentum: float = 0.9):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.lr = lr
        self.momentum = momentum

    def train(self, global_model: nn.Module, epochs: int) -> ClientUpdate:
        model_local = type(global_model)()  # reinstantiate architecture
        model_local.load_state_dict(global_model.state_dict())
        model_local.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_local.parameters(), lr=self.lr, momentum=self.momentum)
        model_local.train()
        for _ in range(epochs):
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model_local(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return ClientUpdate(state_dict={k: v.cpu() for k, v in model_local.state_dict().items()}, num_samples=len(self.dataloader.dataset))
