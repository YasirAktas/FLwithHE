from dataclasses import dataclass
from typing import Dict,Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

@dataclass
@dataclass
class ClientUpdate:
    plain_state: Dict[str, torch.Tensor]   # plaintext parametreler
    enc_state: Dict[str, Any]              # şifreli parametreler (ciphertext objeleri)
    num_samples: int

ENCRYPT_PREFIXES = ("model.layer4.", "model.fc.")

def should_encrypt(name: str) -> bool:
    return name.startswith(ENCRYPT_PREFIXES)

class Client:
    def __init__(self, client_id: int, dataloader: DataLoader, device: torch.device, lr: float, momentum: float = 0.9, weight_decay: float = 0.0, scheduler: str = "none"):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler

    def train(self, global_model: nn.Module, epochs: int,  encryption_context=None) -> ClientUpdate:
        model_local = type(global_model)()  # reinstantiate architecture
        model_local.load_state_dict(global_model.state_dict())
        model_local.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_local.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        if self.scheduler == "step":
            sched = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        elif self.scheduler == "cosine":
            sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            sched = None
        model_local.train()
        for _ in range(epochs):
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model_local(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            if sched is not None:
                sched.step()
        state = {k: v.detach().cpu() for k, v in model_local.state_dict().items()}

        plain_state = {}
        enc_state = {}

        for k, v in state.items():
            if should_encrypt(k) and encryption_context is not None:
                enc_state[k] = encryption_context.encrypt(v)
            else:
                plain_state[k] = v

        return ClientUpdate(
            plain_state=plain_state,
            enc_state=enc_state,
            num_samples=len(self.dataloader.dataset),
        )
