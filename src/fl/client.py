import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.he.encryption import PaillierContext

@dataclass
class ClientUpdate:
    state_dict: Dict[str, torch.Tensor]
    num_samples: int
    train_time: float = 0.0
    encrypt_time: float = 0.0

class Client:
    def __init__(self, client_id: int, dataloader: DataLoader, device: torch.device, lr: float, momentum: float = 0.9, weight_decay: float = 0.0, scheduler: str = "none", encryption_context: Optional[object] = None):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.encryption_context = encryption_context

    def train(self, global_model: nn.Module, epochs: int) -> ClientUpdate:
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
        train_start = time.time()
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
        train_time = time.time() - train_start
        sd = {k: v.cpu() for k, v in model_local.state_dict().items()}
        encrypt_time = 0.0
        if self.encryption_context is not None:
            enc_start = time.time()
            # PaillierContext: only encrypt the final classifier layer; keep
            # all other parameters in plaintext to reduce overhead.
            if isinstance(self.encryption_context, PaillierContext):
                enc_sd: Dict[str, torch.Tensor] = {}
                for name, tensor in sd.items():
                    if self._is_last_layer_param(name):
                        enc_sd[name] = self.encryption_context.encrypt(tensor)
                    else:
                        enc_sd[name] = tensor
                sd = enc_sd
            else:
                # CKKS or other contexts: encrypt all parameters (original behavior).
                sd = {k: self.encryption_context.encrypt(v) for k, v in sd.items()}
            encrypt_time = time.time() - enc_start
        return ClientUpdate(state_dict=sd, num_samples=len(self.dataloader.dataset), train_time=train_time, encrypt_time=encrypt_time)

    def _is_last_layer_param(self, name: str) -> bool:
        """Return True if this parameter belongs to the final classifier layer.

        python -m src.fl.fedavg_runner --dataset ptbxl --ptbxl_model logistic --num_clients 5 --rounds 5 --local_epochs 1        Desteklenen mimariler:
        - SimpleCNN (MNIST):        son Linear → 'classifier.3.*'
        - ResNetCIFAR10:            son Linear → 'model.fc.*'
        - PTBXL_Logistic:           tek Linear → 'linear.*'
        - PTBXL_CNN_Medium:         son Linear → 'fc2.*'
        - PTBXL_CNN_Large:          son Linear → 'fc3.*'
        """
        if name.startswith("classifier.3."):
            return True
        if name.startswith("linear."):
            return True
        if name.startswith("model.fc."):
            return True
        if name.startswith("fc."):    # PTBXL_LSTM
            return True
        if name.startswith("fc2."):   # PTBXL_CNN_Medium
            return True
        if name.startswith("fc3."):   # PTBXL_CNN_Large
            return True
        return False
