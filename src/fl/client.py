import time
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.he.encryption import PaillierContext
from src.privacy.dp_utils import clip_and_gaussian_noise_delta, clip_and_laplace_noise_delta

@dataclass
class ClientUpdate:
    state_dict: Dict[str, torch.Tensor]
    num_samples: int
    train_time: float = 0.0
    encrypt_time: float = 0.0
    is_model_delta: bool = False
    raw_update_norm: float = 0.0
    clipped_update_norm: float = 0.0
    clipping_factor: float = 1.0
    gaussian_std: float = 0.0
    laplace_scale: float = 0.0
    laplace_expected_noise_l2: float = 0.0

class Client:
    def __init__(self, client_id: int, dataloader: DataLoader, device: torch.device, lr: float, momentum: float = 0.9, weight_decay: float = 0.0, scheduler: str = "none", encryption_context: Optional[object] = None, dp_clip_norm: Optional[float] = None, dp_noise_multiplier: float = 0.0, dp_mechanism: str = "gaussian", dp_laplace_epsilon_per_round: float = 0.0):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.encryption_context = encryption_context
        self.dp_clip_norm = dp_clip_norm  # DP: L2 clip normu (None → DP kapalı)
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_mechanism = dp_mechanism
        self.dp_laplace_epsilon_per_round = dp_laplace_epsilon_per_round

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
        raw_update_norm = 0.0
        clipped_update_norm = 0.0
        clipping_factor = 1.0
        gaussian_std = 0.0
        laplace_scale = 0.0
        laplace_expected_noise_l2 = 0.0
        is_model_delta = False

        # DP is applied to the transmitted client update delta, not to full model states.
        if self.dp_clip_norm is not None and self.dp_mechanism in {"gaussian", "laplace"}:
            global_sd_cpu = {k: v.cpu() for k, v in global_model.state_dict().items()}
            delta = {k: sd[k] - global_sd_cpu[k] for k in global_sd_cpu if sd[k].is_floating_point()}
            if self.dp_mechanism == "gaussian":
                sd, raw_update_norm, clipped_update_norm, clipping_factor, gaussian_std = clip_and_gaussian_noise_delta(
                    delta,
                    clip_norm=self.dp_clip_norm,
                    noise_multiplier=self.dp_noise_multiplier,
                )
            else:
                sd, raw_update_norm, clipped_update_norm, clipping_factor, laplace_scale = clip_and_laplace_noise_delta(
                    delta,
                    clip_norm=self.dp_clip_norm,
                    epsilon=self.dp_laplace_epsilon_per_round,
                )
                dim = sum(v.numel() for v in sd.values() if v.is_floating_point())
                # For iid Laplace(0, b), E||noise||_2 is on the order of sqrt(2 * d) * b.
                laplace_expected_noise_l2 = math.sqrt(2.0 * float(dim)) * laplace_scale if dim > 0 else 0.0
            sd = {
                key: torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
                for key, value in sd.items()
            }
            is_model_delta = True

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
        return ClientUpdate(
            state_dict=sd,
            num_samples=len(self.dataloader.dataset),
            train_time=train_time,
            encrypt_time=encrypt_time,
            is_model_delta=is_model_delta,
            raw_update_norm=raw_update_norm,
            clipped_update_norm=clipped_update_norm,
            clipping_factor=clipping_factor,
            gaussian_std=gaussian_std,
            laplace_scale=laplace_scale,
            laplace_expected_noise_l2=laplace_expected_noise_l2,
        )

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
