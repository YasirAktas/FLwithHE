import time
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.he.encryption import PaillierContext
from src.privacy.dp_utils import apply_dp_sgd

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
    noise_scale: float = 0.0
    noise_norm: float = 0.0
    signal_noise_ratio: float = 0.0
    laplace_expected_noise_l2: float = 0.0

class Client:
    def __init__(self, client_id: int, dataloader: DataLoader, device: torch.device, lr: float, momentum: float = 0.9, weight_decay: float = 0.0, scheduler: str = "none", encryption_context: Optional[object] = None, dp_clip_norm: Optional[float] = None, dp_mechanism: str = "gaussian", dp_epsilon: float = 0.0, dp_delta: float = 1e-5, dp_debug: bool = False, dp_clip_strategy: str = "fixed", dp_clip_quantile: float = 50.0, dp_clip_alpha: float = 0.9, dp_clip_min: float = 0.1, dp_clip_max: float = 10.0):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.encryption_context = encryption_context
        self.dp_clip_norm = dp_clip_norm  # DP: L2 clip normu (None → DP kapalı)
        self.dp_mechanism = dp_mechanism
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_debug = dp_debug
        self.dp_clip_strategy = dp_clip_strategy
        self.dp_clip_quantile = dp_clip_quantile
        self.dp_clip_alpha = dp_clip_alpha
        self.dp_clip_min = dp_clip_min
        self.dp_clip_max = dp_clip_max
        self.dp_sgd_clip_state: Dict[str, float] = {}

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
        dp_sgd_stats = {"raw_norm": 0.0, "clipped_norm": 0.0, "noise_norm": 0.0, "noise_scale": 0.0, "clip_norm": 0.0, "clip_factor": 1.0, "steps": 0.0}
        use_dp_sgd = self.dp_clip_norm is not None
        for epoch_idx in range(epochs):
            if use_dp_sgd:
                epoch_stats = apply_dp_sgd(
                    model_local,
                    self.dataloader,
                    optimizer,
                    mechanism=self.dp_mechanism,
                    clip_norm=self.dp_clip_norm,
                    epsilon=self.dp_epsilon,
                    delta=self.dp_delta,
                    criterion=criterion,
                    device=self.device,
                    debug=self.dp_debug,
                    debug_prefix=f"[DP DEBUG][client={self.client_id}][epoch={epoch_idx + 1}] ",
                    clip_strategy=self.dp_clip_strategy,
                    clip_quantile=self.dp_clip_quantile,
                    clip_alpha=self.dp_clip_alpha,
                    clip_min=self.dp_clip_min,
                    clip_max=self.dp_clip_max,
                    clip_state=self.dp_sgd_clip_state,
                )
                steps = epoch_stats["steps"]
                old_steps = dp_sgd_stats["steps"]
                total_steps = old_steps + steps
                if total_steps > 0:
                    for key in ["raw_norm", "clipped_norm", "noise_norm", "noise_scale", "clip_norm", "clip_factor"]:
                        dp_sgd_stats[key] = (
                            dp_sgd_stats[key] * old_steps + epoch_stats[key] * steps
                        ) / total_steps
                    dp_sgd_stats["steps"] = total_steps
            else:
                batch_idx = 0
                for images, labels in self.dataloader:
                    batch_idx += 1
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model_local(images)
                    loss = criterion(outputs, labels)
                    if self.dp_debug:
                        if torch.isnan(loss).item():
                            print(f"[DP DEBUG][client={self.client_id}][epoch={epoch_idx + 1}][batch={batch_idx}] LOSS IS NAN")
                            raise SystemExit(1)
                        if torch.isinf(loss).item():
                            print(f"[DP DEBUG][client={self.client_id}][epoch={epoch_idx + 1}][batch={batch_idx}] LOSS IS INF")
                            raise SystemExit(1)
                        if loss.item() > 1e5:
                            print(
                                f"[DP DEBUG][client={self.client_id}][epoch={epoch_idx + 1}][batch={batch_idx}] "
                                f"WARNING: loss explosion loss={loss.item():.6f}"
                            )
                    loss.backward()
                    if self.dp_debug:
                        total_grad_norm_sq = 0.0
                        has_bad_grad = False
                        for name, param in model_local.named_parameters():
                            if param.grad is None:
                                continue
                            grad = param.grad.detach()
                            if torch.isnan(grad).any().item() or torch.isinf(grad).any().item():
                                print(
                                    f"[DP DEBUG][client={self.client_id}][epoch={epoch_idx + 1}][batch={batch_idx}] "
                                    f"BAD GRADIENT in {name}: has_nan={torch.isnan(grad).any().item()} has_inf={torch.isinf(grad).any().item()}"
                                )
                                has_bad_grad = True
                            total_grad_norm_sq += grad.norm(p=2).item() ** 2
                        total_grad_norm = math.sqrt(total_grad_norm_sq)
                        if total_grad_norm > 1e3:
                            print(
                                f"[DP DEBUG][client={self.client_id}][epoch={epoch_idx + 1}][batch={batch_idx}] "
                                f"WARNING: gradient norm is large grad_norm={total_grad_norm:.6f}"
                            )
                        if has_bad_grad:
                            raise SystemExit(1)
                    optimizer.step()
            if sched is not None:
                sched.step()
        train_time = time.time() - train_start
        sd = {k: v.detach().clone() for k, v in model_local.state_dict().items()}
        raw_update_norm = 0.0
        clipped_update_norm = 0.0
        clipping_factor = 1.0
        gaussian_std = 0.0
        laplace_scale = 0.0
        noise_scale = 0.0
        noise_norm = 0.0
        signal_noise_ratio = 0.0
        laplace_expected_noise_l2 = 0.0
        is_model_delta = False

        if use_dp_sgd:
            raw_update_norm = dp_sgd_stats["raw_norm"]
            clipped_update_norm = dp_sgd_stats["clipped_norm"]
            clipping_factor = dp_sgd_stats["clip_factor"]
            noise_norm = dp_sgd_stats["noise_norm"]
            noise_scale = dp_sgd_stats["noise_scale"]
            signal_noise_ratio = noise_norm / (clipped_update_norm + 1e-12)
            if self.dp_mechanism == "gaussian":
                gaussian_std = noise_scale
            else:
                laplace_scale = noise_scale

        sd = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in sd.items()}

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
            noise_scale=noise_scale,
            noise_norm=noise_norm,
            signal_noise_ratio=signal_noise_ratio,
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
