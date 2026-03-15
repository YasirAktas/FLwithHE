import torch
import torch.nn as nn


class PTBXL_Logistic(nn.Module):
    """Logistic regression baseline for PTB-XL.

    Girdi: (batch, 1000, 12)  →  düzleştirilir  →  (batch, 12000)
    Çıktı: (batch, 5)  — 5 süper sınıf
    """

    def __init__(self, input_dim: int = 12000, num_classes: int = 5):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1000, 12) veya zaten düz
        if x.dim() == 3:
            x = x.flatten(1)          # (batch, 12000)
        return self.linear(x)