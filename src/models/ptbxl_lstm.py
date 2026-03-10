import torch
import torch.nn as nn


class PTBXL_LSTM(nn.Module):
    """Bidirectional LSTM for PTB-XL ECG classification.

    Girdi: (batch, 1000, 12)  — 1000 zaman adımı, 12 derivasyon
    Çıktı: (batch, 5)         — 5 süper sınıf
    """

    def __init__(self, input_size=12, hidden_size=64, num_layers=2,
                 num_classes=5, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 bidirectional

    def forward(self, x):
        # x: (batch, 1000, 12)
        out, _ = self.lstm(x)       # (batch, 1000, hidden*2)
        out = out[:, -1, :]         # son zaman adımı → (batch, hidden*2)
        return self.fc(out)
