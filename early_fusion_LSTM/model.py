import torch
import torch.nn as nn


class EarlyFusionLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, T, D]
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]               # [B, hidden_dim]
        logits = self.classifier(self.dropout(last_hidden))
        return logits