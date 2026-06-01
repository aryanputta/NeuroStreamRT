"""
MLP baseline for band-power features.

Input: (batch, n_features)  where n_features = n_channels * n_bands
Output: (batch, n_classes)
"""

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
