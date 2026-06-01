"""
EEGNet: compact CNN for EEG classification.

From: Lawhern et al., 2018. "EEGNet: A Compact Convolutional Neural Network
for EEG-based Brain-Computer Interfaces."

Architecture:
  Block 1: Temporal conv (F1=8, kernel_size=sfreq//2) → DepthwiseConv across channels (D=2)
  Block 2: SeparableConv (F2=16, kernel_size=16) → Pointwise
  Classify: Flatten → Dense(n_classes)

Input: (batch, 1, n_channels, n_samples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 19,
        n_samples: int = 512,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.25,
        sfreq: int = 256,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        # Block 1: temporal conv
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, sfreq // 2), padding=(0, sfreq // 4 - 1), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Depthwise conv across channels
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: separable conv
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # compute flattened size
        self._flatten_size = self._get_flatten_size(n_channels, n_samples)
        self.classifier = nn.Linear(self._flatten_size, n_classes)

    def _get_flatten_size(self, n_channels: int, n_samples: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.temporal_conv(dummy)
            x = self.depthwise(x)
            x = self.separable(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, n_samples) -> add channel dim
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.temporal_conv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
