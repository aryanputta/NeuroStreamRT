"""
ShallowConvNet: shallow CNN for EEG, from Schirrmeister et al. 2017.
"Deep Learning With Convolutional Neural Networks for EEG Decoding."

Architecture:
  Conv1: temporal (40 filters, kernel 25)
  Conv2: spatial (across channels)
  Squaring activation + mean pooling + log
  Dense(n_classes)

Input: (batch, n_channels, n_samples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 19,
        n_samples: int = 512,
        n_classes: int = 3,
        n_filters: int = 40,
        filter_len: int = 25,
        pool_len: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.temporal = nn.Conv2d(1, n_filters, kernel_size=(1, filter_len), bias=False)
        self.spatial = nn.Conv2d(n_filters, n_filters, kernel_size=(n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_len), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)

        self._flatten_size = self._get_flatten_size(n_channels, n_samples)
        self.classifier = nn.Linear(self._flatten_size, n_classes)

    def _get_flatten_size(self, n_channels: int, n_samples: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.temporal(dummy)
            x = self.spatial(x)
            x = self.bn(x)
            x = x ** 2
            x = self.pool(x)
            x = torch.log(torch.clamp(x, min=1e-6))
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
