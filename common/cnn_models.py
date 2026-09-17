from __future__ import annotations

import torch
from torch import nn

torch.set_num_threads(1)


class CNN1D(nn.Module):
    """Shared 1D CNN for tabular sequences and token/feature sequences."""

    def __init__(self, input_length: int, output_dim: int, architecture: str, task: str):
        super().__init__()
        if architecture not in {"3-layer", "5-layer"}:
            raise ValueError("architecture must be '3-layer' or '5-layer'")
        self.architecture = architecture
        self.task = task
        layers: list[nn.Module] = [nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU()]
        if architecture == "3-layer":
            layers += [nn.Conv1d(16, 8, kernel_size=3, padding=1), nn.ReLU()]
        else:
            layers += [
                nn.Conv1d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 8, kernel_size=3, padding=1),
                nn.ReLU(),
            ]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(8, output_dim)
        self.input_length = input_length

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            values = values.unsqueeze(1)
        values = self.features(values)
        values = self.pool(values).flatten(1)
        output = self.output(values)
        if self.task == "binary":
            return torch.sigmoid(output)
        return output


def build_cnn(input_length: int, output_dim: int, architecture: str, task: str) -> CNN1D:
    return CNN1D(input_length, output_dim, architecture, task)