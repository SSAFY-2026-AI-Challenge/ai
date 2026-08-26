from __future__ import annotations

import torch
from torch import nn


class CreditMLP(nn.Module):
    """출결·숙제·소비 피처를 300~1000점 점수로 사상하는 다층 퍼셉트론."""

    def __init__(
        self,
        in_dim: int = 3,
        hidden: tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(prev, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features)
        unit = torch.sigmoid(self.head(hidden))
        return 300.0 + 700.0 * unit
