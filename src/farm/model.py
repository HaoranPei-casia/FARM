"""FARM readout architecture."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FARMReadout(nn.Module):
    """Attention-pool frozen visual tokens and predict frame failure logits.

    The expected input shape is ``[..., tokens, input_width]``. A trajectory
    tensor with shape ``[time, tokens, input_width]`` therefore produces one
    logit per timestep.
    """

    def __init__(self, input_width: int = 1024, hidden_width: int = 32) -> None:
        super().__init__()
        if input_width < 1 or hidden_width < 1:
            raise ValueError("input_width and hidden_width must be positive")
        self.input_width = input_width
        self.hidden_width = hidden_width
        self.input_projection = nn.Linear(input_width, hidden_width)
        self.normalization = nn.LayerNorm(hidden_width)
        self.token_score = nn.Linear(hidden_width, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )

    def forward_features(self, tokens: Tensor) -> Tensor:
        if tokens.ndim < 2:
            raise ValueError("tokens must have shape [..., token, feature]")
        if tokens.shape[-1] != self.input_width:
            raise ValueError(
                f"expected token width {self.input_width}, got {tokens.shape[-1]}"
            )
        hidden = F.gelu(self.normalization(self.input_projection(tokens)))
        attention = torch.softmax(self.token_score(hidden).squeeze(-1), dim=-1)
        pooled = torch.sum(hidden * attention.unsqueeze(-1), dim=-2)
        return F.gelu(self.classifier[0](pooled))

    def forward_classifier(self, features: Tensor) -> Tensor:
        return self.classifier[2](features).squeeze(-1)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.forward_classifier(self.forward_features(tokens))
