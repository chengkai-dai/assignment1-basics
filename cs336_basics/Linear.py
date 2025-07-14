import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(
            out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.randn(
            out_features, device=device, dtype=dtype))
        self._initialize_weights()

    def _initialize_weights(self):
        fan_in = self.in_features
        fan_out = self.out_features
        std = math.sqrt(2.0 / (fan_in + fan_out))

        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std
        )

        torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input @ self.weight.T + self.bias

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward(input)

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
