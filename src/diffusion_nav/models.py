import math

import torch
from torch import nn


class BC_Chunk(nn.Module):
    def __init__(self, obs_dim: int, obs_horizon: int = 2, pred_horizon: int = 16, hidden_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        for name, value in {"obs_dim": obs_dim, "obs_horizon": obs_horizon, "pred_horizon": pred_horizon, "hidden_dim": hidden_dim}.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a finite number in [0, 1)")

        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)
        input_dim = obs_dim * obs_horizon
        output_dim = pred_horizon * 2
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if self.dropout > 0.0:
            layers.append(nn.Dropout(self.dropout))
        layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        if self.dropout > 0.0:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3:
            raise ValueError("obs must have shape (batch, obs_horizon, obs_dim)")
        if obs.shape[1:] != (self.obs_horizon, self.obs_dim):
            raise ValueError(f"obs shape must end with ({self.obs_horizon}, {self.obs_dim})")
        if not torch.is_floating_point(obs):
            raise TypeError("obs must be a floating point tensor")
        if not torch.isfinite(obs).all():
            raise ValueError("obs must contain only finite values")

        flattened_obs = obs.reshape(obs.shape[0], self.obs_horizon * self.obs_dim)
        action_chunk = self.network(flattened_obs)
        return action_chunk.reshape(obs.shape[0], self.pred_horizon, 2)
