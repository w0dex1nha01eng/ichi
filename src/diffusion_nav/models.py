import torch
from torch import nn


class BC_1(nn.Module):

    def __init__(self,obs_dim: int,obs_horizon: int = 2,hidden_dim: int = 256) -> None:
        super().__init__()
        for name, value in {"obs_dim": obs_dim,"obs_horizon": obs_horizon,"hidden_dim": hidden_dim}.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.hidden_dim = hidden_dim
        input_dim = obs_dim * obs_horizon
        self.network = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,2))

    def forward(self,obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3:
            raise ValueError("obs must have shape (batch, obs_horizon, obs_dim)")
        if obs.shape[1:] != (self.obs_horizon,self.obs_dim):
            raise ValueError(f"obs shape must end with ({self.obs_horizon}, {self.obs_dim})")
        if not torch.is_floating_point(obs):
            raise TypeError("obs must be a floating point tensor")
        if not torch.isfinite(obs).all():
            raise ValueError("obs must contain only finite values")

        flattened_obs = obs.reshape(obs.shape[0],self.obs_horizon * self.obs_dim)
        return self.network(flattened_obs)
