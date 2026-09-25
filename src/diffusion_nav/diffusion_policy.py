import math

import numpy as np
import torch
from torch import nn

from .dataset import Normalizer
from .evaluate import EpisodeResult, run_chunk_policy_episode
from .training import choose_device


class DiffusionSchedule(nn.Module):
    def __init__(self, diffusion_steps: int, schedule_type: str = "cosine", beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        if not isinstance(diffusion_steps, int) or isinstance(diffusion_steps, bool):
            raise TypeError("diffusion_steps must be an integer")
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2")
        if schedule_type not in {"cosine", "linear"}:
            raise ValueError("schedule_type must be cosine or linear")
        if not all(math.isfinite(value) for value in (beta_start, beta_end)):
            raise ValueError("beta bounds must be finite")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("beta bounds must satisfy 0 < beta_start < beta_end < 1")

        if schedule_type == "cosine":
            offset = 0.008
            positions = torch.linspace(0, diffusion_steps, diffusion_steps + 1, dtype=torch.float32)
            alpha_bars = torch.cos(((positions / diffusion_steps + offset) / (1.0 + offset)) * math.pi / 2.0).square()
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
            betas = betas.clamp(min=1e-6, max=0.999)
        else:
            betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous_alpha_bars = torch.cat((torch.ones(1, dtype=torch.float32), alpha_bars[:-1]))
        posterior_variance = betas * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)

        self.diffusion_steps = diffusion_steps
        self.schedule_type = schedule_type
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool):
            raise TypeError("embedding_dim must be an integer")
        if embedding_dim < 4:
            raise ValueError("embedding_dim must be at least 4")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape (batch,)")
        half_dim = self.embedding_dim // 2
        denominator = max(half_dim - 1, 1)
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / denominator)
        angles = timesteps.to(torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if embedding.shape[1] < self.embedding_dim:
            embedding = nn.functional.pad(embedding, (0, self.embedding_dim - embedding.shape[1]))
        return embedding


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, obs_dim: int, obs_horizon: int = 8, pred_horizon: int = 32, hidden_dim: int = 256, time_embedding_dim: int = 64,
                 diffusion_steps: int = 50, schedule_type: str = "cosine", beta_start: float = 1e-4,
                 beta_end: float = 2e-2, dropout: float = 0.1) -> None:
        super().__init__()
        integer_parameters = {"obs_dim": obs_dim, "obs_horizon": obs_horizon, "pred_horizon": pred_horizon, "hidden_dim": hidden_dim, "time_embedding_dim": time_embedding_dim}
        for name, value in integer_parameters.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a number")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a finite number in [0, 1)")

        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim
        self.diffusion_steps = diffusion_steps
        self.schedule_type = schedule_type
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.dropout = float(dropout)

        observation_input_dim = obs_dim * obs_horizon
        action_input_dim = pred_horizon * 2
        self.schedule = DiffusionSchedule(diffusion_steps=diffusion_steps, schedule_type=schedule_type, beta_start=beta_start, beta_end=beta_end)
        self.observation_encoder = nn.Sequential(nn.Linear(observation_input_dim, hidden_dim), nn.SiLU(), nn.Dropout(self.dropout))
        self.action_encoder = nn.Sequential(nn.Linear(action_input_dim, hidden_dim), nn.SiLU())
        self.time_encoder = nn.Sequential(SinusoidalTimeEmbedding(time_embedding_dim), nn.Linear(time_embedding_dim, hidden_dim), nn.SiLU())
        self.noise_predictor = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Dropout(self.dropout), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, action_input_dim))

    def config_dict(self) -> dict[str, int | float | str]:
        return {"obs_dim": self.obs_dim, "obs_horizon": self.obs_horizon, "pred_horizon": self.pred_horizon, "hidden_dim": self.hidden_dim,
                "time_embedding_dim": self.time_embedding_dim, "diffusion_steps": self.diffusion_steps, "schedule_type": self.schedule_type,
                "beta_start": self.beta_start, "beta_end": self.beta_end, "dropout": self.dropout}

    def forward(self, observations: torch.Tensor, noisy_actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        expected_observation_shape = (self.obs_horizon, self.obs_dim)
        expected_action_shape = (self.pred_horizon, 2)
        if observations.ndim != 3 or observations.shape[1:] != expected_observation_shape:
            raise ValueError(f"observations must have shape (batch, {self.obs_horizon}, {self.obs_dim})")
        if noisy_actions.ndim != 3 or noisy_actions.shape[1:] != expected_action_shape:
            raise ValueError(f"noisy_actions must have shape (batch, {self.pred_horizon}, 2)")
        if observations.shape[0] != noisy_actions.shape[0]:
            raise ValueError("observations and noisy_actions must use the same batch size")
        if timesteps.shape != (observations.shape[0],):
            raise ValueError("timesteps must have shape (batch,)")
        if timesteps.dtype not in (torch.int32, torch.int64):
            raise TypeError("timesteps must contain integers")
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.diffusion_steps):
            raise ValueError("timesteps are outside the diffusion schedule")
        if not torch.is_floating_point(observations) or not torch.is_floating_point(noisy_actions):
            raise TypeError("observations and noisy_actions must be floating point tensors")
        if not torch.isfinite(observations).all() or not torch.isfinite(noisy_actions).all():
            raise ValueError("model inputs must contain only finite values")

        observation_features = self.observation_encoder(observations.flatten(start_dim=1))
        action_features = self.action_encoder(noisy_actions.flatten(start_dim=1))
        time_features = self.time_encoder(timesteps)
        combined_features = torch.cat((observation_features, action_features, time_features), dim=1)
        predicted_noise = self.noise_predictor(combined_features)
        return predicted_noise.reshape(observations.shape[0], self.pred_horizon, 2)

    def add_noise(self, clean_actions: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if clean_actions.shape != noise.shape:
            raise ValueError("clean_actions and noise must have the same shape")
        if clean_actions.ndim != 3 or clean_actions.shape[1:] != (self.pred_horizon, 2):
            raise ValueError(f"clean_actions must have shape (batch, {self.pred_horizon}, 2)")
        if timesteps.shape != (clean_actions.shape[0],):
            raise ValueError("timesteps must have shape (batch,)")
        alpha_bars = self.schedule.alpha_bars[timesteps].reshape(-1, 1, 1)
        return alpha_bars.sqrt() * clean_actions + (1.0 - alpha_bars).sqrt() * noise

    def diffusion_loss(self, observations: torch.Tensor, clean_actions: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        batch_size = observations.shape[0]
        timesteps = torch.randint(0, self.diffusion_steps, (batch_size,), device=observations.device, generator=generator)
        noise = torch.randn(clean_actions.shape, device=clean_actions.device, dtype=clean_actions.dtype, generator=generator)
        noisy_actions = self.add_noise(clean_actions, timesteps, noise)
        predicted_noise = self(observations, noisy_actions, timesteps)
        return nn.functional.mse_loss(predicted_noise, noise)

    @torch.inference_mode()
    def sample(self, observations: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        if observations.ndim != 3 or observations.shape[1:] != (self.obs_horizon, self.obs_dim):
            raise ValueError(f"observations must have shape (batch, {self.obs_horizon}, {self.obs_dim})")
        actions = torch.randn((observations.shape[0], self.pred_horizon, 2), device=observations.device, dtype=observations.dtype, generator=generator)
        for step in reversed(range(self.diffusion_steps)):
            timesteps = torch.full((observations.shape[0],), step, device=observations.device, dtype=torch.long)
            predicted_noise = self(observations, actions, timesteps)
            alpha = self.schedule.alphas[step]
            alpha_bar = self.schedule.alpha_bars[step]
            beta = self.schedule.betas[step]
            mean = torch.rsqrt(alpha) * (actions - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise)
            if step > 0:
                noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
                actions = mean + self.schedule.posterior_variance[step].sqrt() * noise
            else:
                actions = mean
        if not torch.isfinite(actions).all():
            raise FloatingPointError("reverse diffusion produced non-finite actions")
        return actions


def _make_generator(device: torch.device, seed: int) -> torch.Generator | None:
    if device.type not in {"cpu", "cuda"}:
        return None
    return torch.Generator(device=device).manual_seed(seed)


class DiffusionPolicy:
    def __init__(self, model: ConditionalDiffusionModel, normalizer: Normalizer, action_low, action_high, device: torch.device | str = "cpu", seed: int = 0) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        self.model = model
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.seed = seed
        if self.action_low.shape != (2,) or self.action_high.shape != (2,):
            raise ValueError("action bounds must have shape (2,)")
        if np.any(self.action_low >= self.action_high):
            raise ValueError("action low must be smaller than action high")
        if normalizer.observation_mean.shape != (model.obs_dim,):
            raise ValueError("normalizer observation dimension doesn't match model")
        self.model.to(self.device)
        self.model.eval()
        self.generator = _make_generator(self.device, seed)

    def reset(self) -> None:
        self.generator = _make_generator(self.device, self.seed)

    @torch.inference_mode()
    def act(self, observation_history) -> np.ndarray:
        observation_history = np.asarray(observation_history, dtype=np.float32)
        expected_shape = (self.model.obs_horizon, self.model.obs_dim)
        if observation_history.shape != expected_shape:
            raise ValueError(f"observation_history must have shape {expected_shape}")
        normalized_observation = self.normalizer.normalize_observation(observation_history)
        observation_tensor = torch.as_tensor(normalized_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        normalized_actions = self.model.sample(observation_tensor, generator=self.generator).squeeze(0).cpu().numpy()
        actions = self.normalizer.denormalize_action(normalized_actions)
        return np.clip(actions, self.action_low, self.action_high).astype(np.float32)


def load_diffusion_policy(checkpoint_path, action_low, action_high, device: torch.device | str = "auto", seed: int = 0) -> DiffusionPolicy:
    device = choose_device(str(device)) if not isinstance(device, torch.device) else device
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required_keys = {"policy_type", "model_config", "model_state_dict", "normalizer"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing_keys)}")
    if checkpoint["policy_type"] != "diffusion_policy":
        raise ValueError("checkpoint doesn't contain a diffusion policy")
    model = ConditionalDiffusionModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    return DiffusionPolicy(model, normalizer, action_low, action_high, device=device, seed=seed)


def run_diffusion_policy_episode(env, policy: DiffusionPolicy, action_horizon: int = 1) -> EpisodeResult:
    return run_chunk_policy_episode(env, policy, action_horizon=action_horizon)
