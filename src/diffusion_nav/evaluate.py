from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .dataset import Normalizer
from .env import NavigationEnv
from .models import BC_Chunk
from .training import choose_device


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    observations: np.ndarray
    actions: np.ndarray
    states: np.ndarray
    rewards: np.ndarray
    success: bool
    collision: bool
    truncated: bool

    @property
    def steps(self) -> int:
        return len(self.actions)


class BCChunkPolicy:
    def __init__(self, model: BC_Chunk, normalizer: Normalizer, action_low, action_high, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        if self.action_low.shape != (2,) or self.action_high.shape != (2,):
            raise ValueError("action bounds must have shape (2,)")
        if not np.all(np.isfinite(self.action_low)) or not np.all(np.isfinite(self.action_high)):
            raise ValueError("action bounds must be finite")
        if np.any(self.action_low >= self.action_high):
            raise ValueError("action low must be smaller than action high")
        if normalizer.observation_mean.shape != (model.obs_dim,):
            raise ValueError("normalizer observation dimension doesn't match model")
        self.model.to(self.device)
        self.model.eval()

    def reset(self) -> None:
        return None

    @torch.inference_mode()
    def act(self, observation_history) -> np.ndarray:
        observation_history = np.asarray(observation_history, dtype=np.float32)
        expected_shape = (self.model.obs_horizon, self.model.obs_dim)
        if observation_history.shape != expected_shape:
            raise ValueError(f"observation_history must have shape {expected_shape}")
        normalized_observation = self.normalizer.normalize_observation(observation_history)
        observation_tensor = torch.as_tensor(normalized_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        normalized_actions = self.model(observation_tensor).squeeze(0).cpu().numpy()
        actions = self.normalizer.denormalize_action(normalized_actions)
        if not np.all(np.isfinite(actions)):
            raise FloatingPointError("policy produced a non-finite action")
        return np.clip(actions, self.action_low, self.action_high).astype(np.float32)


def initialize_observation_history(observation, obs_horizon: int) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    if observation.ndim != 1 or observation.size == 0:
        raise ValueError("observation must be a non-empty one-dimensional array")
    if not isinstance(obs_horizon, int) or isinstance(obs_horizon, bool) or obs_horizon <= 0:
        raise ValueError("obs_horizon must be a positive integer")
    if not np.all(np.isfinite(observation)):
        raise ValueError("observation must contain only finite values")
    return np.repeat(observation[None], obs_horizon, axis=0)


def update_observation_history(observation_history, observation) -> np.ndarray:
    observation_history = np.asarray(observation_history, dtype=np.float32)
    observation = np.asarray(observation, dtype=np.float32)
    if observation_history.ndim != 2 or observation.shape != (observation_history.shape[1],):
        raise ValueError("observation doesn't match observation history")
    return np.concatenate([observation_history[1:], observation[None]], axis=0).astype(np.float32)


def run_chunk_policy_episode(env: NavigationEnv, policy: BCChunkPolicy, action_horizon: int = 1, *, frame_callback: Callable[[NavigationEnv, str], None] | None = None) -> EpisodeResult:
    if not isinstance(action_horizon, int) or isinstance(action_horizon, bool) or not 1 <= action_horizon <= policy.model.pred_horizon:
        raise ValueError("action_horizon must be between 1 and pred_horizon")

    observation, _ = env.reset()
    policy.reset()
    if frame_callback is not None:
        frame_callback(env, "running")
    observation_history = initialize_observation_history(observation, policy.model.obs_horizon)
    observations = []
    actions = []
    states = []
    rewards = []
    success = False
    collision = False
    truncated = False

    while True:
        predicted_actions = policy.act(observation_history)
        for action in predicted_actions[:action_horizon]:
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            states.append(np.array([env.pose.x, env.pose.y, env.pose.theta], dtype=np.float32))
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(env.previous_action.copy())
            rewards.append(float(reward))
            observation_history = update_observation_history(observation_history, observation)
            success = bool(info["success"])
            collision = bool(info["collision"])
            if frame_callback is not None:
                status = "success" if success else "collision" if collision else "timeout" if truncated else "running"
                frame_callback(env, status)
            if terminated or truncated:
                break
        if terminated or truncated:
            break

    return EpisodeResult(np.stack(observations).astype(np.float32), np.stack(actions).astype(np.float32), np.stack(states).astype(np.float32), np.asarray(rewards, dtype=np.float32),
        success, collision, bool(truncated))


def load_bc_chunk_policy(checkpoint_path, action_low, action_high, device: torch.device | str = "auto") -> BCChunkPolicy:
    device = choose_device(str(device)) if not isinstance(device, torch.device) else device
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for key in ("model_state_dict", "model_config", "normalizer"):
        if key not in checkpoint:
            raise ValueError(f"checkpoint is missing {key}")
    if checkpoint.get("policy_type") != "bc_chunk":
        raise ValueError("checkpoint doesn't contain a BC-Chunk policy")
    model = BC_Chunk(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    return BCChunkPolicy(model, normalizer, action_low, action_high, device)
