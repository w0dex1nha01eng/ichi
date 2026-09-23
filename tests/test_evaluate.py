from pathlib import Path

import numpy as np
import torch
from torch import nn

from diffusion_nav.dataset import Normalizer
from diffusion_nav.env import NavigationEnv
from diffusion_nav.evaluate import (
    BCPolicy,
    initialize_observation_history,
    load_bc_policy,
    run_policy_episode,
    update_observation_history,
)
from diffusion_nav.kinematics import Pose2D
from diffusion_nav.models import BCOneStep
from diffusion_nav.training import save_bc_checkpoint


def create_constant_policy(
    obs_dim: int,
    obs_horizon: int,
    action: np.ndarray,
) -> BCPolicy:
    """Create a deterministic policy whose denormalized output equals the requested action."""
    model = BCOneStep(
        obs_dim=obs_dim,
        obs_horizon=obs_horizon,
        hidden_dim=16,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    normalizer = Normalizer(
        observation_mean=np.zeros(obs_dim, dtype=np.float32),
        observation_std=np.ones(obs_dim, dtype=np.float32),
        action_mean=np.asarray(action, dtype=np.float32),
        action_std=np.ones(2, dtype=np.float32),
    )
    return BCPolicy(
        model=model,
        normalizer=normalizer,
        action_low=np.array([0.0, -2.0], dtype=np.float32),
        action_high=np.array([1.0, 2.0], dtype=np.float32),
        device="cpu",
    )


def test_observation_history_repeats_initial_sample_and_shifts_forward() -> None:
    """History initialization and updates should match Gate5 boundary-padding semantics."""
    # Arrange
    first_observation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    next_observation = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    # Act
    initial_history = initialize_observation_history(first_observation, obs_horizon=2)
    updated_history = update_observation_history(initial_history, next_observation)

    # Assert
    np.testing.assert_allclose(
        initial_history,
        np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        updated_history,
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    )


def test_bc_policy_denormalizes_and_clips_network_output() -> None:
    """Inference must convert normalized predictions back into legal environment actions."""
    # Arrange
    model = BCOneStep(obs_dim=4, obs_horizon=2, hidden_dim=8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        final_layer = model.network[-1]
        assert isinstance(final_layer, nn.Linear)
        final_layer.bias.copy_(torch.tensor([10.0, -10.0]))

    normalizer = Normalizer(
        observation_mean=np.zeros(4, dtype=np.float32),
        observation_std=np.ones(4, dtype=np.float32),
        action_mean=np.array([0.5, 0.0], dtype=np.float32),
        action_std=np.array([0.25, 0.5], dtype=np.float32),
    )
    policy = BCPolicy(
        model=model,
        normalizer=normalizer,
        action_low=np.array([0.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
    )
    observation_history = np.zeros((2, 4), dtype=np.float32)

    # Act
    action = policy.act(observation_history)

    # Assert
    np.testing.assert_allclose(action, np.array([1.0, -1.0], dtype=np.float32))
    assert action.dtype == np.float32


def test_policy_rollout_reaches_nearby_goal_without_obstacles() -> None:
    """A constant straight-driving BC policy should complete a simple closed-loop rollout."""
    # Arrange
    environment = NavigationEnv(
        width=4.0,
        height=4.0,
        robot_radius=0.15,
        start_pose=Pose2D(x=1.0, y=2.0, theta=0.0),
        goal=(2.0, 2.0),
        obstacles=[],
        dt=0.1,
        max_steps=30,
        goal_radius=0.15,
        num_rays=8,
        lidar_max_range=3.0,
        lidar_sample_step=0.1,
        max_linear_velocity=1.0,
        max_angular_velocity=2.0,
    )
    observation_dim = int(environment.observation_space.shape[0])
    policy = create_constant_policy(
        obs_dim=observation_dim,
        obs_horizon=2,
        action=np.array([0.5, 0.0], dtype=np.float32),
    )

    # Act
    result = run_policy_episode(environment, policy)

    # Assert
    assert result.success is True
    assert result.collision is False
    assert result.truncated is False
    assert 1 <= result.steps < environment.max_steps
    assert result.observations.shape == (result.steps, observation_dim)
    assert result.actions.shape == (result.steps, 2)
    assert result.states.shape == (result.steps, 3)
    assert result.rewards.shape == (result.steps,)


def test_load_bc_policy_reconstructs_model_and_normalizer(tmp_path: Path) -> None:
    """A Gate6 checkpoint should be directly usable by the evaluation policy loader."""
    # Arrange
    source_policy = create_constant_policy(
        obs_dim=4,
        obs_horizon=2,
        action=np.array([0.4, -0.2], dtype=np.float32),
    )
    optimizer = torch.optim.AdamW(source_policy.model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "bc_policy.pt"
    save_bc_checkpoint(
        checkpoint_path=checkpoint_path,
        model=source_policy.model,
        optimizer=optimizer,
        normalizer=source_policy.normalizer,
        epoch=1,
        metrics={"train_loss": 0.0, "validation_loss": 0.0},
    )

    # Act
    loaded_policy = load_bc_policy(
        checkpoint_path=checkpoint_path,
        action_low=np.array([0.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        device="cpu",
    )
    loaded_action = loaded_policy.act(np.zeros((2, 4), dtype=np.float32))

    # Assert
    np.testing.assert_allclose(
        loaded_action,
        np.array([0.4, -0.2], dtype=np.float32),
        atol=1e-6,
    )
