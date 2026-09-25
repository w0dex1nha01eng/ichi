from pathlib import Path

import numpy as np
import pytest
import torch

from diffusion_nav.collision import Rectangle
from diffusion_nav.dataset import Normalizer
from diffusion_nav.env import NavigationEnv
from diffusion_nav.evaluate import (
    BCChunkPolicy,
    initialize_observation_history,
    load_bc_chunk_policy,
    run_chunk_policy_episode,
    update_observation_history,
)
from diffusion_nav.kinematics import Pose2D
from diffusion_nav.models import BC_Chunk
from diffusion_nav.training import save_bc_chunk_checkpoint


def create_rollout_environment(
    obstacles: list[Rectangle] | None = None,
    max_steps: int = 30,
    goal: tuple[float, float] = (2.0, 2.0),
) -> NavigationEnv:
    """Build a small environment whose termination reason is easy to control."""
    return NavigationEnv(
        width=4.0,
        height=4.0,
        robot_radius=0.15,
        start_pose=Pose2D(x=1.0, y=2.0, theta=0.0),
        goal=goal,
        obstacles=obstacles or [],
        dt=0.1,
        max_steps=max_steps,
        goal_radius=0.15,
        num_rays=8,
        lidar_max_range=3.0,
        lidar_sample_step=0.1,
        max_linear_velocity=1.0,
        max_angular_velocity=2.0,
    )


def create_constant_chunk_policy(
    obs_dim: int,
    obs_horizon: int,
    pred_horizon: int,
    action: np.ndarray,
) -> BCChunkPolicy:
    """Create a chunk policy that returns the same legal action at each horizon step."""
    model = BC_Chunk(
        obs_dim=obs_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
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
    return BCChunkPolicy(
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


def test_bc_chunk_policy_returns_denormalized_action_sequence() -> None:
    """Chunk inference should preserve both the prediction horizon and action dimensions."""
    # Arrange
    policy = create_constant_chunk_policy(
        obs_dim=5,
        obs_horizon=2,
        pred_horizon=4,
        action=np.array([0.4, -0.25], dtype=np.float32),
    )
    observation_history = np.zeros((2, 5), dtype=np.float32)

    # Act
    predicted_actions = policy.act(observation_history)

    # Assert
    assert predicted_actions.shape == (4, 2)
    assert predicted_actions.dtype == np.float32
    np.testing.assert_allclose(
        predicted_actions,
        np.tile(np.array([0.4, -0.25], dtype=np.float32), (4, 1)),
    )


def test_chunk_policy_executes_receding_action_chunks_until_success() -> None:
    """The evaluator should replan after each configured chunk and record every executed step."""
    # Arrange
    environment = create_rollout_environment()
    observation_dim = int(environment.observation_space.shape[0])
    policy = create_constant_chunk_policy(
        obs_dim=observation_dim,
        obs_horizon=2,
        pred_horizon=4,
        action=np.array([0.5, 0.0], dtype=np.float32),
    )

    # Act
    captured_status = []
    result = run_chunk_policy_episode(
        environment,
        policy,
        action_horizon=2,
        frame_callback=lambda env, status: captured_status.append((env.step_count, status)),
    )

    # Assert
    assert result.success is True
    assert result.collision is False
    assert result.truncated is False
    assert result.steps > 2
    assert result.observations.shape[0] == result.actions.shape[0]
    assert result.states.shape == (result.steps, 3)
    assert result.rewards.shape == (result.steps,)
    assert captured_status[0] == (0, "running")
    assert captured_status[-1] == (result.steps, "success")
    assert len(captured_status) == result.steps + 1


def test_pygame_renderer_returns_rgb_frame() -> None:
    """The off-screen Pygame renderer should return one RGB image for video encoding."""
    pytest.importorskip("pygame")
    from diffusion_nav.rendering import PygameRenderer

    environment = create_rollout_environment()
    renderer = PygameRenderer(size=(320, 240))

    frame = renderer.render(environment, action_horizon=2, status="running", trajectory=[(1.0, 2.0)])

    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8


def test_chunk_policy_stops_remaining_actions_after_collision() -> None:
    """A collision during an open-loop chunk should prevent later predicted actions from running."""
    # Arrange
    obstacle = Rectangle(xmin=1.35, ymin=1.5, xmax=1.60, ymax=2.5)
    environment = create_rollout_environment(obstacles=[obstacle], goal=(3.5, 2.0))
    observation_dim = int(environment.observation_space.shape[0])
    policy = create_constant_chunk_policy(
        obs_dim=observation_dim,
        obs_horizon=2,
        pred_horizon=5,
        action=np.array([1.0, 0.0], dtype=np.float32),
    )

    # Act
    result = run_chunk_policy_episode(environment, policy, action_horizon=5)

    # Assert
    assert result.success is False
    assert result.collision is True
    assert result.truncated is False
    assert result.steps == 2


def test_chunk_policy_reports_truncation_inside_an_action_chunk() -> None:
    """The environment step limit should stop chunk execution at the exact final step."""
    # Arrange
    environment = create_rollout_environment(max_steps=3, goal=(3.5, 2.0))
    observation_dim = int(environment.observation_space.shape[0])
    policy = create_constant_chunk_policy(
        obs_dim=observation_dim,
        obs_horizon=2,
        pred_horizon=4,
        action=np.array([0.0, 0.0], dtype=np.float32),
    )

    # Act
    result = run_chunk_policy_episode(environment, policy, action_horizon=4)

    # Assert
    assert result.success is False
    assert result.collision is False
    assert result.truncated is True
    assert result.steps == environment.max_steps


def test_chunk_checkpoint_loads_as_chunk_policy(tmp_path: Path) -> None:
    """Saved Gate7 metadata should rebuild the full-horizon inference policy."""
    # Arrange
    source_policy = create_constant_chunk_policy(
        obs_dim=4,
        obs_horizon=2,
        pred_horizon=3,
        action=np.array([0.35, -0.15], dtype=np.float32),
    )
    optimizer = torch.optim.AdamW(source_policy.model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "bc_chunk.pt"
    save_bc_chunk_checkpoint(
        checkpoint_path=checkpoint_path,
        model=source_policy.model,
        optimizer=optimizer,
        normalizer=source_policy.normalizer,
        epoch=2,
        metrics={"train_loss": 0.1, "validation_loss": 0.2},
    )

    # Act
    loaded_policy = load_bc_chunk_policy(
        checkpoint_path=checkpoint_path,
        action_low=np.array([0.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        device="cpu",
    )
    predicted_actions = loaded_policy.act(np.zeros((2, 4), dtype=np.float32))

    # Assert
    assert isinstance(loaded_policy, BCChunkPolicy)
    assert predicted_actions.shape == (3, 2)
    np.testing.assert_allclose(
        predicted_actions,
        np.tile(np.array([0.35, -0.15], dtype=np.float32), (3, 1)),
        atol=1e-6,
    )
