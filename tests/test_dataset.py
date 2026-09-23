from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from diffusion_nav.collision import Rectangle
from diffusion_nav.dataset import (
    ExpertEpisode,
    Normalizer,
    SequenceDataset,
    collect_expert_episodes,
    load_expert_dataset,
    plan_expert_path,
    run_expert_episode,
    save_expert_dataset,
)
from diffusion_nav.env import NavigationEnv
from diffusion_nav.export import PurePursuitConfig, PurePursuitExpert
from diffusion_nav.kinematics import Pose2D


def create_navigation_environment(
    obstacles: list[Rectangle] | None = None,
) -> NavigationEnv:
    """Build the deterministic environment shared by expert-data tests."""
    return NavigationEnv(
        width=6.0,
        height=4.0,
        robot_radius=0.2,
        start_pose=Pose2D(x=1.0, y=2.0, theta=0.0),
        goal=(5.0, 2.0),
        obstacles=obstacles or [],
        dt=0.1,
        max_steps=100,
        goal_radius=0.2,
        num_rays=8,
        lidar_max_range=3.0,
        lidar_sample_step=0.1,
        max_linear_velocity=1.0,
        max_angular_velocity=2.0,
    )


def create_successful_episode(values: list[float], map_seed: int) -> ExpertEpisode:
    """Create a compact, valid episode with values that reveal boundary padding."""
    value_array = np.asarray(values, dtype=np.float32)
    observations = np.column_stack([value_array, value_array + 0.5]).astype(np.float32)
    actions = np.column_stack([value_array, value_array * 0.1]).astype(np.float32)
    states = np.column_stack(
        [value_array, np.zeros_like(value_array), np.zeros_like(value_array)]
    ).astype(np.float32)
    return ExpertEpisode(
        observations=observations,
        actions=actions,
        states=states,
        map_seed=map_seed,
        success=True,
        collision=False,
        truncated=False,
    )


@pytest.fixture
def expert_config() -> PurePursuitConfig:
    """Provide a conservative controller configuration for deterministic rollouts."""
    return PurePursuitConfig(
        lookahead_distance=0.5,
        linear_velocity=1.0,
        max_angular_velocity=2.0,
        goal_tolerance=0.2,
    )


@pytest.fixture
def saved_dataset_path(tmp_path: Path) -> Path:
    """Persist two episodes whose boundaries are easy to inspect."""
    episodes = [
        create_successful_episode([1.0, 2.0], map_seed=10),
        create_successful_episode([10.0, 11.0, 12.0], map_seed=11),
    ]
    dataset_path = tmp_path / "expert_data.npz"
    save_expert_dataset(episodes, dataset_path)
    return dataset_path


def test_expert_rollout_reaches_goal_with_aligned_arrays(
    expert_config: PurePursuitConfig,
) -> None:
    """A successful expert rollout should store one aligned sample per environment step."""
    # Arrange
    environment = create_navigation_environment()
    expert = PurePursuitExpert(expert_config)

    # Act
    episode = run_expert_episode(
        env=environment,
        expert=expert,
        grid_step=0.5,
        map_seed=7,
    )

    # Assert
    assert episode.success is True
    assert episode.collision is False
    assert episode.observations.shape[0] == episode.actions.shape[0]
    assert episode.actions.shape[0] == episode.states.shape[0]
    assert episode.observations.shape[1] == environment.observation_space.shape[0]
    assert episode.actions.shape[1] == 2
    assert episode.observations.dtype == np.float32
    assert episode.actions.dtype == np.float32
    assert episode.states.dtype == np.float32
    assert np.all(episode.actions >= environment.action_space.low)
    assert np.all(episode.actions <= environment.action_space.high)


def test_planning_failure_returns_empty_episode(
    expert_config: PurePursuitConfig,
) -> None:
    """An impenetrable wall should stop collection without creating invalid samples."""
    # Arrange
    wall = Rectangle(xmin=2.8, ymin=0.0, xmax=3.2, ymax=4.0)
    environment = create_navigation_environment(obstacles=[wall])
    expert = PurePursuitExpert(expert_config)

    # Act
    planned_path = plan_expert_path(environment, grid_step=0.5)
    episode = run_expert_episode(
        env=environment,
        expert=expert,
        grid_step=0.5,
        map_seed=8,
    )

    # Assert
    assert planned_path is None
    assert episode.success is False
    assert episode.observations.shape == (0, environment.observation_space.shape[0])
    assert episode.actions.shape == (0, 2)


def test_dataset_round_trip_preserves_boundaries_and_statistics(
    saved_dataset_path: Path,
) -> None:
    """Serialization should preserve episode metadata and reversible normalization."""
    # Arrange
    source_episode = create_successful_episode([1.0, 2.0], map_seed=10)

    # Act
    saved_data = load_expert_dataset(saved_dataset_path)
    sequence_dataset = SequenceDataset(
        saved_dataset_path,
        obs_horizon=2,
        pred_horizon=3,
        normalize_data=False,
    )
    normalizer = Normalizer(
        observation_mean=saved_data["observation_mean"],
        observation_std=saved_data["observation_std"],
        action_mean=saved_data["action_mean"],
        action_std=saved_data["action_std"],
    )
    reconstructed_actions = normalizer.denormalize_action(
        normalizer.normalize_action(source_episode.actions)
    )

    # Assert
    assert len(sequence_dataset) == 5
    assert saved_data["episode_ends"].tolist() == [2, 5]
    assert saved_data["map_seeds"].tolist() == [10, 11]
    np.testing.assert_allclose(
        reconstructed_actions,
        source_episode.actions,
        atol=1e-6,
    )


def test_sequence_padding_never_crosses_episode_boundary(
    saved_dataset_path: Path,
) -> None:
    """Padding should repeat edge samples instead of leaking adjacent episodes."""
    # Arrange
    sequence_dataset = SequenceDataset(
        saved_dataset_path,
        obs_horizon=2,
        pred_horizon=3,
        normalize_data=False,
    )

    # Act
    first_sample = sequence_dataset[0]
    second_episode_first_sample = sequence_dataset[2]

    # Assert
    np.testing.assert_allclose(first_sample["obs"][:, 0], [1.0, 1.0])
    np.testing.assert_allclose(first_sample["action"][:, 0], [1.0, 2.0, 2.0])
    np.testing.assert_allclose(second_episode_first_sample["obs"][:, 0], [10.0, 10.0])
    np.testing.assert_allclose(
        second_episode_first_sample["action"][:, 0],
        [10.0, 11.0, 12.0],
    )
    assert second_episode_first_sample["map_seed"] == 11


def test_normalizer_rejects_misaligned_action_array() -> None:
    """Observation and action arrays must contain the same number of samples."""
    # Arrange
    observations = np.ones((3, 2), dtype=np.float32)
    actions = np.ones((2, 2), dtype=np.float32)

    # Act and assert
    with pytest.raises(ValueError, match="actions must have shape"):
        Normalizer.from_data(observations, actions)


def test_collection_keeps_successes_and_records_failures(
    expert_config: PurePursuitConfig,
) -> None:
    """Batch collection should keep valid demonstrations and explain rejected maps."""
    # Arrange
    sealed_wall = [Rectangle(xmin=2.8, ymin=0.0, xmax=3.2, ymax=4.0)]

    def environment_factory(map_seed: int) -> NavigationEnv:
        selected_obstacles = [] if map_seed == 1 else sealed_wall
        return create_navigation_environment(selected_obstacles)

    typed_factory: Callable[[int], NavigationEnv] = environment_factory

    # Act
    episodes, failures = collect_expert_episodes(
        env_factory=typed_factory,
        map_seeds=[1, 2],
        expert_config=expert_config,
        grid_step=0.5,
    )

    # Assert
    assert [episode.map_seed for episode in episodes] == [1]
    assert failures == {2: "planning_failed"}
