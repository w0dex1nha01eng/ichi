import math

import pytest

from diffusion_nav.dataset import run_expert_episode
from diffusion_nav.env import NavigationEnv
from diffusion_nav.export import PurePursuitConfig, PurePursuitExpert
from diffusion_nav.kinematics import Pose2D, wrap_angle


def make_config(**overrides: float) -> PurePursuitConfig:
    """Build an expert config whose rotation values can be overridden per test."""
    values: dict[str, float] = {
        "lookahead_distance": 0.5,
        "linear_velocity": 1.0,
        "max_angular_velocity": 2.0,
        "goal_tolerance": 0.2,
    }
    values.update(overrides)
    return PurePursuitConfig(**values)


def make_expert(config: PurePursuitConfig) -> PurePursuitExpert:
    """Return an expert reset on a straight eastbound path."""
    expert = PurePursuitExpert(config)
    expert.reset([(1.0, 1.0), (5.0, 1.0)])
    return expert


def test_facing_away_from_goal_starts_with_in_place_rotation() -> None:
    """A start heading opposite the goal should first rotate gently without driving."""
    # Arrange
    expert = make_expert(make_config())

    # Act
    action = expert.act(Pose2D(1.0, 1.0, math.pi))

    # Assert
    assert action[0] == 0.0
    assert action[1] == -1.0


def test_rotation_switches_to_driving_once_heading_enters_exit_cone() -> None:
    """The expert should keep rotating at 1 rad/s until within 45 degrees of the path."""
    # Arrange
    config = make_config()
    expert = make_expert(config)
    theta = math.pi

    # Act
    actions = []
    for _ in range(40):
        action = expert.act(Pose2D(1.0, 1.0, theta))
        actions.append(action)
        if action[0] > 0:
            break
        theta = wrap_angle(theta + action[1] * 0.1)

    # Assert
    rotating_actions = actions[:-1]
    assert len(rotating_actions) == 24
    assert all(action[0] == 0.0 for action in rotating_actions)
    assert all(action[1] == -1.0 for action in rotating_actions)
    assert abs(wrap_angle(0.0 - theta)) <= config.rotation_exit_angle
    assert actions[-1][0] == pytest.approx(1.0)


def test_heading_within_start_cone_skips_rotation() -> None:
    """A heading already facing the goal should drive immediately."""
    # Arrange
    expert = make_expert(make_config())

    # Act
    action = expert.act(Pose2D(1.0, 1.0, 0.0))

    # Assert
    assert action[0] == pytest.approx(1.0)


def test_rotation_decision_is_made_only_at_episode_start() -> None:
    """A backward heading later in the episode must not trigger in-place rotation."""
    # Arrange
    expert = make_expert(make_config())

    # Act
    first_action = expert.act(Pose2D(1.0, 1.0, 0.0))
    later_action = expert.act(Pose2D(2.0, 1.0, math.pi))

    # Assert
    assert first_action[0] == pytest.approx(1.0)
    assert later_action[0] == pytest.approx(1.0)


def test_reset_reevaluates_start_decision() -> None:
    """Reset should forget a previous start decision for the new path."""
    # Arrange
    config = make_config()
    expert = make_expert(config)

    # Act
    rotating_action = expert.act(Pose2D(1.0, 1.0, math.pi))
    expert.reset([(1.0, 1.0), (5.0, 1.0)])
    driving_action = expert.act(Pose2D(1.0, 1.0, 0.0))

    # Assert
    assert rotating_action[0] == 0.0
    assert driving_action[0] == pytest.approx(1.0)


def test_rotation_speed_is_configurable() -> None:
    """Rotation speed should come from config instead of a hard-coded value."""
    # Arrange
    expert = make_expert(make_config(rotation_speed=0.5))

    # Act
    action = expert.act(Pose2D(1.0, 1.0, math.pi))

    # Assert
    assert action[0] == 0.0
    assert action[1] == pytest.approx(-0.5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("rotation_start_angle", 0.0, id="zero-start-angle"),
        pytest.param("rotation_start_angle", math.pi + 0.01, id="oversized-start-angle"),
        pytest.param("rotation_exit_angle", 0.0, id="zero-exit-angle"),
        pytest.param("rotation_exit_angle", math.pi + 0.01, id="oversized-exit-angle"),
        pytest.param("rotation_speed", 0.0, id="zero-rotation-speed"),
        pytest.param("rotation_speed", -1.0, id="negative-rotation-speed"),
    ],
)
def test_config_rejects_invalid_rotation_parameters(field: str, value: float) -> None:
    """Rotation thresholds must lie in (0, pi] and rotation speed must be positive."""
    # Act and assert
    with pytest.raises(ValueError):
        make_config(**{field: value})


def test_episode_with_backward_facing_start_succeeds_and_starts_rotating() -> None:
    """A full rollout facing away from the goal should still succeed cleanly."""
    # Arrange
    environment = NavigationEnv(
        width=6.0,
        height=4.0,
        robot_radius=0.2,
        start_pose=Pose2D(1.0, 2.0, math.pi),
        goal=(5.0, 2.0),
        obstacles=[],
        dt=0.1,
        max_steps=100,
        goal_radius=0.2,
        num_rays=8,
        lidar_max_range=3.0,
        lidar_sample_step=0.1,
        max_linear_velocity=1.0,
        max_angular_velocity=2.0,
    )
    expert = PurePursuitExpert(make_config())

    # Act
    episode = run_expert_episode(
        env=environment,
        expert=expert,
        grid_step=0.5,
        map_seed=42,
    )

    # Assert
    assert episode.success is True
    assert episode.collision is False
    assert episode.actions[0, 0] == 0.0
    assert abs(episode.actions[0, 1]) == 1.0
