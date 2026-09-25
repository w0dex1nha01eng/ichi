import numpy as np
import pytest

from diffusion_nav.evaluate_randomized_test import _randomized_environment, _sample_range


def test_randomized_test_environment_reproduces_seeded_start_and_goal() -> None:
    environment_config = {
        "width": 10.0,
        "height": 8.0,
        "robot_radius": 0.25,
        "start_pose": [1.0, 1.0, 0.0],
        "goal": [9.0, 7.0],
        "dt": 0.1,
        "max_steps": 300,
        "goal_radius": 0.3,
        "num_rays": 16,
        "lidar_max_range": 5.0,
        "lidar_sample_step": 0.05,
        "max_linear_velocity": 1.0,
        "max_angular_velocity": 2.0,
    }
    map_config = {
        "obstacle_count": 0,
        "placement_attempts": 10,
        "obstacle_width_range": [0.5, 1.5],
        "obstacle_height_range": [0.5, 1.5],
        "clearance_from_start_goal": 0.5,
        "obstacle_separation": 0.25,
    }
    test_config = {
        "start_x_range": [1.5, 2.5],
        "start_y_range": [1.5, 2.5],
        "goal_x_range": [8.5, 9.0],
        "goal_y_range": [6.5, 7.0],
        "start_heading": 0.0,
    }

    first_environment = _randomized_environment(200000, environment_config, map_config, test_config)
    repeated_environment = _randomized_environment(200000, environment_config, map_config, test_config)
    other_environment = _randomized_environment(200001, environment_config, map_config, test_config)

    assert first_environment.start_pose == repeated_environment.start_pose
    assert first_environment.goal == repeated_environment.goal
    assert first_environment.start_pose != other_environment.start_pose
    assert first_environment.goal != other_environment.goal
    assert 1.5 <= first_environment.start_pose.x <= 2.5
    assert 1.5 <= first_environment.start_pose.y <= 2.5
    assert 8.5 <= first_environment.goal[0] <= 9.0
    assert 6.5 <= first_environment.goal[1] <= 7.0
    assert first_environment.start_pose.theta == 0.0


def test_randomized_test_environment_samples_reproducible_heading() -> None:
    environment_config = {
        "width": 10.0,
        "height": 8.0,
        "robot_radius": 0.25,
        "start_pose": [1.0, 1.0, 0.0],
        "goal": [9.0, 7.0],
        "dt": 0.1,
        "max_steps": 300,
        "goal_radius": 0.3,
        "num_rays": 16,
        "lidar_max_range": 5.0,
        "lidar_sample_step": 0.05,
        "max_linear_velocity": 1.0,
        "max_angular_velocity": 2.0,
    }
    map_config = {
        "obstacle_count": 0,
        "placement_attempts": 10,
        "obstacle_width_range": [0.5, 1.5],
        "obstacle_height_range": [0.5, 1.5],
        "clearance_from_start_goal": 0.5,
        "obstacle_separation": 0.25,
    }
    test_config = {
        "start_x_range": [1.5, 2.5],
        "start_y_range": [1.5, 2.5],
        "goal_x_range": [8.5, 9.0],
        "goal_y_range": [6.5, 7.0],
        "start_heading_range": [-np.pi, np.pi],
    }

    first_environment = _randomized_environment(300000, environment_config, map_config, test_config)
    repeated_environment = _randomized_environment(300000, environment_config, map_config, test_config)
    other_environment = _randomized_environment(300001, environment_config, map_config, test_config)

    assert first_environment.start_pose.theta == repeated_environment.start_pose.theta
    assert first_environment.start_pose.theta != other_environment.start_pose.theta
    assert -np.pi <= first_environment.start_pose.theta <= np.pi
    assert environment_config["start_pose"] == [1.0, 1.0, 0.0]


def test_randomized_test_range_rejects_nonfinite_or_reversed_values() -> None:
    assert _sample_range({"coordinate": [1.5, 2.5]}, "coordinate") == (1.5, 2.5)

    with pytest.raises(ValueError, match="ascending"):
        _sample_range({"coordinate": [2.5, 1.5]}, "coordinate")

    with pytest.raises(ValueError, match="finite"):
        _sample_range({"coordinate": [float("nan"), 2.5]}, "coordinate")
