import math

import numpy as np

from .geometry import world_to_robot
from .kinematics import Pose2D


def build_observation(pose: Pose2D, goal: tuple[float, float], previous_action, lidar_distances) -> np.ndarray:
    goal = np.asarray(goal, dtype=np.float32)
    previous_action = np.asarray(previous_action, dtype=np.float32)
    lidar_distances = np.asarray(lidar_distances, dtype=np.float32)

    if goal.shape != (2,):
        raise ValueError(f"goal shape must be (2,), got {goal.shape}")
    if previous_action.shape != (2,):
        raise ValueError(f"previous_action shape must be (2,), got {previous_action.shape}")
    if lidar_distances.ndim != 1:
        raise ValueError("lidar_distances must be a one-dimensional array")
    if lidar_distances.size == 0:
        raise ValueError("lidar_distances cannot be empty")

    pose_values = np.array([pose.x, pose.y, pose.theta], dtype=np.float32)
    if not np.all(np.isfinite(pose_values)):
        raise ValueError("pose values must be finite ")
    if not np.all(np.isfinite(goal)):
        raise ValueError("goal values must be finite")
    if not np.all(np.isfinite(previous_action)):
        raise ValueError("previous_action values must be finite")
    if not np.all(np.isfinite(lidar_distances)):
        raise ValueError("lidar_distances values must be finite")
    if np.any(lidar_distances < 0):
        raise ValueError("lidar distances cannot be negative")

    world_dx = goal[0] - pose.x
    world_dy = goal[1] - pose.y
    goal_forward_local, goal_left_local = world_to_robot(world_dx, world_dy, pose.theta)
    goal_angle = math.atan2(goal_left_local, goal_forward_local)
    previous_v = previous_action[0]
    previous_omega = previous_action[1]
    observation_base = np.array([goal_forward_local, goal_left_local, math.sin(goal_angle), math.cos(goal_angle), previous_v, previous_omega], dtype=np.float32)
    observation = np.concatenate([observation_base, lidar_distances]).astype(np.float32)
    if not np.all(np.isfinite(observation)):
        raise ValueError("observation contains infinite values")

    return observation
