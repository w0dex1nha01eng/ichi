import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import world_to_robot
from .kinematics import Pose2D

Waypoint = tuple[float, float]


@dataclass(frozen=True, slots=True)
class PurePursuitConfig:

    lookahead_distance: float
    linear_velocity: float
    max_angular_velocity: float
    goal_tolerance: float

    def __post_init__(self) -> None:
        parameters = {
            "lookahead_distance": self.lookahead_distance,
            "linear_velocity": self.linear_velocity,
            "max_angular_velocity": self.max_angular_velocity,
            "goal_tolerance": self.goal_tolerance,
        }
        for name, value in parameters.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")


def select_lookahead_point(pose: Pose2D,path: Sequence[Waypoint],start_index: int,lookahead_distance: float) -> tuple[Waypoint, int]:
    if not all(math.isfinite(value) for value in (pose.x, pose.y, pose.theta)):
        raise ValueError("pose values must be finite")
    if len(path) == 0:
        raise ValueError("path must contain at least one waypoint")
    if not isinstance(start_index, int) or isinstance(start_index, bool):
        raise TypeError("start_index must be an integer")
    if not 0 <= start_index < len(path):
        raise ValueError("start_index is outside the path")
    if not math.isfinite(lookahead_distance) or lookahead_distance <= 0:
        raise ValueError("lookahead_distance must be a finite positive number")

    checked_path: list[Waypoint] = []
    for point in path:
        if len(point) != 2:
            raise ValueError("each waypoint must contain x and y")
        point_x = float(point[0])
        point_y = float(point[1])
        if not math.isfinite(point_x) or not math.isfinite(point_y):
            raise ValueError("waypoint coordinates must be finite")
        checked_path.append((point_x, point_y))

    closest_index = start_index
    closest_distance_squared = float("inf")
    for index in range(start_index, len(checked_path)):
        point_x, point_y = checked_path[index]
        distance_squared = (point_x - pose.x) ** 2 + (point_y - pose.y) ** 2
        if distance_squared < closest_distance_squared:
            closest_distance_squared = distance_squared
            closest_index = index

    lookahead_squared = lookahead_distance**2
    for index in range(closest_index, len(checked_path)):
        point_x, point_y = checked_path[index]
        distance_squared = (point_x - pose.x) ** 2 + (point_y - pose.y) ** 2
        if distance_squared >= lookahead_squared:
            return checked_path[index], index

    final_index = len(checked_path) - 1
    return checked_path[final_index], final_index


def compute_pure_pursuit_action(pose: Pose2D,target: Waypoint,linear_velocity: float,max_angular_velocity: float) -> np.ndarray:
    if not all(math.isfinite(value) for value in (pose.x, pose.y, pose.theta)):
        raise ValueError("pose values must be finite")
    if len(target) != 2:
        raise ValueError("target must contain x and y")

    target_x = float(target[0])
    target_y = float(target[1])
    if not math.isfinite(target_x) or not math.isfinite(target_y):
        raise ValueError("target coordinates must be finite")
    if not math.isfinite(linear_velocity) or linear_velocity < 0:
        raise ValueError("linear_velocity must be a finite non-negative number")
    if not math.isfinite(max_angular_velocity) or max_angular_velocity <= 0:
        raise ValueError("max_angular_velocity must be a finite positive number")

    dx_world = target_x - pose.x
    dy_world = target_y - pose.y
    target_forward, target_left = world_to_robot(dx_world, dy_world, pose.theta)
    target_distance_squared = target_forward**2 + target_left**2

    if target_distance_squared <= 1e-12:
        return np.zeros(2, dtype=np.float32)

    curvature = 2.0 * target_left / target_distance_squared
    angular_velocity = linear_velocity * curvature
    clipped_angular_velocity = float(np.clip(angular_velocity,-max_angular_velocity,max_angular_velocity))

    return np.array([linear_velocity, clipped_angular_velocity],dtype=np.float32,)

class PurePursuitExpert:

    def __init__(self, config: PurePursuitConfig) -> None:
        self.config = config
        self.path: tuple[Waypoint, ...] = ()
        self.target_index = 0

    def reset(self, path: Sequence[Waypoint] | None) -> None:
        if path is None or len(path) == 0:
            raise ValueError("path must contain at least one waypoint")

        checked_path: list[Waypoint] = []
        for point in path:
            if len(point) != 2:
                raise ValueError("each waypoint must contain x and y")
            x = float(point[0])
            y = float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("waypoint coordinates must be finite")
            checked_path.append((x, y))

        self.path = tuple(checked_path)
        self.target_index = 0

    def reached_goal(self, pose: Pose2D) -> bool:
        if not self.path:
            return False

        goal_x, goal_y = self.path[-1]
        distance = math.hypot(goal_x - pose.x, goal_y - pose.y)
        return distance <= self.config.goal_tolerance

    def act(self, pose: Pose2D) -> np.ndarray:
        if not self.path:
            raise RuntimeError("reset must load a path before act is called")

        if self.reached_goal(pose):
            return np.zeros(2, dtype=np.float32)

        target, target_index = select_lookahead_point(pose=pose,path=self.path,start_index=self.target_index, lookahead_distance=self.config.lookahead_distance)
        self.target_index = target_index

        return compute_pure_pursuit_action(pose=pose,target=target,linear_velocity=self.config.linear_velocity,max_angular_velocity=self.config.max_angular_velocity)
