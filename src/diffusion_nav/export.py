import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import world_to_robot
from .kinematics import Pose2D, wrap_angle

Waypoint = tuple[float, float]


@dataclass(frozen=True, slots=True)
class PurePursuitConfig:
    lookahead_distance: float
    linear_velocity: float
    max_angular_velocity: float
    goal_tolerance: float
    rotation_start_angle: float = math.pi / 2
    rotation_exit_angle: float = math.pi / 4
    rotation_speed: float = 1.0

    def __post_init__(self) -> None:
        parameters = {"lookahead_distance": self.lookahead_distance, "linear_velocity": self.linear_velocity, "max_angular_velocity": self.max_angular_velocity,
            "goal_tolerance": self.goal_tolerance, "rotation_speed": self.rotation_speed}
        for name, value in parameters.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        angle_parameters = {"rotation_start_angle": self.rotation_start_angle, "rotation_exit_angle": self.rotation_exit_angle}
        for name, value in angle_parameters.items():
            if not math.isfinite(value) or value <= 0 or value > math.pi:
                raise ValueError(f"{name} must be a finite angle in (0, pi]")


def _project_onto_segment(pose: Pose2D, start: Waypoint, end: Waypoint) -> tuple[float, float]:
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    length_squared = segment_x * segment_x + segment_y * segment_y
    if length_squared == 0:
        return start
    projection = ((pose.x - start[0]) * segment_x + (pose.y - start[1]) * segment_y) / length_squared
    projection = min(max(projection, 0.0), 1.0)
    return start[0] + projection * segment_x, start[1] + projection * segment_y


def select_carrot_point(pose: Pose2D, path: Sequence[Waypoint], lookahead_distance: float) -> Waypoint:
    if not all(math.isfinite(value) for value in (pose.x, pose.y, pose.theta)):
        raise ValueError("pose values must be finite")
    if len(path) == 0:
        raise ValueError("path must contain at least one waypoint")
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
    if len(checked_path) == 1:
        return checked_path[0]

    closest_distance_squared = float("inf")
    closest_segment = 0
    closest_point = checked_path[0]
    for index in range(len(checked_path) - 1):
        point_x, point_y = _project_onto_segment(pose, checked_path[index], checked_path[index + 1])
        distance_squared = (point_x - pose.x) ** 2 + (point_y - pose.y) ** 2
        if distance_squared < closest_distance_squared:
            closest_distance_squared = distance_squared
            closest_segment = index
            closest_point = (point_x, point_y)

    segment_lengths = [0.0]
    for index in range(len(checked_path) - 1):
        segment_lengths.append(segment_lengths[-1] + math.hypot(checked_path[index + 1][0] - checked_path[index][0], checked_path[index + 1][1] - checked_path[index][1]))
    traveled = segment_lengths[closest_segment] + math.hypot(closest_point[0] - checked_path[closest_segment][0], closest_point[1] - checked_path[closest_segment][1])
    target_traveled = traveled + lookahead_distance
    if target_traveled >= segment_lengths[-1]:
        return checked_path[-1]
    for index in range(len(checked_path) - 1):
        if target_traveled <= segment_lengths[index + 1]:
            segment_length = segment_lengths[index + 1] - segment_lengths[index]
            fraction = (target_traveled - segment_lengths[index]) / segment_length if segment_length > 0 else 0.0
            return (checked_path[index][0] + fraction * (checked_path[index + 1][0] - checked_path[index][0]),
                    checked_path[index][1] + fraction * (checked_path[index + 1][1] - checked_path[index][1]))
    return checked_path[-1]


def compute_pure_pursuit_action(pose: Pose2D, target: Waypoint, linear_velocity: float, max_angular_velocity: float) -> np.ndarray:
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
    clipped_angular_velocity = float(np.clip(angular_velocity, -max_angular_velocity, max_angular_velocity))

    return np.array([linear_velocity, clipped_angular_velocity], dtype=np.float32)


class PurePursuitExpert:
    def __init__(self, config: PurePursuitConfig) -> None:
        self.config = config
        self.path: tuple[Waypoint, ...] = ()
        self._start_decision_made = False
        self._rotate_in_place = False
        self._rotation_target_angle = 0.0

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
        self._start_decision_made = False
        self._rotate_in_place = False
        self._rotation_target_angle = 0.0

    def reached_goal(self, pose: Pose2D) -> bool:
        if not self.path:
            return False

        goal_x, goal_y = self.path[-1]
        distance = math.hypot(goal_x - pose.x, goal_y - pose.y)
        return distance <= self.config.goal_tolerance

    def _decide_start_rotation(self, pose: Pose2D) -> None:
        if not all(math.isfinite(value) for value in (pose.x, pose.y, pose.theta)):
            raise ValueError("pose values must be finite")
        if len(self.path) < 2:
            return
        goal_x, goal_y = self.path[-1]
        goal_direction = math.atan2(goal_y - pose.y, goal_x - pose.x)
        if abs(wrap_angle(goal_direction - pose.theta)) <= self.config.rotation_start_angle:
            return
        next_x, next_y = self.path[1]
        path_direction = math.atan2(next_y - pose.y, next_x - pose.x)
        if abs(wrap_angle(path_direction - pose.theta)) <= self.config.rotation_exit_angle:
            return
        self._rotate_in_place = True
        self._rotation_target_angle = path_direction

    def act(self, pose: Pose2D) -> np.ndarray:
        if not self.path:
            raise RuntimeError("reset must load a path before act is called")

        if self.reached_goal(pose):
            return np.zeros(2, dtype=np.float32)

        if not self._start_decision_made:
            self._start_decision_made = True
            self._decide_start_rotation(pose)

        if self._rotate_in_place:
            angle_error = wrap_angle(self._rotation_target_angle - pose.theta)
            if abs(angle_error) > self.config.rotation_exit_angle:
                rotation_direction = 1.0 if angle_error > 0 else -1.0
                return np.array([0.0, rotation_direction * self.config.rotation_speed], dtype=np.float32)
            self._rotate_in_place = False

        target = select_carrot_point(pose=pose, path=self.path, lookahead_distance=self.config.lookahead_distance)
        return compute_pure_pursuit_action(pose=pose, target=target, linear_velocity=self.config.linear_velocity, max_angular_velocity=self.config.max_angular_velocity)
