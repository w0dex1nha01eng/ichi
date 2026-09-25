import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass(frozen=True, slots=True)
class Twist2D:
    linear_velocity: float
    angular_velocity: float


@dataclass(frozen=True, slots=True)
class DifferentialDrive:
    wheel_radius: float
    wheel_separation: float

    def __post_init__(self) -> None:
        if self.wheel_radius <= 0:
            raise ValueError("wheel_radius must be positive")
        if self.wheel_separation <= 0:
            raise ValueError("wheel_separation can't be negative")

    def forward_kine(self, left_wheel_speed: float, right_wheel_speed: float) -> Twist2D:
        left_linear_velocity = self.wheel_radius * left_wheel_speed
        right_linear_velocity = self.wheel_radius * right_wheel_speed
        linear_velocity = (left_linear_velocity + right_linear_velocity) / 2.0
        angular_velocity = (right_linear_velocity - left_linear_velocity) / self.wheel_separation
        return Twist2D(linear_velocity=linear_velocity, angular_velocity=angular_velocity)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % math.tau - math.pi


def integrate_pose(pose: Pose2D, twist: Twist2D, dt: float) -> Pose2D:
    if dt <= 0:
        raise ValueError("dt must be positive")
    return Pose2D(x=pose.x + twist.linear_velocity * math.cos(pose.theta) * dt, y=pose.y + twist.linear_velocity * math.sin(pose.theta) * dt,
                  theta=wrap_angle(pose.theta + twist.angular_velocity * dt))
