import math

import pytest

from diffusion_nav.kinematics import DifferentialDrive, Pose2D, Twist2D, integrate_pose, wrap_angle


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0.0, 0.0), (math.tau, 0.0), (-math.tau, 0.0), (3.0 * math.pi / 2.0, -math.pi / 2.0)],
)
def test_wrap_angle(angle: float, expected: float) -> None:
    assert wrap_angle(angle) == pytest.approx(expected)


def test_equal_wheel_speeds_produce_straight_motion() -> None:
    robot = DifferentialDrive(wheel_radius=0.05, wheel_separation=0.30)
    twist = robot.forward_kine(left_wheel_speed=2.0, right_wheel_speed=2.0)
    assert twist.linear_velocity == pytest.approx(0.10)
    assert twist.angular_velocity == pytest.approx(0.0)


def test_opposite_wheel_speeds_produce_rotation() -> None:
    robot = DifferentialDrive(wheel_radius=0.05, wheel_separation=0.30)
    twist = robot.forward_kine(left_wheel_speed=-2.0, right_wheel_speed=2.0)
    assert twist.linear_velocity == pytest.approx(0.0)
    assert twist.angular_velocity == pytest.approx(2.0 / 3.0)


def test_integrate_straight_motion() -> None:
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    twist = Twist2D(linear_velocity=2.0, angular_velocity=0.0)
    result = integrate_pose(pose, twist, dt=0.5)
    assert result.x == pytest.approx(1.0)
    assert result.y == pytest.approx(0.0)
    assert result.theta == pytest.approx(0.0)


def test_integrate_rotation_in_place() -> None:
    pose = Pose2D(x=2.0, y=3.0, theta=0.0)
    twist = Twist2D(linear_velocity=0.0, angular_velocity=math.pi)
    result = integrate_pose(pose, twist, dt=0.5)
    assert result.x == pytest.approx(2.0)
    assert result.y == pytest.approx(3.0)
    assert result.theta == pytest.approx(math.pi / 2.0)


@pytest.mark.parametrize(
    ("wheel_radius", "wheel_separation"),
    [
        (0.0, 0.30),
        (0.05, 0.0),
    ],
)
def test_reject_invalid_robot_geometry(
    wheel_radius: float,
    wheel_separation: float,
) -> None:
    with pytest.raises(ValueError):
        DifferentialDrive(wheel_radius=wheel_radius, wheel_separation=wheel_separation)


def test_reject_non_positive_time_step() -> None:
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    twist = Twist2D(linear_velocity=1.0, angular_velocity=0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        integrate_pose(pose, twist, dt=0.0)
