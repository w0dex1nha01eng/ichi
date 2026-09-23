import math

import pytest

from diffusion_nav.kinematics import (
    DifferentialDrive,
    Pose2D,
    Twist2D,
    integrate_pose,
    wrap_angle,
)


@pytest.fixture
def differential_drive() -> DifferentialDrive:
    """Return a representative differential-drive robot geometry."""
    return DifferentialDrive(
        wheel_radius=0.05,
        wheel_separation=0.30,
    )


@pytest.mark.parametrize(
    ("angle", "expected_angle"),
    [
        pytest.param(0.0, 0.0, id="zero"),
        pytest.param(math.tau, 0.0, id="positive-full-turn"),
        pytest.param(-math.tau, 0.0, id="negative-full-turn"),
        pytest.param(3.0 * math.pi / 2.0, -math.pi / 2.0, id="three-quarter-turn"),
    ],
)
def test_wrap_angle_maps_values_into_signed_pi_interval(
    angle: float,
    expected_angle: float,
) -> None:
    """Angles that describe the same heading should share one canonical value."""
    # Act
    wrapped_angle = wrap_angle(angle)

    # Assert
    assert wrapped_angle == pytest.approx(expected_angle)


def test_equal_wheel_speeds_produce_straight_motion(
    differential_drive: DifferentialDrive,
) -> None:
    """Equal wheel speeds should create translation without rotation."""
    # Act
    twist = differential_drive.forward_kine(
        left_wheel_speed=2.0,
        right_wheel_speed=2.0,
    )

    # Assert
    assert twist.linear_velocity == pytest.approx(0.10)
    assert twist.angular_velocity == pytest.approx(0.0)


def test_opposite_wheel_speeds_produce_rotation_in_place(
    differential_drive: DifferentialDrive,
) -> None:
    """Opposite wheel speeds should cancel translation and retain angular velocity."""
    # Act
    twist = differential_drive.forward_kine(
        left_wheel_speed=-2.0,
        right_wheel_speed=2.0,
    )

    # Assert
    assert twist.linear_velocity == pytest.approx(0.0)
    assert twist.angular_velocity == pytest.approx(2.0 / 3.0)


def test_integrate_pose_advances_straight_motion() -> None:
    """Forward Euler integration should advance the pose along its current heading."""
    # Arrange
    initial_pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    commanded_twist = Twist2D(linear_velocity=2.0, angular_velocity=0.0)

    # Act
    resulting_pose = integrate_pose(
        pose=initial_pose,
        twist=commanded_twist,
        dt=0.5,
    )

    # Assert
    assert resulting_pose.x == pytest.approx(1.0)
    assert resulting_pose.y == pytest.approx(0.0)
    assert resulting_pose.theta == pytest.approx(0.0)


def test_integrate_pose_applies_rotation_in_place() -> None:
    """Pure angular velocity should update heading while preserving position."""
    # Arrange
    initial_pose = Pose2D(x=2.0, y=3.0, theta=0.0)
    commanded_twist = Twist2D(linear_velocity=0.0, angular_velocity=math.pi)

    # Act
    resulting_pose = integrate_pose(
        pose=initial_pose,
        twist=commanded_twist,
        dt=0.5,
    )

    # Assert
    assert resulting_pose.x == pytest.approx(2.0)
    assert resulting_pose.y == pytest.approx(3.0)
    assert resulting_pose.theta == pytest.approx(math.pi / 2.0)


@pytest.mark.parametrize(
    ("wheel_radius", "wheel_separation"),
    [
        pytest.param(0.0, 0.30, id="zero-wheel-radius"),
        pytest.param(0.05, 0.0, id="zero-wheel-separation"),
    ],
)
def test_differential_drive_rejects_invalid_geometry(
    wheel_radius: float,
    wheel_separation: float,
) -> None:
    """Physical dimensions must be strictly positive."""
    # Act and assert
    with pytest.raises(ValueError):
        DifferentialDrive(
            wheel_radius=wheel_radius,
            wheel_separation=wheel_separation,
        )


def test_integrate_pose_rejects_non_positive_time_step() -> None:
    """Integration with no forward time progression should be rejected."""
    # Arrange
    initial_pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    commanded_twist = Twist2D(linear_velocity=1.0, angular_velocity=0.0)

    # Act and assert
    with pytest.raises(ValueError, match="dt must be positive"):
        integrate_pose(
            pose=initial_pose,
            twist=commanded_twist,
            dt=0.0,
        )
