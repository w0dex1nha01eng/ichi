import math
from itertools import pairwise

import pytest

from diffusion_nav.collision import Rectangle
from diffusion_nav.kinematics import Pose2D
from diffusion_nav.planning import (
    a_star,
    grid_free,
    grid_to_world,
    grid_xy,
    neighbor,
    world_to_grid,
)

MAP_WIDTH = 7.0
MAP_HEIGHT = 7.0
GRID_STEP = 1.0
ROBOT_RADIUS = 0.2
REFERENCE_POSE = Pose2D(x=1.0, y=1.0, theta=0.0)


def assert_path_is_valid(
    path: list[grid_xy] | None,
    start: grid_xy,
    goal: grid_xy,
    obstacles: list[Rectangle],
) -> list[grid_xy]:
    """Assert all structural and collision constraints shared by A* path tests."""
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert len(path) == len(set(path))

    for first_node, second_node in pairwise(path):
        grid_distance = abs(first_node.x - second_node.x) + abs(first_node.y - second_node.y)
        assert grid_distance == 1

    for node in path:
        assert grid_free(
            pose=REFERENCE_POSE,
            grid_x=node.x,
            grid_y=node.y,
            steps=GRID_STEP,
            radius=ROBOT_RADIUS,
            obstacles=obstacles,
            width=MAP_WIDTH,
            height=MAP_HEIGHT,
        )

    return path


def test_coordinate_conversion_selects_nearest_grid_node() -> None:
    """World coordinates should round to the nearest grid intersection."""
    # Act
    rounded_down = world_to_grid(2.49, 3.51, GRID_STEP)
    rounded_up = world_to_grid(2.50, 3.50, GRID_STEP)
    world_point = grid_to_world(2, 4, GRID_STEP)

    # Assert
    assert rounded_down == grid_xy(2, 4)
    assert rounded_up == grid_xy(3, 4)
    assert world_point == pytest.approx((2.0, 4.0))


@pytest.mark.parametrize(
    "invalid_step",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(math.inf, id="infinite"),
        pytest.param(math.nan, id="not-a-number"),
    ],
)
def test_coordinate_conversion_rejects_invalid_grid_step(invalid_step: float) -> None:
    """Grid spacing must be finite and strictly positive."""
    # Act and assert
    with pytest.raises(ValueError):
        world_to_grid(1.0, 1.0, invalid_step)


def test_neighbor_filters_map_boundaries_and_obstacles() -> None:
    """Neighbor expansion should retain only traversable four-connected cells."""
    # Arrange
    obstacle = Rectangle(xmin=2.8, ymin=1.8, xmax=3.2, ymax=2.2)

    # Act
    interior_neighbors = neighbor(
        grid=grid_xy(2, 2),
        steps=GRID_STEP,
        pose=REFERENCE_POSE,
        radius=ROBOT_RADIUS,
        obstacles=[obstacle],
        width=MAP_WIDTH,
        height=MAP_HEIGHT,
    )
    boundary_neighbors = neighbor(
        grid=grid_xy(1, 1),
        steps=GRID_STEP,
        pose=REFERENCE_POSE,
        radius=ROBOT_RADIUS,
        obstacles=[],
        width=MAP_WIDTH,
        height=MAP_HEIGHT,
    )

    # Assert
    assert set(interior_neighbors) == {grid_xy(2, 3), grid_xy(2, 1), grid_xy(1, 2)}
    assert set(boundary_neighbors) == {grid_xy(1, 2), grid_xy(2, 1)}


def test_a_star_returns_shortest_path_on_empty_map() -> None:
    """A* should match Manhattan distance when no obstacle causes a detour."""
    # Arrange
    start = grid_xy(1, 1)
    goal = grid_xy(5, 4)

    # Act
    path = a_star(
        start=start,
        goal=goal,
        pose=REFERENCE_POSE,
        steps=GRID_STEP,
        radius=ROBOT_RADIUS,
        obstacles=[],
        width=MAP_WIDTH,
        height=MAP_HEIGHT,
    )

    # Assert
    verified_path = assert_path_is_valid(path, start, goal, obstacles=[])
    assert len(verified_path) - 1 == 7


def test_a_star_routes_through_available_wall_gap() -> None:
    """A* should find the only valid crossing through a divided map."""
    # Arrange
    obstacles = [
        Rectangle(xmin=3.0, ymin=0.0, xmax=4.0, ymax=2.4),
        Rectangle(xmin=3.0, ymin=3.6, xmax=4.0, ymax=7.0),
    ]
    start = grid_xy(1, 3)
    goal = grid_xy(5, 3)

    # Act
    path = a_star(
        start=start,
        goal=goal,
        pose=REFERENCE_POSE,
        steps=GRID_STEP,
        radius=ROBOT_RADIUS,
        obstacles=obstacles,
        width=MAP_WIDTH,
        height=MAP_HEIGHT,
    )

    # Assert
    verified_path = assert_path_is_valid(path, start, goal, obstacles)
    assert grid_xy(3, 3) in verified_path


def test_a_star_returns_none_when_wall_fully_seals_map() -> None:
    """A fully sealed map should be reported as unreachable."""
    # Arrange
    obstacles = [Rectangle(xmin=3.0, ymin=0.0, xmax=4.0, ymax=7.0)]

    # Act
    path = a_star(
        start=grid_xy(1, 3),
        goal=grid_xy(5, 3),
        pose=REFERENCE_POSE,
        steps=GRID_STEP,
        radius=ROBOT_RADIUS,
        obstacles=obstacles,
        width=MAP_WIDTH,
        height=MAP_HEIGHT,
    )

    # Assert
    assert path is None


@pytest.mark.parametrize(
    ("start", "goal"),
    [
        pytest.param(grid_xy(3, 3), grid_xy(5, 5), id="blocked-start"),
        pytest.param(grid_xy(1, 1), grid_xy(3, 3), id="blocked-goal"),
    ],
)
def test_a_star_rejects_blocked_endpoint(start: grid_xy, goal: grid_xy) -> None:
    """A path request with a blocked start or goal should fail immediately."""
    # Arrange
    obstacle = Rectangle(xmin=2.0, ymin=2.0, xmax=4.0, ymax=4.0)

    # Act and assert
    with pytest.raises(ValueError):
        a_star(
            start=start,
            goal=goal,
            pose=REFERENCE_POSE,
            steps=GRID_STEP,
            radius=ROBOT_RADIUS,
            obstacles=[obstacle],
            width=MAP_WIDTH,
            height=MAP_HEIGHT,
        )
