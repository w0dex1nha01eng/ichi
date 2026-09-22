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

POSE = Pose2D(1.0, 1.0, 0.0)
WIDTH = 7.0
HEIGHT = 7.0
STEPS = 1.0
RADIUS = 0.2


def test_coordinate_conversion_uses_nearest_grid_node():
    assert world_to_grid(2.49, 3.51, STEPS) == grid_xy(2, 4)
    assert world_to_grid(2.5, 3.5, STEPS) == grid_xy(3, 4)
    assert grid_to_world(2, 4, STEPS) == pytest.approx((2.0, 4.0))


@pytest.mark.parametrize("steps", [0.0, -1.0, math.inf, math.nan])
def test_coordinate_conversion_rejects_invalid_steps(steps):
    with pytest.raises(ValueError):
        world_to_grid(1.0, 1.0, steps)


def test_neighbor_filters_walls_and_obstacles():
    obstacle = Rectangle(2.8, 1.8, 3.2, 2.2)
    result = neighbor(grid_xy(2, 2), STEPS, POSE, RADIUS, [obstacle], WIDTH, HEIGHT)
    assert set(result) == {grid_xy(2, 3), grid_xy(2, 1), grid_xy(1, 2)}
    boundary = neighbor(grid_xy(1, 1), STEPS, POSE, RADIUS, [], WIDTH, HEIGHT)
    assert set(boundary) == {grid_xy(1, 2), grid_xy(2, 1)}


def assert_valid_path(path, start, goal, obstacles):
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert len(path) == len(set(path))
    for first, second in pairwise(path):
        assert abs(first.x - second.x) + abs(first.y - second.y) == 1
    for node in path:
        assert grid_free(POSE, node.x, node.y, STEPS, RADIUS, obstacles, WIDTH, HEIGHT)


def test_astar_empty_map_returns_shortest_safe_path():
    start, goal = grid_xy(1, 1), grid_xy(5, 4)
    path = a_star(start, goal, POSE, STEPS, RADIUS, [], WIDTH, HEIGHT)
    assert_valid_path(path, start, goal, [])
    assert len(path) - 1 == 7


def test_astar_routes_through_wall_gap():
    obstacles = [Rectangle(3.0, 0.0, 4.0, 2.4), Rectangle(3.0, 3.6, 4.0, 7.0)]
    start, goal = grid_xy(1, 3), grid_xy(5, 3)
    path = a_star(start, goal, POSE, STEPS, RADIUS, obstacles, WIDTH, HEIGHT)
    assert_valid_path(path, start, goal, obstacles)
    assert grid_xy(3, 3) in path


def test_astar_returns_none_when_wall_is_sealed():
    obstacles = [Rectangle(3.0, 0.0, 4.0, 7.0)]
    assert a_star(grid_xy(1, 3), grid_xy(5, 3), POSE, STEPS, RADIUS, obstacles, WIDTH, HEIGHT) is None


def test_astar_rejects_blocked_start_and_goal():
    obstacle = Rectangle(2.0, 2.0, 4.0, 4.0)
    with pytest.raises(ValueError):
        a_star(grid_xy(3, 3), grid_xy(5, 5), POSE, STEPS, RADIUS, [obstacle], WIDTH, HEIGHT)
    with pytest.raises(ValueError):
        a_star(grid_xy(1, 1), grid_xy(3, 3), POSE, STEPS, RADIUS, [obstacle], WIDTH, HEIGHT)
