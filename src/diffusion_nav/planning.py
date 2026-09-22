import heapq
import math
from dataclasses import dataclass
from itertools import count

from .collision import robot_in_collision
from .kinematics import Pose2D


@dataclass(frozen=True, slots=True)
class grid_xy:
    x: int
    y: int


def _validate_steps(steps: float) -> None:
    if not math.isfinite(steps) or steps <= 0:
        raise ValueError("steps must be a finite positive number")


def world_to_grid(world_x: float, world_y: float, steps: float) -> grid_xy:
    _validate_steps(steps)
    if not math.isfinite(world_x) or not math.isfinite(world_y):
        raise ValueError("world coordinates must be finite")
    grid_x = math.floor(world_x / steps + 0.5)
    grid_y = math.floor(world_y / steps + 0.5)
    return grid_xy(grid_x, grid_y)


def grid_to_world(grid_x: int, grid_y: int, steps: float) -> tuple[float, float]:
    _validate_steps(steps)
    return (grid_x * steps, grid_y * steps)


def grid_free(pose: Pose2D, grid_x, grid_y, steps, radius, obstacles, width, height) -> bool:
    x, y = grid_to_world(grid_x, grid_y, steps)
    candidate_pose = Pose2D(x, y, theta=pose.theta)
    return not robot_in_collision(candidate_pose, radius, obstacles, width, height)


def neighbor(grid: grid_xy, steps, pose: Pose2D, radius, obstacles, width, height) -> list[grid_xy]:
    grid_surrounding = (grid_xy(grid.x, grid.y + 1), grid_xy(grid.x, grid.y - 1), grid_xy(grid.x - 1, grid.y), grid_xy(grid.x + 1, grid.y))
    neighbor_free = []
    for candidate in grid_surrounding:
        if grid_free(pose, candidate.x, candidate.y, steps, radius, obstacles, width, height):
            neighbor_free.append(candidate)
    return neighbor_free


def heuristic(grid: grid_xy, goal: grid_xy) -> int:
    return abs(goal.x - grid.x) + abs(goal.y - grid.y)


def reconstruct_path(came_from: dict[grid_xy, grid_xy], goal: grid_xy) -> list[grid_xy]:
    path = [goal]
    current = goal
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star(start: grid_xy, goal: grid_xy, pose: Pose2D, steps, radius, obstacles, width, height) -> list[grid_xy] | None:
    _validate_steps(steps)
    if not grid_free(pose, start.x, start.y, steps, radius, obstacles, width, height):
        raise ValueError("start is not traversable")
    if not grid_free(pose, goal.x, goal.y, steps, radius, obstacles, width, height):
        raise ValueError("goal is not traversable")

    open_heap: list[tuple[float, int, grid_xy]] = []
    tie_breaker = count()
    came_from: dict[grid_xy, grid_xy] = {}
    g_score: dict[grid_xy, float] = {start: 0.0}

    heapq.heappush(open_heap, (heuristic(start, goal), next(tie_breaker), start))

    while open_heap:
        current_f, _, current = heapq.heappop(open_heap)
        best_current_f = g_score[current] + heuristic(current, goal)
        if current_f > best_current_f:
            continue
        if current == goal:
            return reconstruct_path(came_from, goal)

        for next_grid in neighbor(current, steps, pose, radius, obstacles, width, height):
            tentative_g = g_score[current] + 1.0
            known_g = g_score.get(next_grid, float("inf"))
            if tentative_g < known_g:
                came_from[next_grid] = current
                g_score[next_grid] = tentative_g
                next_f = tentative_g + heuristic(next_grid, goal)
                heapq.heappush(open_heap, (next_f, next(tie_breaker), next_grid))

    return None
