import math


def world_to_robot(dx_world: float, dy_world: float, theta: float) -> tuple[float, float]:
    forward = dx_world * math.cos(theta) + dy_world * math.sin(theta)
    left = dy_world * math.cos(theta) - dx_world * math.sin(theta)
    return (forward, left)
