import math

import numpy as np


def scan_lidar(pose, obstacles, width, height, num_rays, max_range, sample_step):
    if min(num_rays, max_range, sample_step) <= 0:
        raise ValueError("They must be positive")

    SafetySpace = np.full(num_rays, max_range, dtype=np.float32)

    num_steps = math.ceil(max_range / sample_step)
    for i in range(num_rays):
        angle = pose.theta + 2 * math.pi * i / num_rays
        for j in range(num_steps + 1):
            d = min(j * sample_step, max_range)
            sample_x = pose.x + d * math.cos(angle)
            sample_y = pose.y + d * math.sin(angle)

            hit_wall = sample_x <= 0 or sample_x >= width or sample_y <= 0 or sample_y >= height
            hit_obstacle = any(
                rectangle.xmin <= sample_x <= rectangle.xmax
                and rectangle.ymin <= sample_y <= rectangle.ymax
                for rectangle in obstacles
            )
            if hit_wall or hit_obstacle:
                SafetySpace[i] = d
                break

    return SafetySpace
