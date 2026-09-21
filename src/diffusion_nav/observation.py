import math

import numpy as np

from .geometry import world_to_robot


def build_observation(pose, goal, previous_action, lidar_distances):
    world_dx = goal[0] - pose.x
    world_dy = goal[1] - pose.y
    world_vector = world_to_robot(world_dx, world_dy, pose.theta)
    goal_forward_local = world_vector[0]
    goal_left_local = world_vector[1]
    goal_angle = math.atan2(world_vector[1], world_vector[0])
    previous_v = previous_action[0]
    previous_omega = previous_action[1]
    N = len(lidar_distances)
    observation = np.empty(N + 6, dtype=np.float32)
    observation[0] = goal_forward_local
    observation[1] = goal_left_local
    observation[2] = math.sin(goal_angle)
    observation[3] = math.cos(goal_angle)
    observation[4] = previous_v
    observation[5] = previous_omega
    for i, distance in enumerate(lidar_distances):
        observation[i + 6] = distance

    return observation
