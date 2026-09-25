import math

import numpy as np

from .env import NavigationEnv
from .lidar import scan_lidar


class PygameRenderer:
    def __init__(self, size: tuple[int, int] = (960, 720), margin: int = 36) -> None:
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError("pygame is required; install the visualization extra to enable rendering") from error

        if len(size) != 2 or min(size) <= 0 or margin <= 0:
            raise ValueError("size must be positive and margin must be positive")
        pygame.font.init()
        self.pygame = pygame
        self.size = size
        self.margin = margin
        self.surface = pygame.Surface(size)
        self.title_font = pygame.font.SysFont("arial", 22, bold=True)
        self.info_font = pygame.font.SysFont("arial", 16)

    def render(self, env: NavigationEnv, action_horizon: int, status: str, trajectory: list[tuple[float, float]]) -> np.ndarray:
        pygame = self.pygame
        screen = self.surface
        screen.fill((246, 248, 251))

        screen_width, screen_height = self.size
        available_width = screen_width - 2 * self.margin
        top_padding = 52
        bottom_padding = 42
        available_height = screen_height - top_padding - bottom_padding
        scale = min(available_width / env.width, available_height / env.height)
        map_width = round(env.width * scale)
        map_height = round(env.height * scale)
        map_left = (screen_width - map_width) // 2
        map_top = top_padding + (available_height - map_height) // 2
        map_rect = pygame.Rect(map_left, map_top, map_width, map_height)

        def to_pixel(x: float, y: float) -> tuple[int, int]:
            return round(map_left + x * scale), round(map_top + (env.height - y) * scale)

        pygame.draw.rect(screen, (255, 255, 255), map_rect)
        for grid_x in range(math.ceil(env.width) + 1):
            if grid_x <= env.width:
                start = to_pixel(grid_x, 0.0)
                end = to_pixel(grid_x, env.height)
                pygame.draw.line(screen, (232, 235, 239), start, end, 1)
        for grid_y in range(math.ceil(env.height) + 1):
            if grid_y <= env.height:
                start = to_pixel(0.0, grid_y)
                end = to_pixel(env.width, grid_y)
                pygame.draw.line(screen, (232, 235, 239), start, end, 1)

        goal_x, goal_y = env.goal
        goal_center = to_pixel(goal_x, goal_y)
        goal_radius = max(2, round(env.goal_radius * scale))
        goal_layer = pygame.Surface(self.size, pygame.SRCALPHA)
        pygame.draw.circle(goal_layer, (52, 168, 83, 70), goal_center, goal_radius)
        screen.blit(goal_layer, (0, 0))
        pygame.draw.circle(screen, (35, 125, 64), goal_center, goal_radius, 2)
        pygame.draw.circle(screen, (35, 125, 64), goal_center, 4)

        for obstacle in env.obstacles:
            obstacle_left, obstacle_bottom = to_pixel(obstacle.xmin, obstacle.ymin)
            obstacle_right, obstacle_top = to_pixel(obstacle.xmax, obstacle.ymax)
            obstacle_rect = pygame.Rect(obstacle_left, obstacle_top, max(1, obstacle_right - obstacle_left), max(1, obstacle_bottom - obstacle_top))
            pygame.draw.rect(screen, (87, 94, 103), obstacle_rect)
            pygame.draw.rect(screen, (48, 53, 60), obstacle_rect, 2)

        if len(trajectory) > 1:
            path_points = [to_pixel(x, y) for x, y in trajectory]
            pygame.draw.lines(screen, (54, 111, 193), False, path_points, 3)

        lidar_distances = scan_lidar(env.pose, env.obstacles, env.width, env.height, env.num_rays, env.lidar_max_range, env.lidar_sample_step)
        robot_center = to_pixel(env.pose.x, env.pose.y)
        for ray_index, distance in enumerate(lidar_distances):
            angle = env.pose.theta + 2.0 * math.pi * ray_index / env.num_rays
            ray_end = to_pixel(env.pose.x + float(distance) * math.cos(angle), env.pose.y + float(distance) * math.sin(angle))
            pygame.draw.line(screen, (239, 151, 45), robot_center, ray_end, 1)

        robot_radius = max(3, round(env.robot_radius * scale))
        pygame.draw.circle(screen, (57, 112, 205), robot_center, robot_radius)
        pygame.draw.circle(screen, (30, 61, 112), robot_center, robot_radius, 2)
        heading_length = robot_radius * 1.7
        heading_end = (round(robot_center[0] + heading_length * math.cos(env.pose.theta)), round(robot_center[1] - heading_length * math.sin(env.pose.theta)))
        pygame.draw.line(screen, (25, 51, 96), robot_center, heading_end, 3)
        pygame.draw.rect(screen, (40, 45, 52), map_rect, 2)

        status_text = status.replace("_", " ").title()
        title = self.title_font.render(f"BC-Chunk | action_horizon={action_horizon} | {status_text}", True, (35, 42, 52))
        screen.blit(title, (self.margin, 14))
        distance_to_goal = math.hypot(goal_x - env.pose.x, goal_y - env.pose.y)
        action = env.previous_action
        details = self.info_font.render(f"Step {env.step_count}   Distance {distance_to_goal:.2f} m   " f"v {float(action[0]):.2f} m/s   w {float(action[1]):.2f} rad/s", True, (65, 72, 82))
        screen.blit(details, (self.margin, screen_height - 29))

        return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2)).copy()
