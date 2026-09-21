import math
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle as RectanglePatch

from .collision import Rectangle, robot_in_collision
from .kinematics import Pose2D, Twist2D, integrate_pose
from .lidar import scan_lidar
from .observation import build_observation


class NavigationEnv(gym.Env):
    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self,width: float,height: float,robot_radius: float,start_pose: Pose2D,goal: tuple[float, float],obstacles: list[Rectangle],*,
        dt: float = 0.1,max_steps: int = 300,goal_radius: float = 0.3,num_rays: int = 16,lidar_max_range: float = 5.0,lidar_sample_step: float = 0.05,
        max_linear_velocity: float = 1.0,max_angular_velocity: float = 2.0) -> None:
        
        super().__init__()

        if width <= 0 or height <= 0:
            raise ValueError("map dimensions must be positive")
        if robot_radius <= 0:
            raise ValueError("robot_radius must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if goal_radius <= 0:
            raise ValueError("goal_radius must be positive")
        if num_rays <= 0:
            raise ValueError("num_rays must be positive")
        if lidar_max_range <= 0 or lidar_sample_step <= 0:
            raise ValueError("lidar parameters must be positive")
        if max_linear_velocity <= 0 or max_angular_velocity <= 0:
            raise ValueError("velocity limits must be positive")
        if len(goal) != 2 or not np.all(np.isfinite(goal)):
            raise ValueError("goal must contain two finite coordinates")

        goal_x, goal_y = float(goal[0]), float(goal[1])
        if not (0 <= goal_x <= width and 0 <= goal_y <= height):
            raise ValueError("goal must be inside the map")

        self.width = float(width)
        self.height = float(height)
        self.robot_radius = float(robot_radius)
        self.start_pose = start_pose
        self.goal = (goal_x, goal_y)
        self.obstacles = list(obstacles)

        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.goal_radius = float(goal_radius)

        self.num_rays = int(num_rays)
        self.lidar_max_range = float(lidar_max_range)
        self.lidar_sample_step = float(lidar_sample_step)

        self.action_space = spaces.Box(
            low=np.array([0.0, -max_angular_velocity], dtype=np.float32),
            high=np.array([max_linear_velocity, max_angular_velocity], dtype=np.float32),
            dtype=np.float32,
        )

        max_goal_distance = math.hypot(self.width, self.height)
        observation_low = np.concatenate([np.array([-max_goal_distance,-max_goal_distance,-1.0,-1.0,self.action_space.low[0],self.action_space.low[1],],
                                                   dtype=np.float32, ),np.zeros(self.num_rays, dtype=np.float32)])
        
        observation_high = np.concatenate([np.array([max_goal_distance,max_goal_distance,1.0,1.0,self.action_space.high[0],self.action_space.high[1],],
                                                    dtype=np.float32,),np.full(self.num_rays, self.lidar_max_range, dtype=np.float32),])
        
        self.observation_space = spaces.Box(low=observation_low,high=observation_high,dtype=np.float32,)

        self.pose = self.start_pose
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0

    def _distance_to_goal(self) -> float:
        goal_x, goal_y = self.goal
        return math.hypot(goal_x - self.pose.x, goal_y - self.pose.y)

    def _get_obs(self) -> np.ndarray:
        lidar_distances = scan_lidar(pose=self.pose,obstacles=self.obstacles,width=self.width,
                                     height=self.height,num_rays=self.num_rays,max_range=self.lidar_max_range,sample_step=self.lidar_sample_step,)
        return build_observation(pose=self.pose,goal=self.goal,
                                 previous_action=self.previous_action,lidar_distances=lidar_distances,)

    def _get_info(self,*,success: bool,collision: bool,distance_to_goal: float,) -> dict[str, bool | float | int]:
        return {"success": success,"collision": collision,"distance_to_goal": distance_to_goal,"step_count": self.step_count,}

    def reset(self,*,
              seed: int | None = None,options: dict | None = None,) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.pose = self.start_pose
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0

        distance_to_goal = self._distance_to_goal()
        observation = self._get_obs()
        info = self._get_info(success=False,collision=False,distance_to_goal=distance_to_goal)
        return observation, info

    def step(self,action: np.ndarray,) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"action shape must be (2,), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action must contain only finite values")

        executed_action = np.clip(action,self.action_space.low,self.action_space.high,).astype(np.float32)

        old_distance = self._distance_to_goal()
        twist = Twist2D(linear_velocity=float(executed_action[0]),angular_velocity=float(executed_action[1]))
        new_pose = integrate_pose(self.pose, twist, self.dt)

        self.pose = new_pose
        self.previous_action = executed_action.copy()
        self.step_count += 1

        collision = robot_in_collision(pose=self.pose,radius=self.robot_radius,obstacles=self.obstacles,width=self.width,height=self.height)
        new_distance = self._distance_to_goal()
        success = not collision and new_distance <= self.goal_radius

        terminated = collision or success
        truncated = self.step_count >= self.max_steps and not terminated

        reward = old_distance - new_distance
        if success:
            reward += 10.0
        elif collision:
            reward -= 10.0

        observation = self._get_obs()
        info = self._get_info(success=success,collision=collision,distance_to_goal=new_distance)

        return observation, float(reward), bool(terminated), bool(truncated), info

    def render(self, save_path=None):
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.set_xlim(0.0, self.width)
        ax.set_ylim(0.0, self.height)
        ax.set_aspect("equal")

        for rectangle in self.obstacles:
            obstacle_patch = RectanglePatch((rectangle.xmin, rectangle.ymin), rectangle.xmax - rectangle.xmin, rectangle.ymax - rectangle.ymin, facecolor="gray", edgecolor="black", alpha=0.7)
            ax.add_patch(obstacle_patch)

        goal_x, goal_y = self.goal
        goal_patch = Circle((goal_x, goal_y), self.goal_radius, facecolor="green", edgecolor="darkgreen", alpha=0.4)
        ax.add_patch(goal_patch)
        ax.plot(goal_x, goal_y, marker="x", color="darkgreen", markersize=8)

        robot_patch = Circle((self.pose.x, self.pose.y), self.robot_radius, facecolor="royalblue", edgecolor="navy", alpha=0.7)
        ax.add_patch(robot_patch)

        heading_length = self.robot_radius * 1.5
        heading_x = self.pose.x + heading_length * math.cos(self.pose.theta)
        heading_y = self.pose.y + heading_length * math.sin(self.pose.theta)
        ax.plot([self.pose.x, heading_x], [self.pose.y, heading_y], color="navy", linewidth=2)

        lidar_distances = scan_lidar(self.pose, self.obstacles, self.width, self.height, self.num_rays, self.lidar_max_range, self.lidar_sample_step)
        for ray_index, distance in enumerate(lidar_distances):
            ray_angle = self.pose.theta + 2.0 * math.pi * ray_index / self.num_rays
            ray_end_x = self.pose.x + float(distance) * math.cos(ray_angle)
            ray_end_y = self.pose.y + float(distance) * math.sin(ray_angle)
            ax.plot([self.pose.x, ray_end_x], [self.pose.y, ray_end_y], color="orange", linewidth=0.7, alpha=0.5)

        ax.set_xlabel("World x (m)")
        ax.set_ylabel("World y (m)")
        ax.set_title(f"Navigation environment - step {self.step_count}")
        ax.grid(alpha=0.2)

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig, ax
