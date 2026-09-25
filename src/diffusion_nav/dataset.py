import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .collision import Rectangle
from .env import NavigationEnv
from .export import PurePursuitConfig, PurePursuitExpert, Waypoint
from .kinematics import Pose2D
from .planning import a_star, grid_free, grid_to_world, world_to_grid


@dataclass(frozen=True, slots=True)
class ExpertEpisode:
    observations: np.ndarray
    actions: np.ndarray
    states: np.ndarray
    map_seed: int
    success: bool
    collision: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class Normalizer:
    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def __post_init__(self) -> None:
        observation_mean = np.asarray(self.observation_mean, dtype=np.float32)
        observation_std = np.asarray(self.observation_std, dtype=np.float32)
        action_mean = np.asarray(self.action_mean, dtype=np.float32)
        action_std = np.asarray(self.action_std, dtype=np.float32)

        if observation_mean.ndim != 1 or observation_std.shape != observation_mean.shape:
            raise ValueError("observation statistics must be matching one-dimensional arrays")
        if action_mean.shape != (2,) or action_std.shape != (2,):
            raise ValueError("action statistics must have shape (2,)")
        if not all(np.all(np.isfinite(values)) for values in (observation_mean, observation_std, action_mean, action_std)):
            raise ValueError("normalizer statistics must be finite")
        if np.any(observation_std <= 0) or np.any(action_std <= 0):
            raise ValueError("normalizer standard deviations must be positive")

        object.__setattr__(self, "observation_mean", observation_mean.copy())
        object.__setattr__(self, "observation_std", observation_std.copy())
        object.__setattr__(self, "action_mean", action_mean.copy())
        object.__setattr__(self, "action_std", action_std.copy())

    @classmethod
    def from_data(cls, observations, actions, epsilon: float = 1e-6):
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[0] == 0:
            raise ValueError("observations must be a non-empty two-dimensional array")
        if actions.ndim != 2 or actions.shape != (observations.shape[0], 2):
            raise ValueError("actions must have shape (number of observations, 2)")
        if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(actions)):
            raise ValueError("observations and actions must be finite")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number")

        observation_std = np.maximum(np.std(observations, axis=0), epsilon).astype(np.float32)
        action_std = np.maximum(np.std(actions, axis=0), epsilon).astype(np.float32)
        return cls(np.mean(observations, axis=0), observation_std, np.mean(actions, axis=0), action_std)

    def normalize_observation(self, observation) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape[-1:] != self.observation_mean.shape:
            raise ValueError("observation last dimension doesn't match normalizer")
        return ((observation - self.observation_mean) / self.observation_std).astype(np.float32)

    def normalize_action(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape[-1:] != (2,):
            raise ValueError("action last dimension must be 2")
        return ((action - self.action_mean) / self.action_std).astype(np.float32)

    def denormalize_action(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape[-1:] != (2,):
            raise ValueError("action last dimension must be 2")
        return (action * self.action_std + self.action_mean).astype(np.float32)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"observation_mean": self.observation_mean.copy(), "observation_std": self.observation_std.copy(), "action_mean": self.action_mean.copy(), "action_std": self.action_std.copy()}

    @classmethod
    def from_state_dict(cls, state: dict[str, np.ndarray]):
        return cls(state["observation_mean"], state["observation_std"], state["action_mean"], state["action_std"])


def plan_expert_path(env: NavigationEnv, grid_step: float, clearance_margin: float = 0.35) -> tuple[Waypoint, ...] | None:
    start = world_to_grid(env.pose.x, env.pose.y, grid_step)
    goal = world_to_grid(env.goal[0], env.goal[1], grid_step)
    if not grid_free(env.pose, start.x, start.y, grid_step, env.robot_radius, env.obstacles, env.width, env.height):
        return None
    if not grid_free(env.pose, goal.x, goal.y, grid_step, env.robot_radius, env.obstacles, env.width, env.height):
        return None

    grid_path = a_star(start, goal, env.pose, grid_step, env.robot_radius, env.obstacles, env.width, env.height)
    if grid_path is None:
        return None

    path = [grid_to_world(point.x, point.y, grid_step) for point in grid_path]
    if not path or path[-1] != env.goal:
        path.append(env.goal)
    return _simplify_path(env, path, clearance_margin)


def _empty_episode(env: NavigationEnv, map_seed: int) -> ExpertEpisode:
    observation_dim = int(env.observation_space.shape[0])
    return ExpertEpisode(np.empty((0, observation_dim), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float32), int(map_seed), False, False, False)


def run_expert_episode(env: NavigationEnv, expert: PurePursuitExpert, grid_step: float, map_seed: int) -> ExpertEpisode:
    observation, _ = env.reset(seed=map_seed)
    path = plan_expert_path(env, grid_step)
    if path is None:
        return _empty_episode(env, map_seed)

    expert.reset(path)
    observations = []
    actions = []
    states = []
    success = False
    collision = False
    truncated = False

    while True:
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        states.append(np.array([env.pose.x, env.pose.y, env.pose.theta], dtype=np.float32))
        action = expert.act(env.pose)
        observation, _, terminated, truncated, info = env.step(action)
        actions.append(env.previous_action.copy())
        success = bool(info["success"])
        collision = bool(info["collision"])
        if terminated or truncated:
            break

    return ExpertEpisode(np.stack(observations).astype(np.float32), np.stack(actions).astype(np.float32), np.stack(states).astype(np.float32),
        int(map_seed), success, collision, bool(truncated))


def collect_expert_episodes(env_factory: Callable[[int], NavigationEnv], map_seeds: Sequence[int], expert_config: PurePursuitConfig, grid_step: float) -> tuple[list[ExpertEpisode], dict[int, str]]:
    successful_episodes = []
    failed_episodes = {}
    for map_seed in map_seeds:
        env = env_factory(int(map_seed))
        expert = PurePursuitExpert(expert_config)
        episode = run_expert_episode(env, expert, grid_step, int(map_seed))
        if episode.success:
            successful_episodes.append(episode)
        elif episode.collision:
            failed_episodes[int(map_seed)] = "collision"
        elif episode.truncated:
            failed_episodes[int(map_seed)] = "timeout"
        else:
            failed_episodes[int(map_seed)] = "planning_failed"
    return successful_episodes, failed_episodes


def _validate_episode(episode: ExpertEpisode) -> None:
    if episode.observations.ndim != 2 or episode.observations.shape[0] == 0:
        raise ValueError("episode observations must be a non-empty two-dimensional array")
    if episode.actions.shape != (episode.observations.shape[0], 2):
        raise ValueError("episode actions must have shape (episode length, 2)")
    if episode.states.shape != (episode.observations.shape[0], 3):
        raise ValueError("episode states must have shape (episode length, 3)")
    if not all(np.all(np.isfinite(values)) for values in (episode.observations, episode.actions, episode.states)):
        raise ValueError("episode arrays must be finite")


def save_expert_dataset(episodes: Sequence[ExpertEpisode], output_path, normalizer: Normalizer | None = None) -> Normalizer:
    successful_episodes = [episode for episode in episodes if episode.success]
    if len(successful_episodes) == 0:
        raise ValueError("at least one successful episode is required")
    for episode in successful_episodes:
        _validate_episode(episode)

    observation_dims = {episode.observations.shape[1] for episode in successful_episodes}
    if len(observation_dims) != 1:
        raise ValueError("all episodes must use the same observation dimension")

    observations = np.concatenate([episode.observations for episode in successful_episodes], axis=0).astype(np.float32)
    actions = np.concatenate([episode.actions for episode in successful_episodes], axis=0).astype(np.float32)
    states = np.concatenate([episode.states for episode in successful_episodes], axis=0).astype(np.float32)
    episode_ends = np.cumsum([len(episode.actions) for episode in successful_episodes], dtype=np.int64)
    map_seeds = np.asarray([episode.map_seed for episode in successful_episodes], dtype=np.int64)
    if normalizer is None:
        normalizer = Normalizer.from_data(observations, actions)
    if normalizer.observation_mean.shape != (observations.shape[1],):
        raise ValueError("normalizer observation dimension doesn't match dataset")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, observations=observations, actions=actions, states=states,
        episode_ends=episode_ends, map_seeds=map_seeds, observation_mean=normalizer.observation_mean,
        observation_std=normalizer.observation_std, action_mean=normalizer.action_mean, action_std=normalizer.action_std,
    )
    return normalizer


def load_expert_dataset(dataset_path) -> dict[str, np.ndarray]:
    required_keys = {"observations", "actions", "states", "episode_ends", "map_seeds", "observation_mean", "observation_std", "action_mean", "action_std"}
    with np.load(dataset_path, allow_pickle=False) as dataset_file:
        missing_keys = required_keys.difference(dataset_file.files)
        if missing_keys:
            raise ValueError(f"dataset is missing keys: {sorted(missing_keys)}")
        data = {key: dataset_file[key].copy() for key in required_keys}

    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    states = np.asarray(data["states"], dtype=np.float32)
    episode_ends = np.asarray(data["episode_ends"], dtype=np.int64)
    map_seeds = np.asarray(data["map_seeds"], dtype=np.int64)
    if observations.ndim != 2 or observations.shape[0] == 0:
        raise ValueError("dataset observations must be a non-empty two-dimensional array")
    if actions.shape != (observations.shape[0], 2) or states.shape != (observations.shape[0], 3):
        raise ValueError("dataset actions or states have an invalid shape")
    if episode_ends.ndim != 1 or map_seeds.shape != episode_ends.shape or len(episode_ends) == 0:
        raise ValueError("episode_ends and map_seeds must be matching one-dimensional arrays")
    if np.any(np.diff(episode_ends) <= 0) or episode_ends[-1] != len(observations):
        raise ValueError("episode_ends must be strictly increasing and finish at dataset length")
    if not all(np.all(np.isfinite(values)) for values in (observations, actions, states)):
        raise ValueError("dataset arrays must be finite")

    data["observations"] = observations
    data["actions"] = actions
    data["states"] = states
    data["episode_ends"] = episode_ends
    data["map_seeds"] = map_seeds
    return data


def _point_rectangle_distance(x: float, y: float, rectangle: Rectangle) -> float:
    nearest_x = min(max(x, rectangle.xmin), rectangle.xmax)
    nearest_y = min(max(y, rectangle.ymin), rectangle.ymax)
    return math.hypot(x - nearest_x, y - nearest_y)


def _rectangle_gap(first: Rectangle, second: Rectangle) -> float:
    gap_x = max(first.xmin - second.xmax, second.xmin - first.xmax, 0.0)
    gap_y = max(first.ymin - second.ymax, second.ymin - first.ymax, 0.0)
    return math.hypot(gap_x, gap_y)


def _point_segment_distance(x: float, y: float, start_x: float, start_y: float, end_x: float, end_y: float) -> float:
    segment_x = end_x - start_x
    segment_y = end_y - start_y
    if segment_x == 0.0 and segment_y == 0.0:
        return math.hypot(x - start_x, y - start_y)
    projection = ((x - start_x) * segment_x + (y - start_y) * segment_y) / (segment_x * segment_x + segment_y * segment_y)
    projection = min(max(projection, 0.0), 1.0)
    return math.hypot(x - start_x - projection * segment_x, y - start_y - projection * segment_y)


def _segments_intersect(first_start, first_end, second_start, second_end) -> bool:
    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

    first_left = cross(second_start, second_end, first_start)
    first_right = cross(second_start, second_end, first_end)
    second_left = cross(first_start, first_end, second_start)
    second_right = cross(first_start, first_end, second_end)
    return (first_left > 0) != (first_right > 0) and (second_left > 0) != (second_right > 0)


def _segment_collides_with_rectangle(start_x: float, start_y: float, end_x: float, end_y: float, rectangle: Rectangle, radius: float) -> bool:
    for x, y in ((start_x, start_y), (end_x, end_y)):
        if rectangle.xmin <= x <= rectangle.xmax and rectangle.ymin <= y <= rectangle.ymax:
            return True
    corners = (
        (rectangle.xmin, rectangle.ymin),
        (rectangle.xmax, rectangle.ymin),
        (rectangle.xmax, rectangle.ymax),
        (rectangle.xmin, rectangle.ymax),
    )
    for index in range(4):
        if _segments_intersect((start_x, start_y), (end_x, end_y), corners[index], corners[(index + 1) % 4]):
            return True
    clearance = min(_point_rectangle_distance(start_x, start_y, rectangle), _point_rectangle_distance(end_x, end_y, rectangle))
    for corner_x, corner_y in corners:
        clearance = min(clearance, _point_segment_distance(corner_x, corner_y, start_x, start_y, end_x, end_y))
    return clearance < radius


def _simplify_path(env: NavigationEnv, path: Sequence[Waypoint], clearance_margin: float = 0.35) -> tuple[Waypoint, ...]:
    if not math.isfinite(clearance_margin) or clearance_margin < 0:
        raise ValueError("clearance_margin must be a finite non-negative number")
    if len(path) <= 2:
        return tuple(path)
    simplified = [path[0]]
    anchor_index = 0
    while anchor_index < len(path) - 1:
        next_index = anchor_index + 1
        for candidate_index in range(len(path) - 1, anchor_index, -1):
            start_x, start_y = path[anchor_index]
            end_x, end_y = path[candidate_index]
            blocked = any(_segment_collides_with_rectangle(start_x, start_y, end_x, end_y, obstacle, env.robot_radius + clearance_margin) for obstacle in env.obstacles)
            if not blocked:
                next_index = candidate_index
                break
        simplified.append(path[next_index])
        anchor_index = next_index
    return tuple(simplified)


def create_random_environment(map_seed: int, environment_config: dict, map_config: dict) -> NavigationEnv:
    if not isinstance(map_seed, int) or isinstance(map_seed, bool):
        raise TypeError("map_seed must be an integer")

    width = float(environment_config["width"])
    height = float(environment_config["height"])
    robot_radius = float(environment_config["robot_radius"])
    start_values = tuple(float(value) for value in environment_config["start_pose"])
    goal = tuple(float(value) for value in environment_config["goal"])
    if len(start_values) != 3 or len(goal) != 2:
        raise ValueError("start_pose must have three values and goal must have two")
    if not all(math.isfinite(value) for value in (*start_values, *goal)):
        raise ValueError("start_pose and goal values must be finite")

    obstacle_count = int(map_config["obstacle_count"])
    placement_attempts = int(map_config["placement_attempts"])
    obstacle_width_range = tuple(float(value) for value in map_config["obstacle_width_range"])
    obstacle_height_range = tuple(float(value) for value in map_config["obstacle_height_range"])
    if obstacle_count < 0 or placement_attempts <= 0:
        raise ValueError("obstacle_count cannot be negative and placement_attempts must be positive")
    if len(obstacle_width_range) != 2 or len(obstacle_height_range) != 2:
        raise ValueError("obstacle size ranges must each contain a minimum and maximum")
    if min(*obstacle_width_range, *obstacle_height_range) <= 0:
        raise ValueError("obstacle size ranges must be positive")
    if obstacle_width_range[0] > obstacle_width_range[1] or obstacle_height_range[0] > obstacle_height_range[1]:
        raise ValueError("obstacle size range minimum cannot exceed its maximum")

    rng = np.random.default_rng(map_seed)
    obstacles = []
    start_x, start_y, _ = start_values
    goal_x, goal_y = goal
    start_goal_clearance = robot_radius + float(map_config["clearance_from_start_goal"])
    obstacle_separation = float(map_config["obstacle_separation"])
    map_margin = robot_radius

    for _ in range(obstacle_count):
        for _ in range(placement_attempts):
            obstacle_width = float(rng.uniform(*obstacle_width_range))
            obstacle_height = float(rng.uniform(*obstacle_height_range))
            max_x = width - map_margin - obstacle_width
            max_y = height - map_margin - obstacle_height
            if max_x < map_margin or max_y < map_margin:
                raise ValueError("obstacle size range does not fit inside the map")

            xmin = float(rng.uniform(map_margin, max_x))
            ymin = float(rng.uniform(map_margin, max_y))
            candidate = Rectangle(xmin, ymin, xmin + obstacle_width, ymin + obstacle_height)
            if _point_rectangle_distance(start_x, start_y, candidate) < start_goal_clearance:
                continue
            if _point_rectangle_distance(goal_x, goal_y, candidate) < start_goal_clearance:
                continue
            if any(_rectangle_gap(candidate, placed) < obstacle_separation for placed in obstacles):
                continue
            obstacles.append(candidate)
            break
        else:
            raise RuntimeError(f"could not place obstacle {len(obstacles) + 1} for map seed {map_seed}")

    return NavigationEnv(
        width=width, height=height, robot_radius=robot_radius, start_pose=Pose2D(*start_values), goal=goal, obstacles=obstacles,
        dt=float(environment_config.get("dt", 0.1)), max_steps=int(environment_config.get("max_steps", 300)), goal_radius=float(environment_config.get("goal_radius", 0.3)),
        num_rays=int(environment_config.get("num_rays", 16)), lidar_max_range=float(environment_config.get("lidar_max_range", 5.0)),
        lidar_sample_step=float(environment_config.get("lidar_sample_step", 0.05)),
        max_linear_velocity=float(environment_config.get("max_linear_velocity", 1.0)), max_angular_velocity=float(environment_config.get("max_angular_velocity", 2.0)),
    )


def randomize_start_goal(map_seed: int, environment_config: dict, randomization_config: dict | None) -> dict:
    randomized_config = dict(environment_config)
    if randomization_config is None:
        return randomized_config

    map_seed = int(map_seed)
    endpoint_rng = np.random.default_rng(np.random.SeedSequence([map_seed, 94821]))
    ranges = {}
    for name in ("start_x_range", "start_y_range", "goal_x_range", "goal_y_range"):
        values = randomization_config.get(name)
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"{name} must contain a minimum and maximum")
        low, high = float(values[0]), float(values[1])
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"{name} must contain finite values in ascending order")
        ranges[name] = (low, high)

    width = float(environment_config["width"])
    height = float(environment_config["height"])
    robot_radius = float(environment_config["robot_radius"])
    for name in ("start_x_range", "start_y_range"):
        low, high = ranges[name]
        upper_bound = width if name == "start_x_range" else height
        if low < robot_radius or high > upper_bound - robot_radius:
            raise ValueError(f"{name} must keep the robot inside the map")
    for name in ("goal_x_range", "goal_y_range"):
        low, high = ranges[name]
        upper_bound = width if name == "goal_x_range" else height
        if low < 0.0 or high > upper_bound:
            raise ValueError(f"{name} must keep the goal inside the map")

    start_x = float(endpoint_rng.uniform(*ranges["start_x_range"]))
    start_y = float(endpoint_rng.uniform(*ranges["start_y_range"]))
    goal_x = float(endpoint_rng.uniform(*ranges["goal_x_range"]))
    goal_y = float(endpoint_rng.uniform(*ranges["goal_y_range"]))
    if "start_heading_range" in randomization_config:
        heading_values = randomization_config["start_heading_range"]
        if not isinstance(heading_values, (list, tuple)) or len(heading_values) != 2:
            raise ValueError("start_heading_range must contain a minimum and maximum")
        heading_low, heading_high = float(heading_values[0]), float(heading_values[1])
        if not math.isfinite(heading_low) or not math.isfinite(heading_high) or heading_low > heading_high:
            raise ValueError("start_heading_range must contain finite values in ascending order")
        heading = float(endpoint_rng.uniform(heading_low, heading_high))
    else:
        heading = float(randomization_config.get("start_heading", 0.0))
        if not math.isfinite(heading):
            raise ValueError("start_heading must be finite")
    randomized_config["start_pose"] = [start_x, start_y, heading]
    randomized_config["goal"] = [goal_x, goal_y]
    return randomized_config


def _collect_random_split(split_name: str, split_config: dict, environment_config: dict, map_config: dict, expert_config: PurePursuitConfig,
        grid_step: float, used_map_signatures: set[tuple[tuple[float, ...], ...]], endpoint_randomization: dict | None = None) -> tuple[list[ExpertEpisode], dict[int, str]]:
    requested_count = int(split_config["episode_count"])
    seed_start = int(split_config["seed_start"])
    max_attempts = int(split_config["max_attempts"])
    if requested_count <= 0 or max_attempts < requested_count:
        raise ValueError(f"{split_name} episode_count must be positive and max_attempts must cover it")

    episodes = []
    failures = {}
    for map_seed in range(seed_start, seed_start + max_attempts):
        try:
            randomized_environment_config = randomize_start_goal(map_seed, environment_config, endpoint_randomization)
            environment = create_random_environment(map_seed, randomized_environment_config, map_config)
        except RuntimeError as error:
            failures[map_seed] = str(error)
            continue

        signature = tuple(sorted((round(obstacle.xmin, 8), round(obstacle.ymin, 8), round(obstacle.xmax, 8), round(obstacle.ymax, 8)) for obstacle in environment.obstacles))
        if signature in used_map_signatures:
            failures[map_seed] = "duplicate_map"
            continue

        episode = run_expert_episode(environment, PurePursuitExpert(expert_config), grid_step, map_seed)
        if episode.success:
            episodes.append(episode)
            used_map_signatures.add(signature)
        elif episode.collision:
            failures[map_seed] = "collision"
        elif episode.truncated:
            failures[map_seed] = "timeout"
        else:
            failures[map_seed] = "planning_failed"

        if len(episodes) == requested_count:
            break

    if len(episodes) != requested_count:
        raise RuntimeError(
            f"{split_name} generated {len(episodes)} of {requested_count} requested successful episodes; "
            f"increase max_attempts or review the map settings"
        )
    return episodes, failures


def generate_random_datasets(config: dict) -> dict:
    environment_config = config["environment"]
    map_config = config["map_generation"]
    expert_config = PurePursuitConfig(**config["expert"])
    grid_step = float(map_config["grid_step"])
    endpoint_randomization = config.get("start_goal_randomization")
    splits = config["splits"]
    train_config = splits["train"]
    validation_config = splits["validation"]

    train_seed_range = set(range(int(train_config["seed_start"]), int(train_config["seed_start"]) + int(train_config["max_attempts"])))
    validation_seed_range = set(range(int(validation_config["seed_start"]), int(validation_config["seed_start"]) + int(validation_config["max_attempts"])))
    if train_seed_range.intersection(validation_seed_range):
        raise ValueError("train and validation seed ranges must not overlap")

    used_map_signatures: set[tuple[tuple[float, ...], ...]] = set()
    train_episodes, train_failures = _collect_random_split("train", train_config, environment_config, map_config, expert_config, grid_step, used_map_signatures, endpoint_randomization)
    validation_episodes, validation_failures = _collect_random_split("validation", validation_config, environment_config, map_config, expert_config, grid_step,
        used_map_signatures, endpoint_randomization)

    train_path = Path(train_config["output_path"])
    validation_path = Path(validation_config["output_path"])
    save_expert_dataset(train_episodes, train_path)
    save_expert_dataset(validation_episodes, validation_path)
    summary = {
        "start_goal_randomization": endpoint_randomization,
        "train": {
            "dataset_path": str(train_path), "episode_count": len(train_episodes),
            "map_seeds": [episode.map_seed for episode in train_episodes], "start_poses": [episode.states[0].tolist() for episode in train_episodes],
            "goals": [randomize_start_goal(episode.map_seed, environment_config, endpoint_randomization)["goal"] for episode in train_episodes],
            "failed_attempts": {str(seed): reason for seed, reason in train_failures.items()},
        },
        "validation": {
            "dataset_path": str(validation_path), "episode_count": len(validation_episodes),
            "map_seeds": [episode.map_seed for episode in validation_episodes], "start_poses": [episode.states[0].tolist() for episode in validation_episodes],
            "goals": [randomize_start_goal(episode.map_seed, environment_config, endpoint_randomization)["goal"] for episode in validation_episodes],
            "failed_attempts": {str(seed): reason for seed, reason in validation_failures.items()},
        },
    }
    summary_path = Path(config.get("summary_path", "artifacts/dataset_generation.json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random-map expert train and validation datasets.")
    parser.add_argument("--config", default="configs/dataset_generation.yaml")
    arguments = parser.parse_args()
    with Path(arguments.config).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError("dataset generation configuration must contain a mapping")

    summary = generate_random_datasets(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


class SequenceDataset:
    def __init__(self, dataset_path, obs_horizon: int = 2, pred_horizon: int = 16, *, normalize_data: bool = True, normalizer: Normalizer | None = None) -> None:
        if obs_horizon <= 0 or pred_horizon <= 0:
            raise ValueError("sequence horizons must be positive")

        data = load_expert_dataset(dataset_path)
        self.observations = data["observations"]
        self.actions = data["actions"]
        self.episode_ends = data["episode_ends"]
        self.map_seeds = data["map_seeds"]
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.normalize_data = bool(normalize_data)
        self.normalizer = normalizer or Normalizer(data["observation_mean"], data["observation_std"], data["action_mean"], data["action_std"])
        if self.normalizer.observation_mean.shape != (self.observations.shape[1],):
            raise ValueError("normalizer observation dimension doesn't match dataset")

        self.sample_indices = []
        episode_start = 0
        for episode_index, episode_end in enumerate(self.episode_ends):
            for current_index in range(episode_start, int(episode_end)):
                self.sample_indices.append((episode_start, int(episode_end), current_index, episode_index))
            episode_start = int(episode_end)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.int64]:
        episode_start, episode_end, current_index, episode_index = self.sample_indices[index]
        observation_indices = np.clip(np.arange(current_index - self.obs_horizon + 1, current_index + 1), episode_start, episode_end - 1)
        action_indices = np.clip(np.arange(current_index, current_index + self.pred_horizon), episode_start, episode_end - 1)
        observations = self.observations[observation_indices].copy()
        actions = self.actions[action_indices].copy()
        if self.normalize_data:
            observations = self.normalizer.normalize_observation(observations)
            actions = self.normalizer.normalize_action(actions)
        return {"obs": observations.astype(np.float32), "action": actions.astype(np.float32), "map_seed": np.int64(self.map_seeds[episode_index])}
