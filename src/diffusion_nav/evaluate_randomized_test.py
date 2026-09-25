import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .dataset import create_random_environment, load_expert_dataset, plan_expert_path, randomize_start_goal
from .evaluate import load_bc_chunk_policy, run_chunk_policy_episode
from .training import load_config


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError(f"configuration must contain a mapping: {path}")
    return config


def _map_signature(environment) -> tuple[tuple[float, ...], ...]:
    return tuple(sorted((round(obstacle.xmin, 8), round(obstacle.ymin, 8), round(obstacle.xmax, 8), round(obstacle.ymax, 8)) for obstacle in environment.obstacles))


def _sample_range(test_config: dict, name: str) -> tuple[float, float]:
    values = test_config[name]
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} must contain a minimum and maximum")
    low, high = float(values[0]), float(values[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{name} must contain finite values in ascending order")
    return low, high


def _randomized_environment(map_seed: int, environment_config: dict, map_config: dict, test_config: dict):
    randomized_environment_config = randomize_start_goal(map_seed, environment_config, test_config)
    environment = create_random_environment(map_seed, randomized_environment_config, map_config)
    return environment


def _existing_map_signatures(training_config: dict, dataset_config: dict) -> tuple[set[int], set[tuple[tuple[float, ...], ...]]]:
    map_seeds: set[int] = set()
    signatures: set[tuple[tuple[float, ...], ...]] = set()
    environment_config = dataset_config["environment"]
    map_config = dataset_config["map_generation"]
    endpoint_randomization = dataset_config.get("start_goal_randomization")
    for split_name in ("train", "validation"):
        dataset_path = training_config["data"][f"{split_name}_path"]
        dataset = load_expert_dataset(dataset_path)
        for map_seed in np.unique(dataset["map_seeds"]).astype(int):
            map_seeds.add(int(map_seed))
            randomized_environment_config = randomize_start_goal(int(map_seed), environment_config, endpoint_randomization)
            environment = create_random_environment(int(map_seed), randomized_environment_config, map_config)
            signatures.add(_map_signature(environment))
    return map_seeds, signatures


def _generate_test_environments(training_config: dict, dataset_config: dict, test_config: dict) -> tuple[list[tuple[int, object]], dict[str, int]]:
    episode_count = int(test_config["episode_count"])
    seed_start = int(test_config["seed_start"])
    max_attempts = int(test_config["max_attempts"])
    if episode_count <= 0 or max_attempts < episode_count:
        raise ValueError("max_attempts must be at least the positive episode_count")

    environment_config = dataset_config["environment"]
    map_config = dataset_config["map_generation"]
    grid_step = float(map_config["grid_step"])
    used_seeds, used_signatures = _existing_map_signatures(training_config, dataset_config)
    environments = []
    rejected: dict[str, int] = {}

    for map_seed in range(seed_start, seed_start + max_attempts):
        if map_seed in used_seeds:
            rejected["overlapping_seed"] = rejected.get("overlapping_seed", 0) + 1
            continue
        try:
            environment = _randomized_environment(map_seed, environment_config, map_config, test_config)
        except RuntimeError:
            rejected["obstacle_generation_failed"] = rejected.get("obstacle_generation_failed", 0) + 1
            continue

        signature = _map_signature(environment)
        if signature in used_signatures:
            rejected["duplicate_map"] = rejected.get("duplicate_map", 0) + 1
            continue
        if plan_expert_path(environment, grid_step) is None:
            rejected["no_expert_path"] = rejected.get("no_expert_path", 0) + 1
            continue

        environments.append((map_seed, environment))
        used_seeds.add(map_seed)
        used_signatures.add(signature)
        if len(environments) == episode_count:
            return environments, rejected

    raise RuntimeError(f"generated {len(environments)} of {episode_count} test maps after {max_attempts} attempts; " f"rejections: {rejected}")


def evaluate_randomized_test(config: dict) -> dict:
    training_config = load_config(config["training_config"])
    dataset_config = _load_yaml(config["dataset_config"])
    test_config = config["test"]
    checkpoint_path = Path(training_config["output"]["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"BC-Chunk checkpoint not found: {checkpoint_path}")

    environments, rejected = _generate_test_environments(training_config, dataset_config, test_config)
    environment_settings = dataset_config["environment"]
    action_low = np.array([0.0, -float(environment_settings["max_angular_velocity"])], dtype=np.float32)
    action_high = np.array([float(environment_settings["max_linear_velocity"]), float(environment_settings["max_angular_velocity"])], dtype=np.float32)
    policy = load_bc_chunk_policy(checkpoint_path, action_low, action_high, training_config["training"].get("device", "auto"))
    action_horizon = int(training_config.get("evaluation", {}).get("action_horizon", 1))
    records = []

    for map_seed, environment in environments:
        start_pose = [environment.start_pose.x, environment.start_pose.y, environment.start_pose.theta]
        goal = list(environment.goal)
        result = run_chunk_policy_episode(environment, policy, action_horizon=action_horizon)
        final_state = result.states[-1]
        final_distance = math.hypot(goal[0] - float(final_state[0]), goal[1] - float(final_state[1]))
        records.append({"map_seed": map_seed, "start_pose": start_pose, "goal": goal, "success": result.success, "collision": result.collision,
                        "truncated": result.truncated, "steps": result.steps, "final_distance_to_goal": final_distance})

    episode_count = len(records)
    success_count = sum(record["success"] for record in records)
    collision_count = sum(record["collision"] for record in records)
    timeout_count = sum(record["truncated"] for record in records)
    summary = {
        "checkpoint_path": str(checkpoint_path), "training_config_path": config["training_config"],
        "dataset_generation_config_path": config["dataset_config"], "test_episode_count": episode_count,
        "test_seed_start": int(test_config["seed_start"]), "action_horizon": action_horizon,
        "start_x_range": list(_sample_range(test_config, "start_x_range")), "start_y_range": list(_sample_range(test_config, "start_y_range")),
        "goal_x_range": list(_sample_range(test_config, "goal_x_range")), "goal_y_range": list(_sample_range(test_config, "goal_y_range")),
        "start_heading": float(test_config["start_heading"]) if "start_heading" in test_config else None,
        "start_heading_range": list(_sample_range(test_config, "start_heading_range")) if "start_heading_range" in test_config else None,
        "successful_episodes": success_count, "collision_episodes": collision_count, "timeout_episodes": timeout_count,
        "success_rate": success_count / episode_count, "success_rate_percent": 100.0 * success_count / episode_count,
        "mean_steps_all_episodes": float(np.mean([record["steps"] for record in records])),
        "mean_steps_successful_episodes": float(np.mean([record["steps"] for record in records if record["success"]])) if success_count else None,
        "mean_final_distance_to_goal": float(np.mean([record["final_distance_to_goal"] for record in records])),
        "rejected_map_candidates": rejected, "episodes": records,
    }

    report_path = Path(config["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"test success rate: {success_count}/{episode_count} " f"({summary['success_rate_percent']:.2f}%) | " f"collision {collision_count} | timeout {timeout_count}")
    print(f"saved report: {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BC-Chunk on independent randomized start/goal maps.")
    parser.add_argument("--config", default="configs/bc_chunk_randomized_test.yaml")
    arguments = parser.parse_args()
    evaluate_randomized_test(_load_yaml(arguments.config))


if __name__ == "__main__":
    main()
