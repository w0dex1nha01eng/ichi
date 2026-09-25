import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from .dataset import create_random_environment, load_expert_dataset
from .evaluate import load_bc_chunk_policy, run_chunk_policy_episode
from .rendering import PygameRenderer
from .training import load_config


class _Mp4Writer:
    def __init__(self, path: Path, size: tuple[int, int], fps: int) -> None:
        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError("imageio-ffmpeg is required; install the visualization extra to record MP4 videos") from error

        self._writer = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec="libx264", quality=8, pix_fmt_in="rgb24", pix_fmt_out="yuv420p")
        self._writer.send(None)

    def write(self, frame: np.ndarray) -> None:
        self._writer.send(np.ascontiguousarray(frame, dtype=np.uint8))

    def close(self) -> None:
        self._writer.close()


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError(f"configuration must contain a mapping: {path}")
    return config


def _failure_reason(result) -> str | None:
    if result.success:
        return None
    if result.collision:
        return "collision"
    if result.truncated:
        return "timeout"
    return "other_failure"


def _create_frame_callback(renderer: PygameRenderer, writer: _Mp4Writer, action_horizon: int) -> Callable:
    trajectory: list[tuple[float, float]] = []

    def capture_frame(current_env, status: str) -> None:
        trajectory.append((current_env.pose.x, current_env.pose.y))
        writer.write(renderer.render(current_env, action_horizon, status, trajectory))

    return capture_frame

def record_failure_videos(config_path: str | Path = "configs/bc_chunk.yaml", dataset_config_path: str | Path = "configs/dataset_generation.yaml", output_dir: str | Path = "artifacts/failure_videos",
        videos_per_horizon: int = 3, action_horizons: tuple[int, ...] = (2, 4)) -> dict:
    if not isinstance(videos_per_horizon, int) or isinstance(videos_per_horizon, bool) or videos_per_horizon <= 0:
        raise ValueError("videos_per_horizon must be a positive integer")

    config = load_config(config_path)
    dataset_config = _load_yaml(dataset_config_path)
    validation_path = Path(config["data"]["validation_path"])
    checkpoint_path = Path(config["output"]["checkpoint_path"])
    if not validation_path.is_file():
        raise FileNotFoundError(f"validation dataset not found: {validation_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"BC-Chunk checkpoint not found: {checkpoint_path}")

    validation_data = load_expert_dataset(validation_path)
    map_seeds = np.unique(validation_data["map_seeds"]).astype(int).tolist()
    if not map_seeds:
        raise ValueError("validation dataset does not contain map seeds")

    environment_config = dataset_config["environment"]
    map_config = dataset_config["map_generation"]
    max_linear_velocity = float(environment_config["max_linear_velocity"])
    max_angular_velocity = float(environment_config["max_angular_velocity"])
    action_low = np.array([0.0, -max_angular_velocity], dtype=np.float32)
    action_high = np.array([max_linear_velocity, max_angular_velocity], dtype=np.float32)
    device = config["training"].get("device", "auto")
    policy = load_bc_chunk_policy(checkpoint_path, action_low, action_high, device)
    horizons = tuple(action_horizons)
    if not horizons or any(not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= policy.model.pred_horizon for horizon in horizons):
        raise ValueError(f"action_horizons must be integers from 1 to {policy.model.pred_horizon}")

    run_stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_dir) / run_stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    fps = max(1, round(1.0 / float(environment_config.get("dt", 0.1))))
    video_records = []
    search_summary = {}

    for action_horizon in horizons:
        selected_failures = []
        evaluated_count = 0
        for map_seed in map_seeds:
            env = create_random_environment(map_seed, environment_config, map_config)
            result = run_chunk_policy_episode(env, policy, action_horizon=action_horizon)
            evaluated_count += 1
            reason = _failure_reason(result)
            if reason is not None:
                selected_failures.append({"map_seed": map_seed, "reason": reason, "steps": result.steps})
                if len(selected_failures) == videos_per_horizon:
                    break
        search_summary[str(action_horizon)] = {"validation_maps_checked": evaluated_count, "failure_videos_requested": videos_per_horizon, "failures_found": len(selected_failures)}

        for video_index, failure in enumerate(selected_failures, start=1):
            map_seed = failure["map_seed"]
            env = create_random_environment(map_seed, environment_config, map_config)
            filename = f"action_horizon_{action_horizon}_failure_{video_index:02d}_" f"seed_{map_seed}_{failure['reason']}.mp4"
            video_path = run_dir / filename
            pending_path = run_dir / f".{filename}.pending.mp4"
            renderer = PygameRenderer()
            writer = _Mp4Writer(pending_path, renderer.size, fps)
            capture_frame = _create_frame_callback(renderer, writer, action_horizon)

            try:
                recorded_result = run_chunk_policy_episode(env, policy, action_horizon=action_horizon, frame_callback=capture_frame)
            finally:
                writer.close()

            recorded_reason = _failure_reason(recorded_result)
            if recorded_reason != failure["reason"]:
                pending_path.unlink(missing_ok=True)
                raise RuntimeError(f"episode replay changed for map seed {map_seed}: " f"expected {failure['reason']}, got {recorded_reason}")
            pending_path.replace(video_path)
            video_records.append({"action_horizon": action_horizon, "map_seed": map_seed, "reason": recorded_reason, "steps": recorded_result.steps, "video_path": str(video_path)})

    summary = {"checkpoint_path": str(checkpoint_path), "validation_path": str(validation_path), "run_dir": str(run_dir), "search": search_summary, "videos": video_records}
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Record pygame-rendered BC-Chunk validation failures as MP4.")
    parser.add_argument("--config", default="configs/bc_chunk.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset_generation.yaml")
    parser.add_argument("--output-dir", default="artifacts/failure_videos")
    parser.add_argument("--count", type=int, default=3, help="failure videos to record per action horizon")
    arguments = parser.parse_args()
    summary = record_failure_videos(config_path=arguments.config, dataset_config_path=arguments.dataset_config, output_dir=arguments.output_dir, videos_per_horizon=arguments.count)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
