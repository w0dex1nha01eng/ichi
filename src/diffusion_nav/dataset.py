from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .env import NavigationEnv
from .export import PurePursuitConfig, PurePursuitExpert, Waypoint
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
        return {"observation_mean": self.observation_mean.copy(),"observation_std": self.observation_std.copy(),"action_mean": self.action_mean.copy(),"action_std": self.action_std.copy(),}

    @classmethod
    def from_state_dict(cls, state: dict[str, np.ndarray]):
        return cls(state["observation_mean"],state["observation_std"],state["action_mean"],state["action_std"])


def plan_expert_path(env: NavigationEnv, grid_step: float) -> tuple[Waypoint, ...] | None:
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
    return tuple(path)


def _empty_episode(env: NavigationEnv, map_seed: int) -> ExpertEpisode:
    observation_dim = int(env.observation_space.shape[0])
    return ExpertEpisode(np.empty((0, observation_dim),dtype=np.float32),np.empty((0, 2),dtype=np.float32),np.empty((0, 3),dtype=np.float32),int(map_seed),False,False,False)


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
        states.append(np.array([env.pose.x, env.pose.y, env.pose.theta],dtype=np.float32))
        action = expert.act(env.pose)
        observation, _, terminated, truncated, info = env.step(action)
        actions.append(env.previous_action.copy())
        success = bool(info["success"])
        collision = bool(info["collision"])
        if terminated or truncated:
            break

    return ExpertEpisode(np.stack(observations).astype(np.float32),np.stack(actions).astype(np.float32),np.stack(states).astype(np.float32),int(map_seed),success,collision,bool(truncated))


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

    observations = np.concatenate([episode.observations for episode in successful_episodes],axis=0).astype(np.float32)
    actions = np.concatenate([episode.actions for episode in successful_episodes],axis=0).astype(np.float32)
    states = np.concatenate([episode.states for episode in successful_episodes],axis=0).astype(np.float32)
    episode_ends = np.cumsum([len(episode.actions) for episode in successful_episodes],dtype=np.int64)
    map_seeds = np.asarray([episode.map_seed for episode in successful_episodes],dtype=np.int64)
    if normalizer is None:
        normalizer = Normalizer.from_data(observations, actions)
    if normalizer.observation_mean.shape != (observations.shape[1],):
        raise ValueError("normalizer observation dimension doesn't match dataset")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path,observations=observations,actions=actions,states=states,episode_ends=episode_ends,map_seeds=map_seeds,observation_mean=normalizer.observation_mean,observation_std=normalizer.observation_std,action_mean=normalizer.action_mean,action_std=normalizer.action_std)
    return normalizer


def load_expert_dataset(dataset_path) -> dict[str, np.ndarray]:
    required_keys = {"observations","actions","states","episode_ends","map_seeds","observation_mean","observation_std","action_mean","action_std"}
    with np.load(dataset_path, allow_pickle=False) as dataset_file:
        missing_keys = required_keys.difference(dataset_file.files)
        if missing_keys:
            raise ValueError(f"dataset is missing keys: {sorted(missing_keys)}")
        data = {key: dataset_file[key].copy() for key in required_keys}

    observations = np.asarray(data["observations"],dtype=np.float32)
    actions = np.asarray(data["actions"],dtype=np.float32)
    states = np.asarray(data["states"],dtype=np.float32)
    episode_ends = np.asarray(data["episode_ends"],dtype=np.int64)
    map_seeds = np.asarray(data["map_seeds"],dtype=np.int64)
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
        self.normalizer = normalizer or Normalizer(data["observation_mean"],data["observation_std"],data["action_mean"],data["action_std"])
        if self.normalizer.observation_mean.shape != (self.observations.shape[1],):
            raise ValueError("normalizer observation dimension doesn't match dataset")

        self.sample_indices = []
        episode_start = 0
        for episode_index, episode_end in enumerate(self.episode_ends):
            for current_index in range(episode_start, int(episode_end)):
                self.sample_indices.append((episode_start,int(episode_end),current_index,episode_index))
            episode_start = int(episode_end)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.int64]:
        episode_start, episode_end, current_index, episode_index = self.sample_indices[index]
        observation_indices = np.clip(np.arange(current_index - self.obs_horizon + 1,current_index + 1),episode_start,episode_end - 1)
        action_indices = np.clip(np.arange(current_index,current_index + self.pred_horizon),episode_start,episode_end - 1)
        observations = self.observations[observation_indices].copy()
        actions = self.actions[action_indices].copy()
        if self.normalize_data:
            observations = self.normalizer.normalize_observation(observations)
            actions = self.normalizer.normalize_action(actions)
        return {"obs": observations.astype(np.float32),"action": actions.astype(np.float32),"map_seed": np.int64(self.map_seeds[episode_index]),}
