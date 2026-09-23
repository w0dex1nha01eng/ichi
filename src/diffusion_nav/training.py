import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .dataset import Normalizer, SequenceDataset
from .models import BC_1


def choose_device(preferred: str = "auto") -> torch.device:
    preferred = preferred.lower()
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preferred == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if preferred == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if preferred not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return torch.device(preferred)


def set_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if "obs" not in batch or "action" not in batch:
        raise KeyError("batch must contain obs and action")
    observations = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["action"], dtype=torch.float32, device=device)
    if observations.ndim != 3:
        raise ValueError("batch obs must have shape (batch, obs_horizon, obs_dim)")
    if actions.ndim != 3 or actions.shape[0] != observations.shape[0] or actions.shape[2] != 2:
        raise ValueError("batch action must have shape (batch, pred_horizon, 2)")
    if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
        raise ValueError("batch values must be finite")
    return observations, actions[:, 0]


def train_bc_epoch(
    model: BC_1,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    device = torch.device(device)
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch in dataloader:
        observations, target_actions = _prepare_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predicted_actions = model(observations)
        loss = nn.functional.mse_loss(predicted_actions, target_actions)
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss is not finite")
        loss.backward()
        optimizer.step()
        batch_size = observations.shape[0]
        total_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("training dataloader cannot be empty")
    return total_loss / sample_count


@torch.inference_mode()
def validate_bc(model: BC_1, dataloader: DataLoader, device: torch.device | str) -> float:
    device = torch.device(device)
    model.eval()
    total_loss = 0.0
    sample_count = 0
    for batch in dataloader:
        observations, target_actions = _prepare_batch(batch, device)
        loss = nn.functional.mse_loss(model(observations), target_actions)
        if not torch.isfinite(loss):
            raise FloatingPointError("validation loss is not finite")
        batch_size = observations.shape[0]
        total_loss += float(loss) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("validation dataloader cannot be empty")
    return total_loss / sample_count


def save_bc_checkpoint(
    checkpoint_path,
    model: BC_1,
    optimizer: torch.optim.Optimizer,
    normalizer: Normalizer,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "obs_dim": model.obs_dim,
            "obs_horizon": model.obs_horizon,
            "hidden_dim": model.hidden_dim,
        },
        "normalizer": normalizer.state_dict(),
        "epoch": int(epoch),
        "metrics": {name: float(value) for name, value in metrics.items()},
    }
    torch.save(checkpoint, checkpoint_path)


def load_bc_checkpoint(
    checkpoint_path,
    model: BC_1,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    required_keys = {"model_state_dict", "model_config", "normalizer", "epoch", "metrics"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing_keys)}")
    expected_config = {
        "obs_dim": model.obs_dim,
        "obs_horizon": model.obs_horizon,
        "hidden_dim": model.hidden_dim,
    }
    if checkpoint["model_config"] != expected_config:
        raise ValueError("checkpoint model configuration doesn't match model")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("checkpoint doesn't contain optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_config(config_path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError("configuration file must contain a mapping")
    for section in ("data", "model", "training", "output"):
        if not isinstance(config.get(section), dict):
            raise TypeError(f"configuration section {section} is required")
    return config


def train_bc(config: dict) -> tuple[BC_1, Normalizer, list[dict[str, float]]]:
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]
    seed = int(training_config.get("seed", 0))
    set_seed(seed)
    device = choose_device(str(training_config.get("device", "auto")))

    obs_horizon = int(data_config.get("obs_horizon", 2))
    pred_horizon = int(data_config.get("pred_horizon", 1))
    train_dataset = SequenceDataset(data_config["train_path"], obs_horizon, pred_horizon)
    validation_dataset = SequenceDataset(
        data_config["validation_path"],
        obs_horizon,
        pred_horizon,
        normalizer=train_dataset.normalizer,
    )
    overlapping_map_seeds = np.intersect1d(train_dataset.map_seeds, validation_dataset.map_seeds)
    if overlapping_map_seeds.size:
        raise ValueError(
            f"training and validation datasets share map seeds: {overlapping_map_seeds.tolist()}"
        )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(data_config.get("batch_size", 64))
    num_workers = int(data_config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    obs_dim = int(train_dataset.observations.shape[1])
    model = BC_1(obs_dim, obs_horizon, int(model_config.get("hidden_dim", 256))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    epochs = int(training_config.get("epochs", 50))
    if epochs <= 0 or batch_size <= 0 or num_workers < 0:
        raise ValueError(
            "epochs and batch_size must be positive and num_workers cannot be negative"
        )

    history = []
    best_validation_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_bc_epoch(model, train_loader, optimizer, device)
        validation_loss = validate_bc(model, validation_loader, device)
        metrics = {"train_loss": train_loss, "validation_loss": validation_loss}
        history.append(metrics)
        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.6f} | validation_loss {validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_bc_checkpoint(
                output_config["checkpoint_path"],
                model,
                optimizer,
                train_dataset.normalizer,
                epoch,
                metrics,
            )
    return model, train_dataset.normalizer, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the one-step behavior cloning policy.")
    parser.add_argument("--config", default="configs/bc_one_step.yaml")
    arguments = parser.parse_args()
    train_bc(load_config(arguments.config))


if __name__ == "__main__":
    main()
