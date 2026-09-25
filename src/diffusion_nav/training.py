import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .dataset import Normalizer, SequenceDataset
from .models import BC_Chunk


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


def create_reduce_on_plateau_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: dict | None = None) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    if scheduler_config is None:
        scheduler_config = {}
    if not isinstance(scheduler_config, dict):
        raise TypeError("lr_scheduler configuration must be a mapping")

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(scheduler_config.get("factor", 0.5)), patience=int(scheduler_config.get("patience", 3)),
        threshold=float(scheduler_config.get("threshold", 1e-4)), threshold_mode=str(scheduler_config.get("threshold_mode", "rel")), cooldown=int(scheduler_config.get("cooldown", 0)),
        min_lr=float(scheduler_config.get("min_lr", 1e-6)), eps=float(scheduler_config.get("eps", 1e-8)),
    )


def _prepare_chunk_batch(batch, device: torch.device, pred_horizon: int, observation_noise: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    if "obs" not in batch or "action" not in batch:
        raise KeyError("batch must contain obs and action")
    observations = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["action"], dtype=torch.float32, device=device)
    if observations.ndim != 3:
        raise ValueError("batch obs must have shape (batch, obs_horizon, obs_dim)")
    if actions.shape != (observations.shape[0], pred_horizon, 2):
        raise ValueError(f"batch action must have shape (batch, {pred_horizon}, 2)")
    if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
        raise ValueError("batch values must be finite")
    if observation_noise > 0:
        observations = observations + torch.randn_like(observations) * observation_noise
    return observations, actions


def train_bc_chunk_epoch(model: BC_Chunk, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device | str, observation_noise: float = 0.0) -> float:
    device = torch.device(device)
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch in dataloader:
        observations, target_actions = _prepare_chunk_batch(batch, device, model.pred_horizon, observation_noise)
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
def validate_bc_chunk(model: BC_Chunk, dataloader: DataLoader, device: torch.device | str) -> float:
    device = torch.device(device)
    model.eval()
    total_loss = 0.0
    sample_count = 0
    for batch in dataloader:
        observations, target_actions = _prepare_chunk_batch(batch, device, model.pred_horizon)
        loss = nn.functional.mse_loss(model(observations), target_actions)
        if not torch.isfinite(loss):
            raise FloatingPointError("validation loss is not finite")
        batch_size = observations.shape[0]
        total_loss += float(loss) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("validation dataloader cannot be empty")
    return total_loss / sample_count


def save_bc_chunk_checkpoint(checkpoint_path, model: BC_Chunk, optimizer: torch.optim.Optimizer, normalizer: Normalizer, epoch: int, metrics: dict[str, float]) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "policy_type": "bc_chunk",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {"obs_dim": model.obs_dim, "obs_horizon": model.obs_horizon, "pred_horizon": model.pred_horizon, "hidden_dim": model.hidden_dim, "dropout": model.dropout},
        "normalizer": normalizer.state_dict(),
        "epoch": int(epoch),
        "metrics": {name: float(value) for name, value in metrics.items()},
    }
    torch.save(checkpoint, checkpoint_path)


def load_config(config_path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError("configuration file must contain a mapping")
    for section in ("data", "model", "training", "output"):
        if not isinstance(config.get(section), dict):
            raise TypeError(f"configuration section {section} is required")
    return config


def save_loss_curve(history: list[dict[str, float]], output_path: str | Path) -> Path:
    if not history:
        raise ValueError("training history cannot be empty")
    required_metrics = ("train_loss", "validation_loss")
    for epoch_metrics in history:
        if any(name not in epoch_metrics for name in required_metrics):
            raise ValueError("each history entry must contain train_loss and validation_loss")
        if any(not np.isfinite(epoch_metrics[name]) for name in required_metrics):
            raise ValueError("loss history must contain only finite values")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    epochs = np.arange(1, len(history) + 1)
    figure = Figure(figsize=(8, 7))
    FigureCanvasAgg(figure)
    if all("learning_rate" in item for item in history):
        loss_axis, learning_rate_axis = figure.subplots(2, 1, sharex=True)
        learning_rate_axis.step(epochs, [item["learning_rate"] for item in history], where="post", color="darkgreen")
        learning_rate_axis.set_ylabel("Learning rate")
        learning_rate_axis.set_yscale("log")
        learning_rate_axis.set_xlabel("Epoch")
        learning_rate_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        learning_rate_axis.grid(True, alpha=0.3)
    else:
        loss_axis = figure.subplots()
        loss_axis.set_xlabel("Epoch")
        loss_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    loss_axis.plot(epochs, [item["train_loss"] for item in history], label="Training loss")
    loss_axis.plot(epochs, [item["validation_loss"] for item in history], label="Validation loss")
    loss_axis.set_title("Training Progress by Epoch")
    loss_axis.set_ylabel("Mean squared error")
    loss_axis.grid(True, alpha=0.3)
    loss_axis.legend()
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    return output_path


def train_bc_chunk(config: dict) -> tuple[BC_Chunk, Normalizer, list[dict[str, float]]]:
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]
    seed = int(training_config.get("seed", 0))
    set_seed(seed)
    device = choose_device(str(training_config.get("device", "auto")))

    obs_horizon = int(data_config.get("obs_horizon", 2))
    pred_horizon = int(data_config.get("pred_horizon", 16))
    train_dataset = SequenceDataset(data_config["train_path"], obs_horizon, pred_horizon)
    validation_dataset = SequenceDataset(data_config["validation_path"], obs_horizon, pred_horizon, normalizer=train_dataset.normalizer)
    overlapping_map_seeds = np.intersect1d(train_dataset.map_seeds, validation_dataset.map_seeds)
    if overlapping_map_seeds.size:
        raise ValueError(f"training and validation datasets share map seeds: {overlapping_map_seeds.tolist()}")
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(data_config.get("batch_size", 64))
    num_workers = int(data_config.get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    obs_dim = int(train_dataset.observations.shape[1])
    model = BC_Chunk(obs_dim=obs_dim, obs_horizon=obs_horizon, pred_horizon=pred_horizon, hidden_dim=int(model_config.get("hidden_dim", 256)),
        dropout=float(model_config.get("dropout", 0.0))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_config.get("learning_rate", 1e-3)), weight_decay=float(training_config.get("weight_decay", 0.0)))
    scheduler = create_reduce_on_plateau_scheduler(optimizer, training_config.get("lr_scheduler"))
    epochs = int(training_config.get("epochs", 50))
    if epochs <= 0 or batch_size <= 0 or num_workers < 0:
        raise ValueError("epochs and batch_size must be positive and num_workers cannot be negative")
    early_stopping_patience_value = training_config.get("early_stopping_patience")
    if early_stopping_patience_value is not None and (not isinstance(early_stopping_patience_value, int) or isinstance(early_stopping_patience_value, bool)):
        raise TypeError("early_stopping_patience must be an integer when configured")
    early_stopping_patience = early_stopping_patience_value
    early_stopping_min_delta = float(training_config.get("early_stopping_min_delta", 0.0))
    observation_noise = float(training_config.get("observation_noise", 0.0))
    if not np.isfinite(observation_noise) or observation_noise < 0.0:
        raise ValueError("observation_noise must be a finite non-negative number")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive when configured")
    if not np.isfinite(early_stopping_min_delta) or early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be a finite non-negative number")

    history = []
    best_validation_loss = float("inf")
    early_stopping_reference_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_bc_chunk_epoch(model, train_loader, optimizer, device, observation_noise)
        validation_loss = validate_bc_chunk(model, validation_loader, device)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step(validation_loss)
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        metrics = {"train_loss": train_loss, "validation_loss": validation_loss, "learning_rate": learning_rate}
        history.append(metrics)
        print(f"epoch {epoch:03d} | train_loss {train_loss:.6f} | " f"validation_loss {validation_loss:.6f} | lr {learning_rate:.2e}"
              + (f" -> {next_learning_rate:.2e}" if next_learning_rate < learning_rate else ""))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_bc_chunk_checkpoint(output_config["checkpoint_path"], model, optimizer, train_dataset.normalizer, epoch, metrics)
        if validation_loss < early_stopping_reference_loss - early_stopping_min_delta:
            early_stopping_reference_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"early stopping at epoch {epoch:03d} | " f"best_validation_loss {best_validation_loss:.6f}")
            break
    loss_curve_path = output_config.get("loss_curve_path")
    if loss_curve_path is None:
        checkpoint_path = Path(output_config["checkpoint_path"])
        loss_curve_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_loss.png")
    save_loss_curve(history, loss_curve_path)
    print(f"saved loss curve: {loss_curve_path}")
    best_checkpoint = torch.load(output_config["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    return model, train_dataset.normalizer, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a configured behavior cloning policy.")
    parser.add_argument("--config", default="configs/bc_chunk.yaml")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    model_type = config["model"].get("type", "BC_Chunk")
    if model_type != "BC_Chunk":
        raise ValueError(f"unsupported model type: {model_type}")
    train_bc_chunk(config)


if __name__ == "__main__":
    main()
