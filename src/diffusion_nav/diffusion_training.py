import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import Normalizer, SequenceDataset
from .diffusion_policy import ConditionalDiffusionModel
from .training import choose_device, create_reduce_on_plateau_scheduler, load_config, save_loss_curve, set_seed


def _prepare_diffusion_batch(batch, device: torch.device, obs_horizon: int, obs_dim: int, pred_horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    if "obs" not in batch or "action" not in batch:
        raise KeyError("batch must contain obs and action")
    observations = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["action"], dtype=torch.float32, device=device)
    if observations.ndim != 3 or observations.shape[1:] != (obs_horizon, obs_dim):
        raise ValueError(f"batch obs must have shape (batch, {obs_horizon}, {obs_dim})")
    if actions.shape != (observations.shape[0], pred_horizon, 2):
        raise ValueError(f"batch action must have shape (batch, {pred_horizon}, 2)")
    if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
        raise ValueError("batch values must be finite")
    return observations, actions


def _make_generator(device: torch.device, seed: int) -> torch.Generator | None:
    if device.type not in {"cpu", "cuda"}:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def train_diffusion_epoch(model: ConditionalDiffusionModel, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device | str, generator: torch.Generator | None = None) -> float:
    device = torch.device(device)
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch in dataloader:
        observations, actions = _prepare_diffusion_batch(batch, device, model.obs_horizon, model.obs_dim, model.pred_horizon)
        optimizer.zero_grad(set_to_none=True)
        loss = model.diffusion_loss(observations, actions, generator=generator)
        if not torch.isfinite(loss):
            raise FloatingPointError("diffusion training loss is not finite")
        loss.backward()
        optimizer.step()
        batch_size = observations.shape[0]
        total_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("training dataloader cannot be empty")
    return total_loss / sample_count


@torch.inference_mode()
def validate_diffusion(model: ConditionalDiffusionModel, dataloader: DataLoader, device: torch.device | str, seed: int = 0) -> float:
    device = torch.device(device)
    model.eval()
    generator = _make_generator(device, seed)
    total_loss = 0.0
    sample_count = 0
    for batch in dataloader:
        observations, actions = _prepare_diffusion_batch(batch, device, model.obs_horizon, model.obs_dim, model.pred_horizon)
        loss = model.diffusion_loss(observations, actions, generator=generator)
        if not torch.isfinite(loss):
            raise FloatingPointError("diffusion validation loss is not finite")
        batch_size = observations.shape[0]
        total_loss += float(loss) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("validation dataloader cannot be empty")
    return total_loss / sample_count


def save_diffusion_checkpoint(checkpoint_path, model: ConditionalDiffusionModel, optimizer: torch.optim.Optimizer, normalizer: Normalizer, epoch: int, metrics: dict[str, float]) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "policy_type": "diffusion_policy",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config_dict(),
        "normalizer": normalizer.state_dict(),
        "epoch": int(epoch),
        "metrics": {name: float(value) for name, value in metrics.items()},
    }
    torch.save(checkpoint, checkpoint_path)


def train_diffusion_policy(config: dict) -> tuple[ConditionalDiffusionModel, Normalizer, list[dict[str, float]]]:
    if config["model"].get("type") != "DiffusionPolicy":
        raise ValueError("model.type must be DiffusionPolicy")
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]

    seed = int(training_config.get("seed", 0))
    validation_seed = int(training_config.get("validation_seed", seed + 1))
    set_seed(seed)
    device = choose_device(str(training_config.get("device", "auto")))
    obs_horizon = int(data_config.get("obs_horizon", 8))
    pred_horizon = int(data_config.get("pred_horizon", 32))
    train_dataset = SequenceDataset(data_config["train_path"], obs_horizon, pred_horizon)
    validation_dataset = SequenceDataset(data_config["validation_path"], obs_horizon, pred_horizon, normalizer=train_dataset.normalizer)
    overlapping_map_seeds = np.intersect1d(train_dataset.map_seeds, validation_dataset.map_seeds)
    if overlapping_map_seeds.size:
        raise ValueError(f"training and validation datasets share map seeds: {overlapping_map_seeds.tolist()}")

    batch_size = int(data_config.get("batch_size", 64))
    num_workers = int(data_config.get("num_workers", 0))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative")
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=loader_generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = ConditionalDiffusionModel(
        obs_dim=int(train_dataset.observations.shape[1]), obs_horizon=obs_horizon, pred_horizon=pred_horizon, hidden_dim=int(model_config.get("hidden_dim", 256)),
        time_embedding_dim=int(model_config.get("time_embedding_dim", 64)), diffusion_steps=int(model_config.get("diffusion_steps", 50)),
        schedule_type=str(model_config.get("schedule_type", "cosine")), beta_start=float(model_config.get("beta_start", 1e-4)), beta_end=float(model_config.get("beta_end", 2e-2)),
        dropout=float(model_config.get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_config.get("learning_rate", 3e-4)), weight_decay=float(training_config.get("weight_decay", 1e-4)))
    scheduler = create_reduce_on_plateau_scheduler(optimizer, training_config.get("lr_scheduler"))
    epochs = int(training_config.get("epochs", 100))
    patience = int(training_config.get("early_stopping_patience", 12))
    min_delta = float(training_config.get("early_stopping_min_delta", 1e-4))
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and early_stopping_patience must be positive")
    if not math_is_finite_non_negative(min_delta):
        raise ValueError("early_stopping_min_delta must be finite and non-negative")

    train_noise_generator = _make_generator(device, seed)
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    stopping_reference_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_diffusion_epoch(model, train_loader, optimizer, device, generator=train_noise_generator)
        validation_loss = validate_diffusion(model, validation_loader, device, seed=validation_seed)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step(validation_loss)
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        metrics = {"train_loss": train_loss, "validation_loss": validation_loss, "learning_rate": learning_rate}
        history.append(metrics)
        print(f"epoch {epoch:03d} | diffusion_train_loss {train_loss:.6f} | " f"diffusion_validation_loss {validation_loss:.6f} | lr {learning_rate:.2e}"
              + (f" -> {next_learning_rate:.2e}" if next_learning_rate < learning_rate else ""))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_diffusion_checkpoint(output_config["checkpoint_path"], model, optimizer, train_dataset.normalizer, epoch, metrics)
        if validation_loss < stopping_reference_loss - min_delta:
            stopping_reference_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"early stopping at epoch {epoch:03d} | " f"best_validation_loss {best_validation_loss:.6f}")
            break

    loss_curve_path = output_config.get("loss_curve_path")
    if loss_curve_path is None:
        checkpoint_path = Path(output_config["checkpoint_path"])
        loss_curve_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_loss.png")
    save_loss_curve(history, loss_curve_path)
    best_checkpoint = torch.load(output_config["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()
    print(f"saved diffusion loss curve: {loss_curve_path}")
    return model, train_dataset.normalizer, history


def math_is_finite_non_negative(value: float) -> bool:
    return bool(np.isfinite(value) and value >= 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conditional action diffusion policy.")
    parser.add_argument("--config", default="configs/diffusion_policy.yaml")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    train_diffusion_policy(config)


if __name__ == "__main__":
    main()
