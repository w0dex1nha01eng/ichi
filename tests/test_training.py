import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import diffusion_nav.training as training_module
from diffusion_nav.dataset import (
    ExpertEpisode,
    Normalizer,
    SequenceDataset,
    save_expert_dataset,
)
from diffusion_nav.models import BC_Chunk
from diffusion_nav.training import (
    choose_device,
    create_reduce_on_plateau_scheduler,
    save_bc_chunk_checkpoint,
    set_seed,
    train_bc_chunk,
    train_bc_chunk_epoch,
    validate_bc_chunk,
)
from diffusion_nav.validate_bc_chunk import save_validation_result, validate_saved_bc_chunk


def test_reduce_on_plateau_lowers_learning_rate_when_validation_stalls() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scheduler = create_reduce_on_plateau_scheduler(
        optimizer,
        {"factor": 0.5, "patience": 0, "threshold": 0.0, "min_lr": 1e-5},
    )

    scheduler.step(1.0)
    scheduler.step(1.1)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


class ChunkImitationDataset(Dataset):
    """Synthetic demonstrations with distinct expert actions across the horizon."""

    def __init__(self, sample_count: int = 48, pred_horizon: int = 4) -> None:
        generator = torch.Generator().manual_seed(31)
        self.observations = torch.randn(sample_count, 2, 4, generator=generator)
        current_observation = self.observations[:, -1]
        future_actions = []
        for horizon_index in range(pred_horizon):
            offset = horizon_index * 0.05
            horizon_actions = torch.stack(
                [
                    0.7 * current_observation[:, 0] - 0.2 * current_observation[:, 1] + offset,
                    -0.4 * current_observation[:, 2] + 0.5 * current_observation[:, 3] - offset,
                ],
                dim=1,
            )
            future_actions.append(horizon_actions)
        self.actions = torch.stack(future_actions, dim=1)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "obs": self.observations[index],
            "action": self.actions[index],
        }


def create_serializable_episode(sample_count: int, map_seed: int) -> ExpertEpisode:
    """Create a deterministic seed-specific episode for end-to-end training tests."""
    phase = (map_seed % 13) * 0.11
    sample_axis = np.linspace(-1.0, 1.0, sample_count, dtype=np.float32) + phase
    observations = np.column_stack(
        [
            sample_axis,
            sample_axis**2,
            np.sin(sample_axis + phase),
            np.cos(sample_axis - phase),
        ]
    ).astype(np.float32)
    actions = np.column_stack(
        [
            0.5 + 0.2 * sample_axis,
            0.3 * sample_axis - 0.1 * np.sin(sample_axis + phase),
        ]
    ).astype(np.float32)
    states = np.column_stack(
        [
            sample_axis,
            np.zeros_like(sample_axis),
            np.zeros_like(sample_axis),
        ]
    ).astype(np.float32)
    return ExpertEpisode(
        observations=observations,
        actions=actions,
        states=states,
        map_seed=map_seed,
        success=True,
        collision=False,
        truncated=False,
    )


def test_choose_device_resolves_cpu_and_rejects_unknown_name() -> None:
    """Device selection should always support CPU and reject misspelled backends."""
    # Act
    selected_device = choose_device("cpu")

    # Assert
    assert selected_device == torch.device("cpu")
    with pytest.raises(ValueError, match="device must be"):
        choose_device("quantum")


def test_bc_chunk_training_uses_every_action_in_the_horizon() -> None:
    """Training and validation should compare the complete predicted sequence with its labels."""
    # Arrange
    set_seed(29)
    model = BC_Chunk(obs_dim=4, obs_horizon=2, pred_horizon=4, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    dataloader = DataLoader(ChunkImitationDataset(), batch_size=12, shuffle=False)
    parameters_before_training = [parameter.detach().clone() for parameter in model.parameters()]

    # Act
    training_loss = train_bc_chunk_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device="cpu",
    )
    validation_loss = validate_bc_chunk(model, dataloader, device="cpu")

    # Assert
    parameters_after_training = list(model.parameters())
    assert np.isfinite(training_loss)
    assert np.isfinite(validation_loss)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before_training, parameters_after_training)
    )


def test_bc_chunk_can_overfit_small_action_sequence_dataset() -> None:
    """A compact deterministic chunk dataset should be memorized by the Gate7 model."""
    # Arrange
    set_seed(37)
    model = BC_Chunk(obs_dim=4, obs_horizon=2, pred_horizon=4, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    dataloader = DataLoader(
        ChunkImitationDataset(sample_count=32, pred_horizon=4),
        batch_size=32,
    )
    initial_loss = validate_bc_chunk(model, dataloader, device="cpu")

    # Act
    for _ in range(180):
        train_bc_chunk_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device="cpu",
        )
    final_loss = validate_bc_chunk(model, dataloader, device="cpu")

    # Assert
    assert final_loss < initial_loss * 0.01
    assert final_loss < 1e-3


def test_train_bc_chunk_runs_end_to_end_and_saves_chunk_checkpoint(
    tmp_path: Path,
) -> None:
    """Gate7 training should learn full action sequences and save reconstructable metadata."""
    # Arrange
    training_dataset_path = tmp_path / "chunk_training.npz"
    validation_dataset_path = tmp_path / "chunk_validation.npz"
    checkpoint_path = tmp_path / "checkpoints" / "bc_chunk.pt"
    save_expert_dataset(
        [create_serializable_episode(sample_count=24, map_seed=401)],
        training_dataset_path,
    )
    save_expert_dataset(
        [create_serializable_episode(sample_count=20, map_seed=502)],
        validation_dataset_path,
    )
    config = {
        "data": {
            "train_path": str(training_dataset_path),
            "validation_path": str(validation_dataset_path),
            "obs_horizon": 2,
            "pred_horizon": 4,
            "batch_size": 8,
            "num_workers": 0,
        },
        "model": {"hidden_dim": 32, "dropout": 0.1},
        "training": {
            "seed": 41,
            "epochs": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "device": "cpu",
        },
        "output": {"checkpoint_path": str(checkpoint_path)},
    }

    # Act
    model, normalizer, history = train_bc_chunk(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Assert
    assert isinstance(model, BC_Chunk)
    assert model.pred_horizon == 4
    assert isinstance(normalizer, Normalizer)
    assert len(history) == 2
    assert checkpoint["policy_type"] == "bc_chunk"
    assert checkpoint["model_config"]["pred_horizon"] == 4
    assert checkpoint["model_config"]["dropout"] == pytest.approx(0.1)
    assert checkpoint_path.is_file()
    assert (checkpoint_path.parent / "bc_chunk_loss.png").is_file()
    assert all(np.isfinite(epoch["train_loss"]) for epoch in history)
    assert all(np.isfinite(epoch["validation_loss"]) for epoch in history)
    assert all(np.isfinite(epoch["learning_rate"]) for epoch in history)


def test_train_bc_chunk_stops_early_and_restores_best_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patience should cap training and the returned model should use the best epoch weights."""
    # Arrange
    training_dataset_path = tmp_path / "early_stop_training.npz"
    validation_dataset_path = tmp_path / "early_stop_validation.npz"
    checkpoint_path = tmp_path / "early_stop_best.pt"
    save_expert_dataset(
        [create_serializable_episode(sample_count=12, map_seed=601)],
        training_dataset_path,
    )
    save_expert_dataset(
        [create_serializable_episode(sample_count=12, map_seed=602)],
        validation_dataset_path,
    )
    validation_losses = iter((1.0, 0.8, 0.81, 0.82, 0.83))
    epoch_number = 0

    def fake_train_epoch(model, dataloader, optimizer, device, observation_noise=0.0) -> float:
        nonlocal epoch_number
        epoch_number += 1
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float(epoch_number))
        return float(epoch_number)

    monkeypatch.setattr(training_module, "train_bc_chunk_epoch", fake_train_epoch)
    monkeypatch.setattr(
        training_module,
        "validate_bc_chunk",
        lambda model, dataloader, device: next(validation_losses),
    )
    config = {
        "data": {
            "train_path": str(training_dataset_path),
            "validation_path": str(validation_dataset_path),
            "obs_horizon": 2,
            "pred_horizon": 4,
            "batch_size": 4,
            "num_workers": 0,
        },
        "model": {"hidden_dim": 16, "dropout": 0.1},
        "training": {
            "seed": 0,
            "epochs": 20,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.0001,
            "lr_scheduler": {
                "factor": 0.5,
                "patience": 0,
                "threshold": 0.0,
                "min_lr": 0.000001,
            },
            "device": "cpu",
        },
        "output": {"checkpoint_path": str(checkpoint_path)},
    }

    # Act
    model, _, history = train_bc_chunk(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Assert
    assert len(history) == 4
    assert checkpoint["epoch"] == 2
    assert history[3]["learning_rate"] == pytest.approx(0.0005)
    assert all(
        torch.equal(value.cpu(), checkpoint["model_state_dict"][name].cpu())
        for name, value in model.state_dict().items()
    )


def test_bc_chunk_validation_report_uses_checkpoint_model_and_normalizer(tmp_path: Path) -> None:
    """A saved checkpoint should reproduce full-chunk MSE on the supplied validation data."""
    # Arrange
    validation_path = tmp_path / "validation.npz"
    checkpoint_path = tmp_path / "bc_chunk.pt"
    report_path = tmp_path / "validation_result.json"
    save_expert_dataset([create_serializable_episode(sample_count=5, map_seed=91)], validation_path)

    model = BC_Chunk(obs_dim=4, obs_horizon=2, pred_horizon=3, hidden_dim=8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    normalizer = Normalizer(
        observation_mean=np.zeros(4, dtype=np.float32),
        observation_std=np.ones(4, dtype=np.float32),
        action_mean=np.zeros(2, dtype=np.float32),
        action_std=np.ones(2, dtype=np.float32),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_bc_chunk_checkpoint(
        checkpoint_path, model, optimizer, normalizer, epoch=7,
        metrics={"train_loss": 0.1, "validation_loss": 0.25},
    )
    config = {
        "data": {
            "validation_path": str(validation_path),
            "obs_horizon": 8,
            "pred_horizon": 32,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {"type": "BC_Chunk"},
        "training": {"device": "cpu"},
        "output": {"checkpoint_path": str(checkpoint_path)},
    }
    dataset = SequenceDataset(validation_path, 2, 3, normalizer=normalizer)
    expected_loss = float(np.mean(np.stack([dataset[index]["action"] for index in range(5)]) ** 2))

    # Act
    result = validate_saved_bc_chunk(config)
    save_validation_result(result, report_path)

    # Assert
    assert result["validation_loss"] == pytest.approx(expected_loss)
    assert result["checkpoint_validation_loss"] == pytest.approx(0.25)
    assert result["checkpoint_epoch"] == 7
    assert result["sample_count"] == 5
    assert result["episode_count"] == 1
    assert result["map_seeds"] == [91]
    assert result["obs_horizon"] == 2
    assert result["pred_horizon"] == 3
    assert json.loads(report_path.read_text(encoding="utf-8")) == result
