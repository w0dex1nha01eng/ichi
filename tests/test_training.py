from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_nav.dataset import (
    ExpertEpisode,
    Normalizer,
    load_expert_dataset,
    save_expert_dataset,
)
from diffusion_nav.models import BC_1
from diffusion_nav.training import (
    choose_device,
    load_bc_checkpoint,
    save_bc_checkpoint,
    set_seed,
    train_bc,
    train_bc_epoch,
    validate_bc,
)


class LinearImitationDataset(Dataset):
    """Synthetic dataset with a deterministic linear expert policy."""

    def __init__(self, sample_count: int = 64) -> None:
        generator = torch.Generator().manual_seed(23)
        self.observations = torch.randn(sample_count, 2, 4, generator=generator)
        current_observation = self.observations[:, -1]
        expert_actions = torch.stack(
            [
                0.7 * current_observation[:, 0] - 0.2 * current_observation[:, 1],
                -0.4 * current_observation[:, 2] + 0.5 * current_observation[:, 3],
            ],
            dim=1,
        )
        self.actions = expert_actions.unsqueeze(1)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "obs": self.observations[index],
            "action": self.actions[index],
        }


@pytest.fixture
def identity_normalizer() -> Normalizer:
    """Return identity statistics for a four-dimensional synthetic observation."""
    return Normalizer(
        observation_mean=np.zeros(4, dtype=np.float32),
        observation_std=np.ones(4, dtype=np.float32),
        action_mean=np.zeros(2, dtype=np.float32),
        action_std=np.ones(2, dtype=np.float32),
    )


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


def test_one_training_epoch_updates_model_parameters() -> None:
    """A supervised epoch should move at least one learned parameter."""
    # Arrange
    set_seed(5)
    model = BC_1(obs_dim=4, obs_horizon=2, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    dataloader = DataLoader(LinearImitationDataset(), batch_size=16, shuffle=False)
    parameters_before_training = [parameter.detach().clone() for parameter in model.parameters()]

    # Act
    training_loss = train_bc_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device="cpu",
    )

    # Assert
    parameters_after_training = list(model.parameters())
    assert np.isfinite(training_loss)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before_training, parameters_after_training)
    )


def test_model_can_overfit_small_deterministic_dataset() -> None:
    """Gate6 acceptance requires BC-1 to visibly learn a tiny expert mapping."""
    # Arrange
    set_seed(7)
    model = BC_1(obs_dim=4, obs_horizon=2, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    dataloader = DataLoader(LinearImitationDataset(sample_count=32), batch_size=32)
    initial_loss = validate_bc(model, dataloader, device="cpu")

    # Act
    for _ in range(150):
        train_bc_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device="cpu",
        )
    final_loss = validate_bc(model, dataloader, device="cpu")

    # Assert
    assert final_loss < initial_loss * 0.01
    assert final_loss < 1e-3


def test_checkpoint_round_trip_restores_predictions_and_metadata(
    tmp_path: Path,
    identity_normalizer: Normalizer,
) -> None:
    """A saved training checkpoint should fully reconstruct model behavior."""
    # Arrange
    set_seed(11)
    source_model = BC_1(obs_dim=4, obs_horizon=2, hidden_dim=16)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "bc_one_step.pt"
    observations = torch.randn(3, 2, 4)
    expected_predictions = source_model(observations).detach()

    # Act
    save_bc_checkpoint(
        checkpoint_path=checkpoint_path,
        model=source_model,
        optimizer=source_optimizer,
        normalizer=identity_normalizer,
        epoch=4,
        metrics={"train_loss": 0.25, "validation_loss": 0.30},
    )
    restored_model = BC_1(obs_dim=4, obs_horizon=2, hidden_dim=16)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    checkpoint = load_bc_checkpoint(
        checkpoint_path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        device="cpu",
    )
    restored_predictions = restored_model(observations).detach()

    # Assert
    torch.testing.assert_close(restored_predictions, expected_predictions)
    assert checkpoint["epoch"] == 4
    assert checkpoint["metrics"]["validation_loss"] == pytest.approx(0.30)


def test_train_bc_runs_end_to_end_and_saves_best_checkpoint(tmp_path: Path) -> None:
    """The public training entry point should consume Gate5 data and emit a BC-1 checkpoint."""
    # Arrange
    training_dataset_path = tmp_path / "training.npz"
    validation_dataset_path = tmp_path / "validation.npz"
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    save_expert_dataset(
        [create_serializable_episode(sample_count=48, map_seed=101)],
        training_dataset_path,
    )
    save_expert_dataset(
        [create_serializable_episode(sample_count=32, map_seed=202)],
        validation_dataset_path,
    )
    training_data = load_expert_dataset(training_dataset_path)
    validation_data = load_expert_dataset(validation_dataset_path)
    training_observations = {tuple(row) for row in training_data["observations"]}
    validation_observations = {tuple(row) for row in validation_data["observations"]}
    assert training_data["map_seeds"].tolist() == [101]
    assert validation_data["map_seeds"].tolist() == [202]
    assert training_observations.isdisjoint(validation_observations)
    config = {
        "data": {
            "train_path": str(training_dataset_path),
            "validation_path": str(validation_dataset_path),
            "obs_horizon": 2,
            "pred_horizon": 1,
            "batch_size": 16,
            "num_workers": 0,
        },
        "model": {"hidden_dim": 32},
        "training": {
            "seed": 13,
            "epochs": 3,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "device": "cpu",
        },
        "output": {"checkpoint_path": str(checkpoint_path)},
    }

    # Act
    model, normalizer, history = train_bc(config)

    # Assert
    assert isinstance(model, BC_1)
    assert isinstance(normalizer, Normalizer)
    assert len(history) == 3
    assert checkpoint_path.is_file()
    assert all(np.isfinite(epoch["train_loss"]) for epoch in history)
    assert all(np.isfinite(epoch["validation_loss"]) for epoch in history)


def test_train_bc_rejects_shared_map_seeds(tmp_path: Path) -> None:
    """Training and validation must not reuse episodes from the same map seed."""
    # Arrange
    shared_dataset_path = tmp_path / "shared_map.npz"
    save_expert_dataset(
        [create_serializable_episode(sample_count=16, map_seed=303)],
        shared_dataset_path,
    )
    config = {
        "data": {
            "train_path": str(shared_dataset_path),
            "validation_path": str(shared_dataset_path),
            "obs_horizon": 2,
            "pred_horizon": 1,
            "batch_size": 8,
            "num_workers": 0,
        },
        "model": {"hidden_dim": 16},
        "training": {
            "seed": 17,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "device": "cpu",
        },
        "output": {"checkpoint_path": str(tmp_path / "best.pt")},
    }

    # Act and assert
    with pytest.raises(ValueError, match=r"share map seeds: \[303\]"):
        train_bc(config)
