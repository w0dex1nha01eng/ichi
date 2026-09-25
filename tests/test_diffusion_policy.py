from pathlib import Path

import numpy as np
import pytest
import torch

from diffusion_nav.dataset import ExpertEpisode, Normalizer, save_expert_dataset
from diffusion_nav.diffusion_policy import (
    ConditionalDiffusionModel,
    DiffusionPolicy,
    DiffusionSchedule,
    load_diffusion_policy,
)
from diffusion_nav.diffusion_training import (
    save_diffusion_checkpoint,
    train_diffusion_policy,
)


def create_episode(sample_count: int, map_seed: int) -> ExpertEpisode:
    time = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    observations = np.stack(
        (time, np.sin(time * np.pi), np.cos(time * np.pi), np.full_like(time, map_seed % 7)),
        axis=1,
    ).astype(np.float32)
    actions = np.stack((0.5 + 0.2 * time, np.sin(time * np.pi * 2.0)), axis=1).astype(np.float32)
    states = np.stack((time, time * 0.5, time * 0.1), axis=1).astype(np.float32)
    return ExpertEpisode(
        observations=observations,
        actions=actions,
        states=states,
        map_seed=map_seed,
        success=True,
        collision=False,
        truncated=False,
    )


def create_small_model() -> ConditionalDiffusionModel:
    return ConditionalDiffusionModel(
        obs_dim=4,
        obs_horizon=2,
        pred_horizon=3,
        hidden_dim=16,
        time_embedding_dim=8,
        diffusion_steps=4,
        dropout=0.0,
    )


def test_cosine_schedule_progressively_removes_signal() -> None:
    schedule = DiffusionSchedule(diffusion_steps=16, schedule_type="cosine")

    assert schedule.betas.shape == (16,)
    assert torch.all(schedule.betas > 0.0)
    assert torch.all(schedule.betas < 1.0)
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])
    assert schedule.alpha_bars[-1] < 1e-3


def test_diffusion_loss_backpropagates_through_noise_predictor() -> None:
    model = create_small_model()
    observations = torch.randn(5, 2, 4)
    actions = torch.randn(5, 3, 2)

    loss = model.diffusion_loss(
        observations,
        actions,
        generator=torch.Generator().manual_seed(11),
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_reverse_diffusion_is_finite_and_reproducible_with_a_seed() -> None:
    model = create_small_model().eval()
    observations = torch.randn(2, 2, 4)

    first = model.sample(observations, generator=torch.Generator().manual_seed(17))
    second = model.sample(observations, generator=torch.Generator().manual_seed(17))

    assert first.shape == (2, 3, 2)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_diffusion_policy_reset_replays_the_same_sampling_sequence() -> None:
    model = create_small_model()
    normalizer = Normalizer(
        observation_mean=np.zeros(4, dtype=np.float32),
        observation_std=np.ones(4, dtype=np.float32),
        action_mean=np.zeros(2, dtype=np.float32),
        action_std=np.ones(2, dtype=np.float32),
    )
    policy = DiffusionPolicy(
        model,
        normalizer,
        action_low=np.array([0.0, -2.0], dtype=np.float32),
        action_high=np.array([1.0, 2.0], dtype=np.float32),
        seed=23,
    )
    observation_history = np.zeros((2, 4), dtype=np.float32)

    first = policy.act(observation_history)
    policy.reset()
    second = policy.act(observation_history)

    assert first.shape == (3, 2)
    assert np.all(np.isfinite(first))
    assert np.all(first >= np.array([0.0, -2.0], dtype=np.float32))
    assert np.all(first <= np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(first, second)


def test_diffusion_checkpoint_round_trip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "diffusion.pt"
    model = create_small_model()
    normalizer = Normalizer(
        observation_mean=np.zeros(4, dtype=np.float32),
        observation_std=np.ones(4, dtype=np.float32),
        action_mean=np.zeros(2, dtype=np.float32),
        action_std=np.ones(2, dtype=np.float32),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_diffusion_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        normalizer,
        epoch=3,
        metrics={"train_loss": 0.4, "validation_loss": 0.5},
    )

    loaded = load_diffusion_policy(
        checkpoint_path,
        action_low=np.array([0.0, -2.0], dtype=np.float32),
        action_high=np.array([1.0, 2.0], dtype=np.float32),
        device="cpu",
        seed=5,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["policy_type"] == "diffusion_policy"
    assert checkpoint["epoch"] == 3
    assert loaded.model.config_dict() == model.config_dict()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[name])


def test_diffusion_training_entry_point_saves_best_checkpoint(tmp_path: Path) -> None:
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    checkpoint_path = tmp_path / "diffusion.pt"
    curve_path = tmp_path / "diffusion_loss.png"
    save_expert_dataset([create_episode(12, 701)], train_path)
    save_expert_dataset([create_episode(10, 801)], validation_path)
    config = {
        "data": {
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "obs_horizon": 2,
            "pred_horizon": 3,
            "batch_size": 4,
            "num_workers": 0,
        },
        "model": {
            "type": "DiffusionPolicy",
            "hidden_dim": 16,
            "time_embedding_dim": 8,
            "diffusion_steps": 4,
            "schedule_type": "cosine",
            "dropout": 0.0,
        },
        "training": {
            "seed": 3,
            "validation_seed": 4,
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.0,
            "device": "cpu",
        },
        "output": {
            "checkpoint_path": str(checkpoint_path),
            "loss_curve_path": str(curve_path),
        },
    }

    model, normalizer, history = train_diffusion_policy(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert isinstance(model, ConditionalDiffusionModel)
    assert isinstance(normalizer, Normalizer)
    assert len(history) == 2
    assert checkpoint["policy_type"] == "diffusion_policy"
    assert checkpoint["model_config"]["diffusion_steps"] == 4
    assert checkpoint_path.is_file()
    assert curve_path.is_file()
    assert all(np.isfinite(item["train_loss"]) for item in history)
    assert all(np.isfinite(item["validation_loss"]) for item in history)
    assert all(np.isfinite(item["learning_rate"]) for item in history)


@pytest.mark.parametrize("schedule_type", ["quadratic", "unknown"])
def test_schedule_rejects_unknown_types(schedule_type: str) -> None:
    with pytest.raises(ValueError, match="schedule_type"):
        DiffusionSchedule(diffusion_steps=4, schedule_type=schedule_type)
