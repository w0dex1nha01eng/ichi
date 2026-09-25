import pytest
import torch

from diffusion_nav.models import BC_Chunk

OBSERVATION_DIMENSION = 22
OBSERVATION_HORIZON = 2
BATCH_SIZE = 8


def test_bc_chunk_predicts_the_configured_action_sequence() -> None:
    """BC-Chunk should emit a separate linear-angular action for every future step."""
    # Arrange
    prediction_horizon = 5
    model = BC_Chunk(
        obs_dim=OBSERVATION_DIMENSION,
        obs_horizon=OBSERVATION_HORIZON,
        pred_horizon=prediction_horizon,
        hidden_dim=32,
    )
    observation_batch = torch.randn(
        BATCH_SIZE,
        OBSERVATION_HORIZON,
        OBSERVATION_DIMENSION,
        dtype=torch.float32,
    )

    # Act
    predicted_action_chunks = model(observation_batch)

    # Assert
    assert predicted_action_chunks.shape == (BATCH_SIZE, prediction_horizon, 2)
    assert predicted_action_chunks.dtype == torch.float32
    assert torch.isfinite(predicted_action_chunks).all()


def test_bc_chunk_backpropagates_loss_from_all_predicted_actions() -> None:
    """Supervision across the complete action horizon should reach model parameters."""
    # Arrange
    model = BC_Chunk(obs_dim=4, obs_horizon=2, pred_horizon=3, hidden_dim=24)
    observations = torch.randn(6, 2, 4)
    target_action_chunks = torch.randn(6, 3, 2)

    # Act
    loss = torch.nn.functional.mse_loss(model(observations), target_action_chunks)
    loss.backward()

    # Assert
    trainable_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert trainable_gradients
    assert all(gradient is not None for gradient in trainable_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)


def test_bc_chunk_adds_configured_dropout_regularization() -> None:
    """Configured dropout should regularize both hidden layers."""
    # Arrange and act
    model = BC_Chunk(obs_dim=4, obs_horizon=2, pred_horizon=3, hidden_dim=24, dropout=0.1)

    # Assert
    dropout_layers = [layer for layer in model.network if isinstance(layer, torch.nn.Dropout)]
    assert model.dropout == pytest.approx(0.1)
    assert len(dropout_layers) == 2
    assert all(layer.p == pytest.approx(0.1) for layer in dropout_layers)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, float("inf"), True])
def test_bc_chunk_rejects_invalid_dropout(dropout: float) -> None:
    """Dropout probabilities must be finite and stay in the half-open unit interval."""
    # Act and assert
    with pytest.raises(ValueError, match="dropout"):
        BC_Chunk(obs_dim=4, obs_horizon=2, pred_horizon=3, hidden_dim=24, dropout=dropout)


@pytest.mark.parametrize(
    ("obs_dim", "obs_horizon", "pred_horizon", "hidden_dim"),
    [
        pytest.param(4, 2, 0, 16, id="zero-prediction-horizon"),
        pytest.param(4, 2, -1, 16, id="negative-prediction-horizon"),
        pytest.param(4, 2, True, 16, id="boolean-prediction-horizon"),
    ],
)
def test_bc_chunk_rejects_invalid_prediction_horizon(
    obs_dim: int,
    obs_horizon: int,
    pred_horizon: int,
    hidden_dim: int,
) -> None:
    """Prediction horizon must be a positive integer and cannot be a boolean."""
    # Act and assert
    with pytest.raises(ValueError, match="pred_horizon must be a positive integer"):
        BC_Chunk(
            obs_dim=obs_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            hidden_dim=hidden_dim,
        )
