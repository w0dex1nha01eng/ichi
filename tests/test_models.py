import pytest
import torch

from diffusion_nav.models import BC_1

OBSERVATION_DIMENSION = 22
OBSERVATION_HORIZON = 2
BATCH_SIZE = 8


@pytest.fixture
def behavior_cloning_model() -> BC_1:
    """Return a small BC-1 model suitable for fast unit tests."""
    return BC_1(
        obs_dim=OBSERVATION_DIMENSION,
        obs_horizon=OBSERVATION_HORIZON,
        hidden_dim=32,
    )


def test_bc_one_step_maps_observation_history_to_single_action(
    behavior_cloning_model: BC_1,
) -> None:
    """One observation history should produce exactly one linear-angular action pair."""
    # Arrange
    observations = torch.randn(
        BATCH_SIZE,
        OBSERVATION_HORIZON,
        OBSERVATION_DIMENSION,
        dtype=torch.float32,
    )

    # Act
    predicted_actions = behavior_cloning_model(observations)

    # Assert
    assert predicted_actions.shape == (BATCH_SIZE, 2)
    assert predicted_actions.dtype == torch.float32
    assert torch.isfinite(predicted_actions).all()


def test_bc_one_step_supports_gradient_based_optimization(
    behavior_cloning_model: BC_1,
) -> None:
    """The supervised MSE objective should backpropagate through every linear layer."""
    # Arrange
    observations = torch.randn(
        BATCH_SIZE,
        OBSERVATION_HORIZON,
        OBSERVATION_DIMENSION,
        dtype=torch.float32,
    )
    target_actions = torch.randn(BATCH_SIZE, 2, dtype=torch.float32)

    # Act
    loss = torch.nn.functional.mse_loss(
        behavior_cloning_model(observations),
        target_actions,
    )
    loss.backward()

    # Assert
    parameter_gradients = [
        parameter.grad
        for parameter in behavior_cloning_model.parameters()
        if parameter.requires_grad
    ]
    assert parameter_gradients
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)


@pytest.mark.parametrize(
    ("obs_dim", "obs_horizon", "hidden_dim"),
    [
        pytest.param(0, 2, 32, id="zero-observation-dimension"),
        pytest.param(22, 0, 32, id="zero-observation-horizon"),
        pytest.param(22, 2, -1, id="negative-hidden-dimension"),
        pytest.param(22, True, 32, id="boolean-observation-horizon"),
    ],
)
def test_bc_one_step_rejects_invalid_dimensions(
    obs_dim: int,
    obs_horizon: int,
    hidden_dim: int,
) -> None:
    """All network dimensions must be genuine positive integers."""
    # Act and assert
    with pytest.raises(ValueError, match="must be a positive integer"):
        BC_1(
            obs_dim=obs_dim,
            obs_horizon=obs_horizon,
            hidden_dim=hidden_dim,
        )


@pytest.mark.parametrize(
    "invalid_observations",
    [
        pytest.param(torch.zeros(2, 22), id="missing-history-axis"),
        pytest.param(torch.zeros(2, 3, 22), id="wrong-history-length"),
        pytest.param(torch.zeros(2, 2, 21), id="wrong-observation-dimension"),
    ],
)
def test_bc_one_step_rejects_incompatible_input_shape(
    behavior_cloning_model: BC_1,
    invalid_observations: torch.Tensor,
) -> None:
    """Malformed batches should fail before they reach the neural network layers."""
    # Act and assert
    with pytest.raises(ValueError):
        behavior_cloning_model(invalid_observations)
