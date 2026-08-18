import torch

from src.model import create_model


def test_model_output_shape() -> None:
    model = create_model()

    x = torch.randn(
        4,
        3,
        224,
        224,
    )

    output = model(x)

    assert output.shape == (4, 1)


def test_model_output_is_finite() -> None:
    model = create_model()

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    output = model(x)

    assert torch.isfinite(output).all()