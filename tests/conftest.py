import pytest
import torch

from src.model import create_model


@pytest.fixture(scope="session")
def test_model_path(tmp_path_factory):
    """
    Create a deterministic model checkpoint for automated tests.

    The real trained production model is intentionally not required
    for unit/integration tests.
    """

    artifacts_dir = (
        tmp_path_factory.getbasetemp()
        / "test_artifacts"
    )

    artifacts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        artifacts_dir
        / "best_model.pt"
    )

    torch.manual_seed(42)

    model = create_model(
        dropout=0.3,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": "CatsDogsCNN",
        "dropout": 0.3,
        "image_size": 224,
        "class_mapping": {
            "cat": 0,
            "dog": 1,
        },
    }

    torch.save(
        checkpoint,
        model_path,
    )

    return model_path