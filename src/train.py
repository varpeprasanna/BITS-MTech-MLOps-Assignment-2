from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch as mlflow_pytorch
from mlflow.models import infer_signature
import numpy as np
import torch
import yaml
from torch import nn, optim

from src.data_loader import create_dataloaders
from src.evaluation import evaluate_model
from src.model import create_model
from src.utils import set_seed


MLFLOW_EXPERIMENT_NAME = "cats-dogs-baseline-cnn"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(
        PROJECT_ROOT / "params.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def train_one_epoch(
    model: nn.Module,
    data_loader,
    criterion,
    optimizer,
    device,
) -> tuple[float, float]:

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images).squeeze(1)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).long()

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        correct += (
            (predictions == labels.long())
            .sum()
            .item()
        )

        total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def save_training_curves(
    history: dict,
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    # ---------------------------------------------------------
    # Loss Curve
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        epochs,
        history["validation_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------
    # Accuracy Curve
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy",
    )

    plt.plot(
        epochs,
        history["validation_accuracy"],
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "accuracy_curve.png",
        dpi=150,
    )

    plt.close()


def save_confusion_matrix(
    confusion_matrix: np.ndarray,
    output_path: Path,
) -> None:

    plt.figure(figsize=(6, 5))

    plt.imshow(
        confusion_matrix,
        interpolation="nearest",
    )

    plt.title("Test Confusion Matrix")
    plt.colorbar()

    classes = ["Cat", "Dog"]

    tick_marks = np.arange(
        len(classes)
    )

    plt.xticks(
        tick_marks,
        classes,
    )

    plt.yticks(
        tick_marks,
        classes,
    )

    threshold = (
        confusion_matrix.max() / 2
    )

    for i in range(
        confusion_matrix.shape[0]
    ):
        for j in range(
            confusion_matrix.shape[1]
        ):
            plt.text(
                j,
                i,
                str(confusion_matrix[i, j]),
                horizontalalignment="center",
                color=(
                    "white"
                    if confusion_matrix[i, j]
                    > threshold
                    else "black"
                ),
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def main() -> None:

    # =========================================================
    # Load Configuration
    # =========================================================
    config = load_config()

    seed = config["seed"]

    set_seed(seed)

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    dropout = config["model"]["dropout"]

    num_workers = config["data"]["num_workers"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =========================================================
    # MLflow Experiment Setup
    # =========================================================
    #
    # MLflow will use the local tracking backend by default.
    # The mlruns directory will be created in the project root
    # when the experiment is executed from the project directory.
    #
    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    # =========================================================
    # Console Information
    # =========================================================
    print("=" * 70)
    print("CATS VS DOGS BASELINE CNN TRAINING")
    print("=" * 70)

    print(f"Device       : {device}")

    if torch.cuda.is_available():

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA         : "
            f"{torch.version.cuda}"
        )

    print(f"Batch size   : {batch_size}")
    print(f"Epochs       : {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay : {weight_decay}")
    print(f"Dropout      : {dropout}")
    print(f"Seed         : {seed}")

    # =========================================================
    # Paths
    # =========================================================
    processed_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    artifacts_dir = (
        PROJECT_ROOT
        / "artifacts"
    )

    artifacts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # Data Loaders
    # =========================================================
    train_loader, validation_loader, test_loader = (
        create_dataloaders(
            processed_dir=processed_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )

    # =========================================================
    # Model
    # =========================================================
    model = create_model(
        dropout=dropout
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # =========================================================
    # Training History
    # =========================================================
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    best_validation_loss = float("inf")

    best_model_path = (
        artifacts_dir
        / "best_model.pt"
    )

    start_time = time.time()

    # =========================================================
    # MLflow Run
    # =========================================================
    with mlflow.start_run(
        run_name="baseline-cnn-10-epochs"
    ):

        # -----------------------------------------------------
        # Log Training Configuration
        # -----------------------------------------------------
        mlflow.log_params(
            {
                "seed": seed,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "image_size": 224,
                "num_workers": num_workers,
                "optimizer": "Adam",
                "loss_function": "BCEWithLogitsLoss",
                "augmentation": (
                    "RandomHorizontalFlip, "
                    "RandomRotation, "
                    "ColorJitter"
                ),
            }
        )

        # -----------------------------------------------------
        # Training Loop
        # -----------------------------------------------------
        for epoch in range(1, epochs + 1):

            epoch_start = time.time()

            train_loss, train_accuracy = (
                train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                )
            )

            (
                validation_loss,
                validation_metrics,
                _,
                _,
                _,
            ) = evaluate_model(
                model,
                validation_loader,
                device,
                criterion,
            )

            # -------------------------------------------------
            # Store History
            # -------------------------------------------------
            history["train_loss"].append(
                train_loss
            )

            history["train_accuracy"].append(
                train_accuracy
            )

            history["validation_loss"].append(
                validation_loss
            )

            history["validation_accuracy"].append(
                validation_metrics["accuracy"]
            )

            # -------------------------------------------------
            # Log Epoch Metrics to MLflow
            # -------------------------------------------------
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": (
                        validation_metrics["accuracy"]
                    ),
                },
                step=epoch,
            )

            elapsed = (
                time.time()
                - epoch_start
            )

            print(
                f"\nEpoch "
                f"{epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {validation_loss:.4f} | "
                f"Val Acc: "
                f"{validation_metrics['accuracy']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # -------------------------------------------------
            # Save Best Model Checkpoint
            # -------------------------------------------------
            if validation_loss < best_validation_loss:

                best_validation_loss = (
                    validation_loss
                )

                torch.save(
                    {
                        "model_state_dict": (
                            model.state_dict()
                        ),
                        "model_name": "CatsDogsCNN",
                        "dropout": dropout,
                        "image_size": 224,
                        "class_mapping": {
                            "cat": 0,
                            "dog": 1,
                        },
                        "normalization": {
                            "mean": [
                                0.485,
                                0.456,
                                0.406,
                            ],
                            "std": [
                                0.229,
                                0.224,
                                0.225,
                            ],
                        },
                        "seed": seed,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "epochs": epochs,
                        "best_validation_loss": (
                            best_validation_loss
                        ),
                    },
                    best_model_path,
                )

                print(
                    "  → Saved new best model"
                )

        # =====================================================
        # Final Training Time
        # =====================================================
        total_training_time = (
            time.time() - start_time
        )

        # =====================================================
        # Load Best Checkpoint
        # =====================================================
        checkpoint = torch.load(
            best_model_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # =====================================================
        # Final Test Evaluation
        # =====================================================
        (
            test_loss,
            test_metrics,
            confusion,
            test_labels,
            test_predictions,
        ) = evaluate_model(
            model,
            test_loader,
            device,
            criterion,
        )

        # =====================================================
        # Console Test Results
        # =====================================================
        print("\n" + "=" * 70)
        print("FINAL TEST RESULTS")
        print("=" * 70)

        print(
            f"Test Loss     : {test_loss:.4f}"
        )

        print(
            f"Test Accuracy : "
            f"{test_metrics['accuracy']:.4f}"
        )

        print(
            f"Test Precision: "
            f"{test_metrics['precision']:.4f}"
        )

        print(
            f"Test Recall   : "
            f"{test_metrics['recall']:.4f}"
        )

        print(
            f"Test F1       : "
            f"{test_metrics['f1']:.4f}"
        )

        print(
            "\nConfusion Matrix:"
        )

        print(confusion)

        print(
            f"\nTraining time: "
            f"{total_training_time / 60:.2f} minutes"
        )

        # =====================================================
        # Log Final Test Metrics to MLflow
        # =====================================================
        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_accuracy": test_metrics[
                    "accuracy"
                ],
                "test_precision": test_metrics[
                    "precision"
                ],
                "test_recall": test_metrics[
                    "recall"
                ],
                "test_f1": test_metrics[
                    "f1"
                ],
            }
        )

        # =====================================================
        # Generate Training Artifacts
        # =====================================================
        save_training_curves(
            history,
            artifacts_dir,
        )

        save_confusion_matrix(
            confusion,
            artifacts_dir
            / "confusion_matrix.png",
        )

        # =====================================================
        # Save Training History
        # =====================================================
        with open(
            artifacts_dir / "training_history.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                history,
                file,
                indent=4,
            )

        # =====================================================
        # Save Test Metrics
        # =====================================================
        with open(
            artifacts_dir / "test_metrics.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                test_metrics,
                file,
                indent=4,
            )

        # =====================================================
        # Log Artifacts to MLflow
        # =====================================================
        mlflow.log_artifact(
            str(
                artifacts_dir
                / "loss_curve.png"
            )
        )

        mlflow.log_artifact(
            str(
                artifacts_dir
                / "accuracy_curve.png"
            )
        )

        mlflow.log_artifact(
            str(
                artifacts_dir
                / "confusion_matrix.png"
            )
        )

        mlflow.log_artifact(
            str(
                artifacts_dir
                / "training_history.json"
            )
        )

        mlflow.log_artifact(
            str(
                artifacts_dir
                / "test_metrics.json"
            )
        )

        # =====================================================
        # Log Best Model to MLflow
        # =====================================================

        # MLflow PyTorch model export uses torch.export internally.
        # Keep both the model and input example on CPU during export
        # to avoid CPU/CUDA FakeTensor device conflicts.

        mlflow_model = create_model(
            dropout=dropout
        )

        mlflow_model.load_state_dict(
            model.state_dict()
        )

        mlflow_model = mlflow_model.cpu()
        mlflow_model.eval()

        # Get one real test example
        example_images, _ = next(iter(test_loader))

        input_example = example_images[:1].cpu()

        # Generate output using the CPU model
        with torch.no_grad():
            example_output = mlflow_model(
                input_example
            )

        # Infer MLflow signature from CPU tensors
        signature = infer_signature(
            input_example.numpy(),
            example_output.numpy(),
        )

        # Log CPU model to MLflow
        mlflow_pytorch.log_model(
            mlflow_model,
            name="model",
            input_example=input_example,
            signature=signature,
        )

        # =====================================================
        # Final Output
        # =====================================================
        print(
            "\nArtifacts saved to:"
        )

        print(
            f"  {best_model_path}"
        )

        print(
            f"  {artifacts_dir / 'loss_curve.png'}"
        )

        print(
            f"  {artifacts_dir / 'accuracy_curve.png'}"
        )

        print(
            f"  {artifacts_dir / 'confusion_matrix.png'}"
        )

        print(
            f"  {artifacts_dir / 'training_history.json'}"
        )

        print(
            f"  {artifacts_dir / 'test_metrics.json'}"
        )

        print("\nMLflow experiment:")
        print(
            f"  {MLFLOW_EXPERIMENT_NAME}"
        )

        print(
            "\n" + "=" * 70
        )


if __name__ == "__main__":
    main()