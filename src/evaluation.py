from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_binary_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, float]:

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> tuple[
    float,
    dict[str, float],
    np.ndarray,
    list[int],
    list[int],
]:

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(images).squeeze(1)

            loss = criterion(
                logits,
                labels,
            )

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).long()

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            all_labels.extend(
                labels.cpu()
                .numpy()
                .astype(int)
                .tolist()
            )

            all_predictions.extend(
                predictions.cpu()
                .numpy()
                .astype(int)
                .tolist()
            )

    average_loss = (
        total_loss / total_samples
    )

    metrics = calculate_binary_metrics(
        all_labels,
        all_predictions,
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    return (
        average_loss,
        metrics,
        cm,
        all_labels,
        all_predictions,
    )