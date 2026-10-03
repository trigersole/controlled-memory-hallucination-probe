from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        include = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if not include.any():
            continue
        score += include.mean() * abs(labels[include].mean() - probabilities[include].mean())
    return float(score)


def risk_coverage(labels_correct: np.ndarray, probability_error: np.ndarray) -> dict[str, np.ndarray | float]:
    order = np.argsort(probability_error)
    accepted_correct = labels_correct[order]
    coverage = np.arange(1, len(order) + 1, dtype=np.float64) / max(len(order), 1)
    selective_accuracy = np.cumsum(accepted_correct) / np.arange(1, len(order) + 1)
    selective_risk = 1.0 - selective_accuracy
    aurc = float(np.trapz(selective_risk, coverage)) if len(order) > 1 else float(selective_risk[0])
    return {
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "selective_risk": selective_risk,
        "aurc": aurc,
        "order": order,
    }


def binary_error_metrics(labels_correct: np.ndarray, probability_error: np.ndarray) -> dict[str, float]:
    labels_correct = np.asarray(labels_correct, dtype=np.int64)
    probability_error = np.asarray(probability_error, dtype=np.float64)
    labels_error = 1 - labels_correct
    metrics: dict[str, float] = {
        "n": int(len(labels_error)),
        "error_rate": float(labels_error.mean()) if len(labels_error) else float("nan"),
        "brier": float(brier_score_loss(labels_error, probability_error)),
        "ece": expected_calibration_error(labels_error, probability_error),
        "aurc": float(risk_coverage(labels_correct, probability_error)["aurc"]),
    }
    if np.unique(labels_error).size == 2:
        metrics["auroc"] = float(roc_auc_score(labels_error, probability_error))
        metrics["auprc"] = float(average_precision_score(labels_error, probability_error))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics


def bootstrap_metric_differences(
    labels_correct: np.ndarray,
    proposed: np.ndarray,
    baseline: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    names = ("auroc", "auprc", "brier", "ece", "aurc")
    draws: dict[str, list[float]] = {name: [] for name in names}
    n = len(labels_correct)
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        proposal_metrics = binary_error_metrics(labels_correct[indices], proposed[indices])
        baseline_metrics = binary_error_metrics(labels_correct[indices], baseline[indices])
        for name in names:
            difference = proposal_metrics[name] - baseline_metrics[name]
            if np.isfinite(difference):
                draws[name].append(float(difference))
    return {
        name: {
            "mean_difference": float(np.mean(values)) if values else float("nan"),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            if values else [float("nan"), float("nan")],
        }
        for name, values in draws.items()
    }

