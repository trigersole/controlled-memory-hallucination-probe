import numpy as np

from peft_probe.metrics import binary_error_metrics, risk_coverage


def test_perfect_ranking_metrics():
    correct = np.asarray([1, 1, 0, 0])
    error_probability = np.asarray([0.1, 0.2, 0.8, 0.9])
    metrics = binary_error_metrics(correct, error_probability)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    curve = risk_coverage(correct, error_probability)
    assert curve["selective_accuracy"][0] == 1.0

