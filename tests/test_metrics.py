import numpy as np

from peft_probe.metrics import binary_error_metrics, cluster_bootstrap_mean, risk_coverage


def test_perfect_ranking_metrics():
    correct = np.asarray([1, 1, 0, 0])
    error_probability = np.asarray([0.1, 0.2, 0.8, 0.9])
    metrics = binary_error_metrics(correct, error_probability)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    curve = risk_coverage(correct, error_probability)
    assert curve["selective_accuracy"][0] == 1.0


def test_cluster_bootstrap_uses_clusters_and_is_reproducible():
    clusters = {
        "entity_a": np.asarray([1.0, 1.0, 1.0]),
        "entity_b": np.asarray([0.0, 0.0, 0.0]),
    }
    first = cluster_bootstrap_mean(clusters, samples=200, seed=7)
    second = cluster_bootstrap_mean(clusters, samples=200, seed=7)
    assert first == second
    assert first["mean"] == 0.5
    assert first["num_clusters"] == 2
