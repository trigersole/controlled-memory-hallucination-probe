import numpy as np

from peft_probe.cross_model_rank_gate import (
    _cluster_bootstrap_coverages,
    _cluster_bootstrap_methods,
    _correct_probability,
    _coverage_metrics,
    _rank_percentiles,
)


def test_rank_percentiles_preserve_risk_order():
    risk = np.asarray([0.8, 0.1, 0.5])
    percentiles = _rank_percentiles(risk)
    assert percentiles[1] < percentiles[2] < percentiles[0]


def test_fixed_coverage_accepts_lowest_risk_answers():
    labels = np.asarray([1, 1, 0, 0])
    risk = np.asarray([0.1, 0.2, 0.8, 0.9])
    metrics = _coverage_metrics(labels, risk, coverage=0.5)
    assert metrics["coverage"] == 0.5
    assert metrics["selective_accuracy"] == 1.0
    assert metrics["error_recall"] == 1.0
    assert metrics["correct_acceptance"] == 1.0


def test_entity_bootstrap_is_reproducible():
    rows = [
        {"entity_id": "one"},
        {"entity_id": "one"},
        {"entity_id": "two"},
        {"entity_id": "two"},
    ]
    labels = np.asarray([1, 1, 0, 0])
    risk = np.asarray([0.1, 0.2, 0.8, 0.9])
    first = _cluster_bootstrap_coverages(rows, labels, risk, [0.5], 100, 7)
    second = _cluster_bootstrap_coverages(rows, labels, risk, [0.5], 100, 7)
    assert first == second


def test_saved_logistic_model_is_applied_without_refitting():
    report = {
        "fitted_model": {
            "feature_names": ["base_support"],
            "mean": [0.0],
            "std": [1.0],
            "coefficients": [1.0],
            "intercept": 0.0,
        }
    }
    probability = _correct_probability(
        report, ["base_support"], np.asarray([[-1.0], [1.0]])
    )
    assert probability[0] < 0.5 < probability[1]


def test_paired_bootstrap_detects_better_selective_ranking():
    rows = [{"entity_id": str(index)} for index in range(20)]
    labels = np.asarray([1] * 10 + [0] * 10)
    proposed = np.asarray([0.1] * 10 + [0.9] * 10)
    baseline = np.asarray([0.9, 0.1] * 10)
    _, differences = _cluster_bootstrap_methods(
        rows,
        labels,
        {"intervention_support": proposed, "baseline": baseline},
        [0.5],
        100,
        7,
        "intervention_support",
    )
    assert differences["baseline"]["0.5"][0] > 0


def test_paired_bootstrap_supports_explicit_method_comparison():
    rows = [{"entity_id": str(index)} for index in range(20)]
    labels = np.asarray([1] * 10 + [0] * 10)
    maximum = np.asarray([0.1] * 10 + [0.9] * 10)
    base = np.asarray([0.9, 0.1] * 10)
    _, differences = _cluster_bootstrap_methods(
        rows,
        labels,
        {"max_contextual_support": maximum, "base_contextual_support": base},
        [0.5],
        100,
        7,
        "max_contextual_support",
        [("max_contextual_support", "base_contextual_support")],
    )
    comparison = differences["max_contextual_support_minus_base_contextual_support"]
    assert comparison["0.5"][0] > 0
