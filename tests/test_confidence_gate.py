import numpy as np

from peft_probe.confidence_gate import _fit_threshold, _paired_gaps, _threshold_metrics


def test_fit_threshold_separates_correct_and_wrong_validation_answers():
    correct = np.asarray([1, 1, 0, 0])
    risk = np.asarray([0.05, 0.10, 0.80, 0.90])
    threshold, metrics = _fit_threshold(correct, risk)
    assert 0.10 < threshold < 0.80
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["error_recall"] == 1.0
    assert metrics["correct_acceptance"] == 1.0


def test_threshold_metrics_report_selective_accuracy_and_coverage():
    metrics = _threshold_metrics(
        np.asarray([1, 1, 0, 0]), np.asarray([0.1, 0.2, 0.4, 0.9]), 0.3
    )
    assert metrics["coverage"] == 0.5
    assert metrics["selective_accuracy"] == 1.0
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2


def test_paired_gap_matches_exposed_and_withheld_by_fact():
    records = []
    for fact, relation in (("f1", "birthplace"), ("f2", "occupation")):
        records.extend(
            [
                {
                    "fact_id": fact,
                    "entity_id": fact,
                    "relation": relation,
                    "exposure": 1,
                    "mean_logprob": -0.01,
                    "mean_entropy": 0.02,
                    "correct": 1,
                },
                {
                    "fact_id": fact,
                    "entity_id": fact,
                    "relation": relation,
                    "exposure": 0,
                    "mean_logprob": -0.21,
                    "mean_entropy": 0.42,
                    "correct": 0,
                },
            ]
        )
    report = _paired_gaps(records, samples=100, seed=7)
    assert report["all"]["n"] == 2
    assert abs(report["all"]["logprob_gap"]["mean"] - 0.2) < 1e-9
    assert abs(report["all"]["entropy_gap"]["mean"] - 0.4) < 1e-9
    assert report["all"]["logprob_gap"]["fraction_positive"] == 1.0
