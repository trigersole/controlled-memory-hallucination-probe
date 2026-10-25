import numpy as np

from peft_probe.cross_model_support import (
    _assign_control_questions,
    _candidate_source_identity,
    _feature_table,
    _safe_auroc,
)


def test_control_questions_are_relation_matched_and_distinct():
    rows = [
        {
            "candidate_id": f"row-{index}",
            "fact_id": f"fact-{index}",
            "dataset": "synthetic",
            "split": "train",
            "relation": "birthplace",
            "answer": "candidate",
            "expected": "unused reference",
            "question": f"Where is entity {value}?",
        }
        for index, value in enumerate(("Paris", "Rome", "Lima"))
    ]
    assigned = _assign_control_questions(rows, repeats=2, seed=7)
    available = {row["question"] for row in rows}
    for row in assigned:
        assert len(row["control_questions"]) == 2
        assert set(row["control_questions"]) <= available
        assert row["question"] not in row["control_questions"]


def test_duplicate_candidate_sources_do_not_duplicate_control_questions():
    rows = [
        {
            "candidate_id": f"synthetic:{source}:{fact}",
            "fact_id": fact,
            "dataset": "synthetic",
            "split": "train",
            "relation": "occupation",
            "answer": source,
            "expected": "unused",
            "question": f"What does {fact} do?",
        }
        for fact in ("one", "two", "three")
        for source in ("adapter_a", "adapter_b")
    ]
    assigned = _assign_control_questions(rows, repeats=2, seed=7)
    assert all(len(set(row["control_questions"])) == 2 for row in assigned)


def test_feature_table_builds_common_and_intervention_scores():
    row = {
        "scores": {
            "base": {"support_margin": 1.0},
            "adapter_a": {"support_margin": 3.0},
            "adapter_b": {"support_margin": 5.0},
        }
    }
    names, matrix = _feature_table([row])
    values = dict(zip(names, matrix[0], strict=True))
    assert values["mean_support"] == 3.0
    assert values["min_support"] == 1.0
    assert values["max_support"] == 5.0
    assert values["common_lora_shift"] == 3.0
    assert values["adapter_opposition"] == 2.0


def test_source_identity_metric_is_orientation_invariant():
    rows = [
        {"candidate_source": "adapter_a"},
        {"candidate_source": "adapter_a"},
        {"candidate_source": "adapter_b"},
        {"candidate_source": "adapter_b"},
    ]
    assert _candidate_source_identity(rows, np.asarray([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert _candidate_source_identity(rows, np.asarray([0.9, 0.8, 0.2, 0.1])) == 1.0


def test_numpy_auroc_handles_ties():
    labels = np.asarray([0, 0, 1, 1])
    assert _safe_auroc(labels, np.asarray([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert _safe_auroc(labels, np.asarray([0.5, 0.5, 0.5, 0.5])) == 0.5
