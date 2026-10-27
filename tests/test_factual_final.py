import copy
import json

import numpy as np
import pytest
import torch

from peft_probe import factual_final as final
from peft_probe import factual_shift as factual
from peft_probe.config import config_fingerprint
from peft_probe.within_question import BASELINE, evaluate_within_question

A = "-17/correct_a/delta/fraction_8"
TRANSFER = A + "/a_to_b"
INCORRECT = "-17/incorrect/delta/fraction_8"


def saved_run():
    rows = [
        {
            "row_id": f"{split}:{i}",
            "fact_id": split,
            "group_id": split,
            "split": split,
            "question": split,
            "answer": str(i),
            "correct": label,
        }
        for split, labels in (("direction_fit", [1]), ("test", [1, 0, -1]), ("ood", [-1, -1]))
        for i, label in enumerate(labels)
    ]
    raw = np.array([r["correct"] for r in rows], dtype=np.float32)
    artifacts = {
        "selected": A,
        "selected_transfer": A,
        "selected_by_family": {"correct_a": A, "incorrect": INCORRECT},
        "signs": {A: -1, INCORRECT: 1, BASELINE: 1},
    }
    cached = {
        "fitted": {
            "artifacts": artifacts,
            "scores": {A: -raw, INCORRECT: -raw, BASELINE: raw},
            # A's sign must be inherited: transfer is deliberately reversed on test.
            "transfers": {A: raw},
        }
    }
    metrics = {
        "selected_on_validation": A,
        "selected_transfer_on_a_validation": A,
        "selected_by_family": artifacts["selected_by_family"],
        "methods": {
            m: {"orientation": s}
            for m, s in ((A, -1), (TRANSFER, -1), (INCORRECT, 1), (BASELINE, 1))
        },
    }
    predictions = [
        {**row, "method": m, "score": float(raw[index])}
        for index, row in enumerate(rows)
        if row["split"] in {"test", "ood"}
        for m in (A, BASELINE)
    ]
    return cached, metrics, rows, predictions


def test_frozen_transfer_inherits_a_sign_and_control_retains_its_own_sign():
    rows, methods, comparisons, selections = final.frozen_rows(*saved_run())
    test_rows = [r for r in rows if r["split"] == "test"]
    report, _ = evaluate_within_question(test_rows, methods, 100, 42, comparisons)
    assert report["n_mixed_questions"] == 1
    assert report["methods"][A]["within_question_pairwise_accuracy"] == 1
    assert report["methods"][TRANSFER]["within_question_pairwise_accuracy"] == 0
    assert report["methods"][INCORRECT]["within_question_pairwise_accuracy"] == 0
    assert report["paired_comparisons"][:2] == [
        {"left": TRANSFER, "right": A, "mean": -1, "ci95": [-1, -1]},
        {"left": A, "right": INCORRECT, "mean": 1, "ci95": [1, 1]},
    ]
    assert selections["incorrect_matched_geometry"] == INCORRECT
    ood, _ = evaluate_within_question(
        [r for r in rows if r["split"] == "ood"], methods, 100, 42, comparisons
    )
    assert all(item["mean"] is None and item["ci95"] is None for item in ood["paired_comparisons"])
    assert "unavailable" in final._interpret(ood, selections)[0]


@pytest.mark.parametrize(
    "problem", ["sign", "selection", "shape", "nan", "missing", "duplicate", "text", "score"]
)
def test_stale_or_misaligned_saved_inputs_are_rejected(problem):
    cached, metrics, rows, predictions = saved_run()
    if problem == "sign":
        metrics["methods"][TRANSFER]["orientation"] = 1
    elif problem == "selection":
        metrics["selected_on_validation"] = "different"
    elif problem == "shape":
        cached["fitted"]["transfers"][A] = np.zeros(1)
    elif problem == "nan":
        cached["fitted"]["transfers"][A][0] = np.nan
    elif problem == "missing":
        predictions.pop()
    elif problem == "duplicate":
        predictions.append(predictions[0])
    elif problem == "text":
        predictions[0]["answer"] = "different"
    elif problem == "score":
        predictions[0]["score"] = -999
    with pytest.raises(ValueError):
        final.frozen_rows(cached, metrics, rows, predictions)


def test_independent_control_and_matched_geometry_are_both_preserved():
    cached, metrics, rows, predictions = saved_run()
    other = "-9/incorrect/delta/fraction_4"
    cached = copy.deepcopy(cached)
    metrics = copy.deepcopy(metrics)
    for families in (
        cached["fitted"]["artifacts"]["selected_by_family"],
        metrics["selected_by_family"],
    ):
        families["incorrect"] = other
    cached["fitted"]["scores"][other] = cached["fitted"]["scores"][INCORRECT]
    cached["fitted"]["artifacts"]["signs"][other] = -1
    metrics["methods"][other] = {"orientation": -1}
    _, methods, comparisons, selections = final.frozen_rows(cached, metrics, rows, predictions)
    assert INCORRECT in methods and other in methods
    assert comparisons == [(TRANSFER, A), (A, other), (A, INCORRECT), (A, BASELINE)]
    assert selections["incorrect_validation_selected"] == other


def test_paired_intervals_use_same_alias_cluster_draws():
    rows = [
        {
            "fact_id": q,
            "group_id": group,
            "question": q,
            "correct": y,
            "scores": {"a": score, "b": score + 10},
        }
        for q, group, scores in (
            ("q1", "alias", [1, 0]),
            ("q2", "alias", [0, 1]),
            ("q3", "other", [1, 0]),
        )
        for y, score in zip([1, 0], scores, strict=True)
    ]
    report, _ = evaluate_within_question(rows, ["a", "b"], 100, 42, [("a", "b")])
    assert report["n_mixed_clusters"] == 2
    assert report["methods"]["a"]["within_question_ci95"] == [0.5, 1]
    assert report["paired_comparisons"][0]["ci95"] == [0, 0]


def test_final_report_uses_only_cache_and_preserves_original_files(tmp_path, monkeypatch):
    cached, metrics, rows, predictions = saved_run()
    config = {"experiment": {"seed": 42}, "factual_shift": {"bootstrap_samples": 50}}
    results = tmp_path / "results"
    results.mkdir()
    (results / "_SUCCESS.json").write_text('{"complete": true}')
    signature = {
        "checkpoint_version": 1,
        "config_fingerprint": config_fingerprint(config),
        "rows_sha256": factual._digest(rows),
    }
    cached["signature"] = signature
    metrics["rows_sha256"] = signature["rows_sha256"]
    torch.save(cached, results / "analysis_fit.pt")
    (results / "metrics.json").write_text(json.dumps(metrics))
    (results / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in predictions))
    (results / "within_question").mkdir()
    (results / "within_question" / "summary.md").write_text("original diagnostic")
    before = {p: p.read_bytes() for p in results.rglob("*") if p.is_file()}
    monkeypatch.setattr(factual, "root", lambda cfg: tmp_path)
    monkeypatch.setattr(factual, "_replay_rows", lambda cfg: rows)
    for name in ("load_features", "_fit_analysis", "load_base_model", "train"):
        monkeypatch.setattr(factual, name, lambda *a, **kw: pytest.fail("Must only reuse scores"))
    output = final.analyze_final_checks(config)
    assert (output / "_SUCCESS.json").exists()
    text = (output / "summary.md").read_text()
    assert "No mixed questions" in text
    assert "incorrect control" in text
    assert all(p.read_bytes() == original for p, original in before.items())
    report = json.loads((output / "metrics.json").read_text())
    assert report["datasets"]["test"]["paired_comparisons"][1]["mean"] == 1
    config["experiment"]["seed"] = 43
    with pytest.raises(ValueError, match="Checkpoint inputs changed"):
        final.analyze_final_checks(config, force=True)
