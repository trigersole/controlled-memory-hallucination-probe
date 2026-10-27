import json
import subprocess
import sys
from pathlib import Path

import pytest

from peft_probe.within_question import (
    BASELINE,
    QUESTION_CONTROL,
    analyze_saved_predictions,
    evaluate_within_question,
    pooled_auroc,
)

METHOD = "-17/correct_a/delta/fraction_8"


def rows_for(question, labels, scores, baseline=None, group=None):
    return [
        {
            "fact_id": question,
            "group_id": group or question,
            "question": question,
            "correct": label,
            "scores": {METHOD: score, BASELINE: base},
        }
        for label, score, base in zip(labels, scores, baseline or scores, strict=True)
    ]


def test_question_only_scores_can_have_high_pooled_auc_and_chance_within_question():
    rows = (
        rows_for("easy", [1, 1], [10, 10])
        + rows_for("hard", [0, 0], [0, 0])
        + rows_for("mixed", [1, 0], [5, 5])
    )
    report, _ = evaluate_within_question(rows, [METHOD, BASELINE], 100, 7)
    selected = report["methods"][METHOD]
    assert selected["pooled_auroc"] > 0.9
    assert selected["within_question_pairwise_accuracy"] == 0.5
    assert selected["within_question_ci95"] == [0.5, 0.5]
    assert selected["question_centered_pooled_auroc"] == 0.5
    assert report["n_mixed_questions"] == 1
    assert report["n_one_class_questions"] == 2


def test_questions_have_equal_weight_and_identical_methods_have_zero_paired_difference():
    rows = rows_for("three_pairs", [1, 1, 1, 0], [1, 1, 1, 0]) + rows_for(
        "one_pair", [1, 0], [0, 1]
    )
    report, _ = evaluate_within_question(rows, [METHOD, BASELINE], 100, 7)
    assert report["n_correct_incorrect_pairs"] == 4
    assert report["methods"][METHOD]["within_question_pairwise_accuracy"] == 0.5
    assert report["methods"][METHOD]["difference_vs_likelihood"] == {
        "mean": 0.0,
        "ci95": [0.0, 0.0],
    }


def test_bad_frozen_orientation_is_not_flipped_on_test():
    rows = rows_for("mixed", [1, 0], [-2, 2], [2, -2])
    report, _ = evaluate_within_question(rows, [METHOD, BASELINE], 100, 3)
    assert report["methods"][METHOD]["within_question_pairwise_accuracy"] == 0
    assert report["methods"][METHOD]["difference_vs_likelihood"] == {
        "mean": -1.0,
        "ci95": [-1.0, -1.0],
    }


def test_alias_clusters_are_resampled_together_with_all_question_weights():
    rows = (
        rows_for("one", [1, 0], [2, 0], group="shared_alias")
        + rows_for("two", [1, 0], [0, 2], group="shared_alias")
        + rows_for("three", [1, 0], [2, 0], group="other_alias")
    )
    report, _ = evaluate_within_question(rows, [METHOD, BASELINE], 1000, 42)
    assert report["n_mixed_clusters"] == 2
    assert report["methods"][METHOD]["within_question_pairwise_accuracy"] == pytest.approx(2 / 3)
    assert report["methods"][METHOD]["within_question_ci95"] == [0.5, 1.0]


def test_no_mixed_questions_are_reported_as_unavailable_and_ambiguity_is_counted():
    rows = (
        rows_for("correct", [1, 1], [2, 2])
        + rows_for("wrong", [0, 0], [0, 0])
        + rows_for("ambiguous", [-1, -1], [9, 9])
    )
    report, _ = evaluate_within_question(rows, [METHOD, BASELINE], 100, 42)
    assert report["n_mixed_questions"] == 0
    assert report["n_no_graded_questions"] == 1
    assert report["n_ambiguous"] == 2
    assert report["n_graded"] == 4
    assert report["methods"][METHOD]["within_question_pairwise_accuracy"] is None
    assert report["methods"][METHOD]["within_question_ci95"] is None
    assert report["methods"][METHOD]["pooled_auroc"] == 1


def test_pooled_auc_ties_and_degenerate_cases():
    assert pooled_auroc([1, 0], [1, 1]) == 0.5
    assert pooled_auroc([1, 0, -1], [0, 1, 100]) == 0
    assert pooled_auroc([1, 1], [0, 1]) is None


def make_saved_run(tmp_path):
    root = tmp_path / "factual_shift_v1"
    results, candidates = root / "results", root / "candidates"
    results.mkdir(parents=True)
    candidates.mkdir()
    for folder in (results, candidates):
        (folder / "_SUCCESS.json").write_text(json.dumps({"complete": True}))
    (results / "metrics.json").write_text(
        json.dumps(
            {
                "selected_on_validation": METHOD,
                "methods": {METHOD: {"orientation": -1}, BASELINE: {"orientation": 1}},
            }
        )
    )
    metadata, predictions = [], []
    for split in ("test", "ood"):
        for i, correct in enumerate((0, 1, -1)):
            row = {
                "row_id": f"{split}:question:{i}",
                "fact_id": f"{split}:question",
                "group_id": f"{split}:alias",
                "question": f"Question for {split}",
                "answer": f"Answer {i}",
                "correct": correct,
                "split": split,
            }
            metadata.append(row)
            for method in (METHOD, BASELINE):
                predictions.append(
                    {key: row[key] for key in ("row_id", "question", "answer", "correct", "split")}
                    | {"method": method, "score": float(correct)}
                )
    (candidates / "shard_00000.jsonl").write_text(
        "\n".join(json.dumps(row) for row in reversed(metadata)) + "\n"
    )
    (results / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in predictions) + "\n"
    )
    return results


def test_saved_score_analysis_joins_by_id_preserves_inputs_and_records_control(tmp_path):
    results = make_saved_run(tmp_path)
    before = {p: p.read_bytes() for p in results.iterdir() if p.is_file()}
    output = analyze_saved_predictions(results, samples=100, seed=7)
    report = json.loads((output / "metrics.json").read_text())
    assert report["selected_method"] == METHOD
    assert report["orientation"][METHOD] == -1  # Metadata recorded, not applied a second time.
    for split in ("test", "ood"):
        assert (
            report["datasets"][split]["methods"][METHOD]["within_question_pairwise_accuracy"] == 1
        )
        assert (
            report["datasets"][split]["methods"][QUESTION_CONTROL][
                "within_question_pairwise_accuracy"
            ]
            == 0.5
        )
    assert (output / "_SUCCESS.json").exists()
    assert (output / "summary.md").exists()
    assert all(path.read_bytes() == value for path, value in before.items())
    with pytest.raises(ValueError, match="settings changed"):
        analyze_saved_predictions(results, samples=100, seed=8)


@pytest.mark.parametrize(
    "change", ["missing_baseline", "duplicate", "label_mismatch", "nonfinite", "omitted_candidate"]
)
def test_inconsistent_saved_predictions_are_rejected(tmp_path, change):
    results = make_saved_run(tmp_path)
    path = results / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "missing_baseline":
        rows = [row for row in rows if row["method"] != BASELINE]
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "label_mismatch":
        rows[0]["correct"] = 1
    elif change == "nonfinite":
        rows[0]["score"] = float("nan")
    else:
        rows = [row for row in rows if row["row_id"] != "test:question:0"]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError):
        analyze_saved_predictions(results, 10, 7)


def test_script_runs_with_no_site_packages_or_model_libraries(tmp_path):
    results = make_saved_run(tmp_path)
    script = Path("scripts/analyze_within_question.py").resolve()
    process = subprocess.run(
        [sys.executable, "-I", "-S", str(script), str(results), "--bootstrap-samples", "10"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert (results / "within_question" / "summary.md").exists()
