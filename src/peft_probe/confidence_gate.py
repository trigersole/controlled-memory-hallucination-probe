from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import output_dir
from .versioning import artifact_metadata


SCORE_FUNCTIONS = {
    "token_logprob": lambda records: 1.0
    - np.exp(np.asarray([row["mean_logprob"] for row in records], dtype=float)),
    "token_entropy": lambda records: 1.0
    - np.exp(-np.asarray([row["mean_entropy"] for row in records], dtype=float)),
}


def load_confidence_gate_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError("Confidence-gate configuration must be a mapping")
    if settings.get("confidence_gate_schema_version") != 1:
        raise ValueError("confidence_gate_schema_version must be 1")
    candidates = settings.get("candidate_scores", [])
    if not candidates or any(name not in SCORE_FUNCTIONS for name in candidates):
        raise ValueError(f"candidate_scores must use {sorted(SCORE_FUNCTIONS)}")
    if settings.get("fit_split", "validation") != "validation":
        raise ValueError("The confidence gate must be selected on the validation split")
    if settings.get("test_split", "test") != "test":
        raise ValueError("The frozen confidence gate must be evaluated on the test split")
    if int(settings.get("bootstrap_samples", 0)) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    return settings


def _threshold_metrics(
    labels_correct: np.ndarray, risk: np.ndarray, threshold: float
) -> dict[str, float]:
    labels_correct = np.asarray(labels_correct, dtype=int)
    predicted_error = np.asarray(risk, dtype=float) >= threshold
    actual_error = labels_correct == 0
    tp = int(np.sum(predicted_error & actual_error))
    fp = int(np.sum(predicted_error & ~actual_error))
    tn = int(np.sum(~predicted_error & ~actual_error))
    fn = int(np.sum(~predicted_error & actual_error))

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    error_recall = ratio(tp, tp + fn)
    correct_acceptance = ratio(tn, tn + fp)
    accepted = ~predicted_error
    return {
        "n": int(len(labels_correct)),
        "answer_accuracy": (
            float(labels_correct.mean()) if len(labels_correct) else float("nan")
        ),
        "threshold": float(threshold),
        "balanced_accuracy": float(np.nanmean([error_recall, correct_acceptance])),
        "gate_accuracy": ratio(tp + tn, len(labels_correct)),
        "error_recall": error_recall,
        "correct_acceptance": correct_acceptance,
        "error_precision": ratio(tp, tp + fp),
        "coverage": float(accepted.mean()) if len(accepted) else float("nan"),
        "selective_accuracy": (
            float(labels_correct[accepted].mean()) if accepted.any() else float("nan")
        ),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _ranking_metrics(labels_correct: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels_error = 1 - np.asarray(labels_correct, dtype=int)
    if np.unique(labels_error).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels_error, risk)),
        "auprc": float(average_precision_score(labels_error, risk)),
    }


def _fit_threshold(labels_correct: np.ndarray, risk: np.ndarray) -> tuple[float, dict[str, float]]:
    unique = np.unique(np.asarray(risk, dtype=float))
    if not len(unique):
        raise ValueError("Cannot fit a threshold without validation examples")
    candidates = [float(np.nextafter(unique[0], -np.inf))]
    candidates.extend(float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:]))
    candidates.append(float(np.nextafter(unique[-1], np.inf)))
    evaluated = [
        (_threshold_metrics(labels_correct, risk, threshold), threshold)
        for threshold in candidates
    ]
    # Resolve a tie conservatively: catch more errors, then preserve more correct answers.
    best, threshold = max(
        evaluated,
        key=lambda item: (
            item[0]["balanced_accuracy"],
            item[0]["error_recall"],
            item[0]["coverage"],
        ),
    )
    return threshold, best


def _eligible(
    records: list[dict[str, Any]],
    split: str | None,
    exclude_abstentions: bool,
) -> list[dict[str, Any]]:
    selected = records
    if split is not None:
        selected = [row for row in selected if row.get("split") == split]
    selected = [row for row in selected if int(row.get("answer_tokens", 1)) > 0]
    if exclude_abstentions:
        selected = [row for row in selected if not bool(row.get("abstained", False))]
    return selected


def _evaluate(records: list[dict[str, Any]], score_name: str, threshold: float) -> dict[str, Any]:
    labels = np.asarray([row["correct"] for row in records], dtype=int)
    risk = SCORE_FUNCTIONS[score_name](records)
    return {**_ranking_metrics(labels, risk), **_threshold_metrics(labels, risk, threshold)}


def _bootstrap_gap(
    pairs: list[dict[str, Any]], key: str, samples: int, seed: int
) -> dict[str, Any]:
    entities: dict[str, list[float]] = {}
    for pair in pairs:
        entities.setdefault(pair["entity_id"], []).append(float(pair[key]))
    arrays = [np.asarray(entities[name], dtype=float) for name in sorted(entities)]
    values = np.concatenate(arrays)
    rng = np.random.default_rng(seed)
    draws = [
        float(
            np.concatenate(
                [arrays[index] for index in rng.integers(0, len(arrays), len(arrays))]
            ).mean()
        )
        for _ in range(samples)
    ]
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "fraction_positive": float(np.mean(values > 0)),
        "fraction_tied": float(np.mean(values == 0)),
    }


def _paired_gaps(
    records: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    by_fact: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_fact.setdefault(row["fact_id"], []).append(row)
    pairs = []
    for rows in by_fact.values():
        exposed = [row for row in rows if int(row["exposure"]) == 1]
        withheld = [row for row in rows if int(row["exposure"]) == 0]
        if len(exposed) != 1 or len(withheld) != 1:
            continue
        known, unknown = exposed[0], withheld[0]
        pairs.append(
            {
                "entity_id": known["entity_id"],
                "relation": known["relation"],
                "logprob_gap": float(known["mean_logprob"] - unknown["mean_logprob"]),
                "entropy_gap": float(unknown["mean_entropy"] - known["mean_entropy"]),
                "exposed_correct": int(known["correct"]),
                "withheld_correct": int(unknown["correct"]),
            }
        )
    if not pairs:
        raise RuntimeError("No complementary exposed/withheld fact pairs were found")

    def summarize(group: list[dict[str, Any]], offset: int) -> dict[str, Any]:
        return {
            "n": len(group),
            "logprob_gap": _bootstrap_gap(group, "logprob_gap", samples, seed + offset),
            "entropy_gap": _bootstrap_gap(group, "entropy_gap", samples, seed + offset + 1),
            "exposed_accuracy": float(np.mean([row["exposed_correct"] for row in group])),
            "withheld_accuracy": float(np.mean([row["withheld_correct"] for row in group])),
        }

    relations = sorted({row["relation"] for row in pairs})
    return {
        "all": summarize(pairs, 0),
        "by_relation": {
            relation: summarize(
                [row for row in pairs if row["relation"] == relation], 10 + index * 2
            )
            for index, relation in enumerate(relations)
        },
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]
    test = report["synthetic_test"]["all"]

    def table_row(name: str, metrics: dict[str, Any]) -> str:
        values = (
            name,
            metrics["n"],
            metrics["auroc"],
            metrics["balanced_accuracy"],
            metrics["error_recall"],
            metrics["correct_acceptance"],
            metrics["coverage"],
            metrics["selective_accuracy"],
        )
        return "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            *values
        )

    lines = [
        "# Frozen confidence gate",
        "",
        "The score and cutoff were selected using synthetic validation labels only. "
        "Synthetic test and natural benchmark labels are evaluation-only.",
        "",
        f"Selected score: `{selected['score']}`; risk threshold: "
        f"`{selected['threshold']:.6f}`.",
        "",
        "| Evaluation | N | AUROC | Balanced accuracy | Error recall | "
        "Correct acceptance | Coverage | Selective accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        table_row("Synthetic validation", selected["metrics"]),
        table_row("Synthetic test", test),
    ]
    for name, metrics in report["benchmarks"].items():
        lines.append(table_row(name, metrics))
    gap = report["paired_test_gap"]["all"]
    lines.extend(
        [
            "",
            "## Paired same-fact confidence gap",
            "",
            f"Across {gap['n']} test facts, the exposed-minus-withheld log-probability gap "
            f"was {gap['logprob_gap']['mean']:.6f} (95% entity-bootstrap CI "
            f"{gap['logprob_gap']['ci95'][0]:.6f} to "
            f"{gap['logprob_gap']['ci95'][1]:.6f}); "
            f"{gap['logprob_gap']['fraction_positive']:.4f} were positive.",
            "",
            "Positive gaps mean the adapter trained on the exact fact was more confident. "
            "Benchmark application uses the frozen absolute cutoff, not an unavailable "
            "paired oracle.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_confidence_gate(
    config: dict[str, Any], settings: dict[str, Any], force: bool = False
) -> Path:
    # Keep the pure selection helpers importable in lightweight CPU test environments.
    from .collect import configured_feature_keys, load_feature_shards
    from .io_utils import atomic_json

    root = output_dir(config)
    target = root / "results" / "confidence_gate_v1"
    report_path = target / "report.json"
    if report_path.exists() and not force:
        return report_path
    target.mkdir(parents=True, exist_ok=True)

    mode = configured_feature_keys(config)[0]
    synthetic = []
    for source in ("adapter_a", "adapter_b"):
        rows, _ = load_feature_shards(root / "features" / "synthetic" / source, mode)
        synthetic.extend(rows)
    exclude_abstentions = bool(settings.get("exclude_abstentions", True))
    validation = _eligible(synthetic, "validation", exclude_abstentions)
    test = _eligible(synthetic, "test", exclude_abstentions)
    if not validation or not test:
        raise RuntimeError("Synthetic validation and test records are required")

    candidates = {}
    for name in settings["candidate_scores"]:
        labels = np.asarray([row["correct"] for row in validation], dtype=int)
        risk = SCORE_FUNCTIONS[name](validation)
        threshold, metrics = _fit_threshold(labels, risk)
        candidates[name] = {
            "threshold": threshold,
            "metrics": {**_ranking_metrics(labels, risk), **metrics},
        }
    selected_name = max(
        settings["candidate_scores"],
        key=lambda name: (
            candidates[name]["metrics"]["balanced_accuracy"],
            candidates[name]["metrics"]["auroc"],
        ),
    )
    threshold = float(candidates[selected_name]["threshold"])

    synthetic_groups = {"all": test}
    synthetic_groups.update(
        {
            source: [row for row in test if row["source"] == source]
            for source in ("adapter_a", "adapter_b")
        }
    )
    synthetic_groups.update(
        {
            f"relation/{relation}": [
                row for row in test if row["relation"] == relation
            ]
            for relation in sorted({row["relation"] for row in test})
        }
    )
    benchmark_results = {}
    for benchmark in config["benchmarks"]["datasets"]:
        benchmark_root = root / "features" / "benchmarks" / benchmark
        if not benchmark_root.exists():
            continue
        rows, _ = load_feature_shards(benchmark_root, mode)
        rows = [
            row
            for row in _eligible(rows, None, exclude_abstentions)
            if int(row["correct"]) >= 0
        ]
        if rows:
            benchmark_results[benchmark] = _evaluate(rows, selected_name, threshold)

    report = {
        "method": "confidence_gate_v1",
        "selection_policy": "maximize balanced accuracy; ties favour error recall then coverage",
        "exclude_abstentions": exclude_abstentions,
        "candidate_validation_results": candidates,
        "selection": {
            "score": selected_name,
            "threshold": threshold,
            "metrics": candidates[selected_name]["metrics"],
        },
        "synthetic_test": {
            name: _evaluate(rows, selected_name, threshold)
            for name, rows in synthetic_groups.items()
            if rows
        },
        "paired_test_gap": _paired_gaps(
            test, int(settings["bootstrap_samples"]), int(settings.get("seed", 62626))
        ),
        "benchmarks": benchmark_results,
        **artifact_metadata(),
    }
    atomic_json(report_path, report)
    _write_markdown(target / "report.md", report)
    return report_path
