"""CPU-only diagnostic of frozen saved scores, using the Python standard library."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

BASELINE = "base_likelihood"
QUESTION_CONTROL = "question_mean_control"
SCHEMA = 1


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pooled_auroc(labels: list[int], scores: list[float]) -> float | None:
    """P(correct ranks above incorrect), with half credit for equal scores."""
    pairs = sorted(
        (score, label) for label, score in zip(labels, scores, strict=True) if label >= 0
    )
    positive = sum(label == 1 for _, label in pairs)
    negative = sum(label == 0 for _, label in pairs)
    if not positive or not negative:
        return None
    wins, previous_negatives = 0.0, 0
    for _, tied in itertools.groupby(pairs, key=lambda pair: pair[0]):
        labels_in_tie = [label for _, label in tied]
        pos = labels_in_tie.count(1)
        neg = labels_in_tie.count(0)
        wins += pos * (previous_negatives + 0.5 * neg)
        previous_negatives += neg
    return wins / (positive * negative)


def _quantiles(values: list[float]) -> list[float] | None:
    if not values:
        return None
    ordered = sorted(values)
    result = []
    for q in (0.025, 0.975):
        index = (len(ordered) - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        result.append(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower))
    return result


def evaluate_within_question(
    rows: list[dict], methods: list[str], samples: int, seed: int,
    comparisons: list[tuple[str, str]] | None = None,
) -> tuple[dict, list[dict]]:
    """Each row has fact_id/group_id/correct/scores. Each mixed question has equal weight."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    for left, right in comparisons or []:
        if left not in methods or right not in methods:
            raise ValueError("Paired comparison refers to an unknown method")
    grouped = defaultdict(list)
    for row in rows:
        if row["correct"] not in (-1, 0, 1):
            raise ValueError("Correctness must be -1, 0, or 1")
        if any(not math.isfinite(row["scores"][method]) for method in methods):
            raise ValueError("Candidate scores must be finite")
        grouped[row["fact_id"]].append(row)
    questions, centered = [], {method: [] for method in methods}
    labels = []
    for fact_id, candidates in grouped.items():
        clusters = {row["group_id"] for row in candidates}
        if len(clusters) != 1:
            raise ValueError(f"Question {fact_id} has inconsistent bootstrap group IDs")
        positive = [r for r in candidates if r["correct"] == 1]
        negative = [r for r in candidates if r["correct"] == 0]
        pair_count = len(positive) * len(negative)
        accuracies, ties = {}, {}
        for method in methods:
            scores = [row["scores"][method] for row in candidates]
            mean = math.fsum(scores) / len(scores)
            centered[method].extend(score - mean for score in scores)
            pairwise_values = [
                int(pos["scores"][method] > neg["scores"][method])
                + 0.5 * int(pos["scores"][method] == neg["scores"][method])
                for pos in positive
                for neg in negative
            ]
            accuracies[method] = math.fsum(pairwise_values) / pair_count if pair_count else None
            ties[method] = sum(
                pos["scores"][method] == neg["scores"][method]
                for pos in positive
                for neg in negative
            )
        labels.extend(r["correct"] for r in candidates)
        questions.append(
            {
                "fact_id": fact_id,
                "group_id": candidates[0]["group_id"],
                "question": candidates[0]["question"],
                "n_candidates": len(candidates),
                "n_correct": len(positive),
                "n_incorrect": len(negative),
                "n_ambiguous": sum(r["correct"] == -1 for r in candidates),
                "n_pairs": pair_count,
                "pairwise_accuracy": accuracies,
                "ties": ties,
            }
        )
    mixed = [q for q in questions if q["n_pairs"]]
    clusters = defaultdict(list)
    for question in mixed:
        clusters[question["group_id"]].append(question)
    # Resample the same question/alias clusters for every method and paired difference.
    totals = [
        {
            "n": len(items),
            "sums": {m: math.fsum(q["pairwise_accuracy"][m] for q in items) for m in methods},
        }
        for items in clusters.values()
    ]
    bootstrap = {m: [] for m in methods}
    differences = {m: [] for m in methods if m != BASELINE}
    rng = random.Random(seed)
    for _ in range(samples if totals else 0):
        draw = [totals[rng.randrange(len(totals))] for _ in totals]
        count = sum(item["n"] for item in draw)
        means = {m: math.fsum(item["sums"][m] for item in draw) / count for m in methods}
        for method in methods:
            bootstrap[method].append(means[method])
            if method != BASELINE and BASELINE in methods:
                differences[method].append(means[method] - means[BASELINE])
    reports = {}
    # Use grouped ordering for both labels and centered scores; pooled scores use original rows.
    original_labels = [r["correct"] for r in rows]
    for method in methods:
        mean = (
            math.fsum(q["pairwise_accuracy"][method] for q in mixed) / len(mixed) if mixed else None
        )
        reports[method] = {
            "pooled_auroc": pooled_auroc(original_labels, [r["scores"][method] for r in rows]),
            "within_question_pairwise_accuracy": mean,
            "within_question_ci95": _quantiles(bootstrap[method]),
            "question_centered_pooled_auroc": pooled_auroc(labels, centered[method]),
            "n_tied_pairs": sum(q["ties"][method] for q in mixed),
        }
        if method != BASELINE and BASELINE in methods:
            baseline_mean = (
                math.fsum(q["pairwise_accuracy"][BASELINE] for q in mixed) / len(mixed)
                if mixed
                else None
            )
            reports[method]["difference_vs_likelihood"] = {
                "mean": mean - baseline_mean if mixed else None,
                "ci95": _quantiles(differences[method]),
            }
    result = {
        "n_candidates": len(rows),
        "n_graded": sum(r["correct"] >= 0 for r in rows),
        "n_ambiguous": sum(r["correct"] < 0 for r in rows),
        "n_questions": len(questions),
        "n_mixed_questions": len(mixed),
        "n_mixed_clusters": len(clusters),
        "n_no_graded_questions": sum(
            not q["n_correct"] and not q["n_incorrect"] for q in questions
        ),
        "n_one_class_questions": sum(
            bool(q["n_correct"]) != bool(q["n_incorrect"]) for q in questions
        ),
        "n_correct_incorrect_pairs": sum(q["n_pairs"] for q in mixed),
        "methods": reports,
    }
    if comparisons is not None:
        result["paired_comparisons"] = [
            {
                "left": left,
                "right": right,
                "mean": (
                    reports[left]["within_question_pairwise_accuracy"]
                    - reports[right]["within_question_pairwise_accuracy"] if mixed else None
                ),
                "ci95": _quantiles([
                    a - b for a, b in zip(bootstrap[left], bootstrap[right], strict=True)
                ]),
            }
            for left, right in comparisons
        ]
    return result, questions


def analyze_saved_predictions(
    results: str | Path, samples: int = 500, seed: int = 42, force: bool = False
) -> Path:
    """Read only completed saved results and candidate metadata; never refit or flip scores."""
    results = Path(results).resolve()
    marker = results / "_SUCCESS.json"
    if not marker.exists() or not json.loads(marker.read_text(encoding="utf-8")).get("complete"):
        raise FileNotFoundError(f"Original analysis is not complete: {marker}")
    metric_path, prediction_path = results / "metrics.json", results / "predictions.jsonl"
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    selected = metrics["selected_on_validation"]
    methods = list(dict.fromkeys([selected, BASELINE]))
    by_method = {method: {} for method in methods}
    for row in _read_rows(prediction_path):
        if row["method"] not in by_method:
            continue
        if row["split"] not in {"test", "ood"}:
            raise ValueError("Diagnostic accepts held-out test/OOD predictions only")
        table = by_method[row["method"]]
        if row["row_id"] in table:
            raise ValueError(f"Duplicate prediction for {row['row_id']} / {row['method']}")
        if not math.isfinite(float(row["score"])):
            raise ValueError(f"Nonfinite saved score for {row['row_id']}")
        table[row["row_id"]] = row
    if not by_method[selected] or any(
        set(table) != set(by_method[selected]) for table in by_method.values()
    ):
        raise ValueError("Selected and likelihood scores must cover exactly the same candidates")
    candidate_dir = results.parent / "candidates"
    if not (candidate_dir / "_SUCCESS.json").exists():
        raise FileNotFoundError(f"Completed candidate metadata is required: {candidate_dir}")
    metadata, hashes, expected_ids = {}, {}, set()
    for path in sorted(candidate_dir.glob("shard_*.jsonl")):
        hashes[path.name] = _sha(path)
        for row in _read_rows(path):
            if row["split"] in {"test", "ood"}:
                if row["row_id"] in expected_ids:
                    raise ValueError(f"Duplicate candidate metadata for {row['row_id']}")
                expected_ids.add(row["row_id"])
            if row["row_id"] in by_method[selected]:
                if row["row_id"] in metadata:
                    raise ValueError(f"Duplicate candidate metadata for {row['row_id']}")
                metadata[row["row_id"]] = row
    if expected_ids != set(by_method[selected]) or set(metadata) != expected_ids:
        raise ValueError("Candidate metadata does not cover all saved predictions")
    rows = []
    question_identity = {}
    for row_id, candidate in metadata.items():
        for method in methods:
            prediction = by_method[method][row_id]
            if any(
                prediction[key] != candidate[key]
                for key in ("question", "answer", "correct", "split")
            ):
                raise ValueError(f"Saved predictions and candidate metadata disagree: {row_id}")
        identity = (candidate["split"], candidate["question"], candidate["group_id"])
        previous = question_identity.setdefault(candidate["fact_id"], identity)
        if previous != identity:
            raise ValueError(f"Inconsistent identity for question {candidate['fact_id']}")
        rows.append(
            {
                "row_id": row_id,
                "fact_id": candidate["fact_id"],
                "group_id": candidate["group_id"],
                "question": candidate["question"],
                "split": candidate["split"],
                "correct": candidate["correct"],
                "scores": {m: float(by_method[m][row_id]["score"]) for m in methods},
            }
        )
    # This control shares one score across all answers to each question. It uses no labels.
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["fact_id"]].append(row)
    for candidates in grouped.values():
        mean = math.fsum(row["scores"][selected] for row in candidates) / len(candidates)
        for row in candidates:
            row["scores"][QUESTION_CONTROL] = mean
    methods.append(QUESTION_CONTROL)
    signature = {
        "schema_version": SCHEMA,
        "metrics_sha256": _sha(metric_path),
        "predictions_sha256": _sha(prediction_path),
        "candidate_sha256": hashes,
        "bootstrap_samples": samples,
        "seed": seed,
    }
    output = results / "within_question"
    output.mkdir(exist_ok=True)
    manifest = output / "_MANIFEST.json"
    if (
        manifest.exists()
        and json.loads(manifest.read_text(encoding="utf-8")) != signature
        and not force
    ):
        raise ValueError(
            "Within-question inputs/settings changed; use --force to rebuild this diagnostic only"
        )
    (output / "_SUCCESS.json").unlink(missing_ok=True)
    _json(manifest, signature)
    reports, question_rows = {}, []
    for split in ("test", "ood"):
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            continue
        print(f"[within-question] Evaluating {split}: {len(subset)} candidates", flush=True)
        report, questions = evaluate_within_question(subset, methods, samples, seed)
        reports[split] = report
        question_rows.extend({"split": split, **row} for row in questions)
    _json(
        output / "metrics.json",
        {
            "selected_method": selected,
            "signature": signature,
            "orientation": {
                m: metrics.get("methods", {}).get(m, {}).get("orientation") for m in methods
            },
            "datasets": reports,
            "limitations": [
                "No methods or signs are selected using within-question/test labels.",
                "Question-mean control is derived from answer scores, not prompt-only activations.",
                "Only questions with both graded correct and incorrect candidates inform the primary metric.",
                "Intervals resample question/alias groups; questions have equal weight within each estimate.",
            ],
        },
    )
    _json(output / "per_question.json", question_rows)
    lines = [
        "# Within-question correctness diagnostic",
        "",
        f"Frozen method: `{selected}`",
        "",
        "Scores and signs are used exactly as saved. Pairwise ties receive half credit.",
        "Each mixed question has equal weight; 95% intervals bootstrap question/alias groups.",
        "",
        "| Dataset | Method | Pooled AUROC | Within-question accuracy | 95% CI | Mixed questions |",
        "|---|---|---:|---:|---|---:|",
    ]

    def fmt(value):
        return f"{value:.4f}" if value is not None else "unavailable"

    for split, report in reports.items():
        for method, item in report["methods"].items():
            ci = item["within_question_ci95"]
            interval = f"[{fmt(ci[0])}, {fmt(ci[1])}]" if ci else "unavailable"
            lines.append(
                f"| {split} | {method} | {fmt(item['pooled_auroc'])} | "
                f"{fmt(item['within_question_pairwise_accuracy'])} | {interval} | "
                f"{report['n_mixed_questions']} |"
            )
        diff = report["methods"][selected].get("difference_vs_likelihood", {})
        lines.extend(
            [
                "",
                (
                    f"{split}: selected minus likelihood within-question difference: "
                    f"{fmt(diff.get('mean'))}; paired 95% CI: {diff.get('ci95')}."
                ),
                "",
                (
                    f"Eligible: {report['n_mixed_questions']}/{report['n_questions']} questions; "
                    f"{report['n_mixed_clusters']} bootstrap clusters; "
                    f"{report['n_ambiguous']} ambiguous candidates excluded from comparisons."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "Chance/tied pairwise accuracy is 0.5. High pooled AUROC with near-chance",
            "within-question ranking is consistent with question-level familiarity/difficulty.",
            "Above-chance within-question ranking indicates answer-specific predictive information",
            "on mixed questions; it does not establish a causal truth representation.",
            "",
            "The question-mean control uses answer scores and is not a question-only activation test.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json(output / "_SUCCESS.json", {"complete": True, "signature": signature})
    return output
