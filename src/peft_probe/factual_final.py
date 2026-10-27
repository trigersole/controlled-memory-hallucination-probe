"""Final paired diagnostics from an existing, trusted local analysis checkpoint."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .config import config_fingerprint
from .within_question import BASELINE, _json, _read_rows, _sha, evaluate_within_question


def frozen_rows(cached: dict, metrics: dict, rows: list[dict], predictions: list[dict]):
    """Validate frozen selections/signs and join cached arrays to their original corpus."""
    fitted = cached["fitted"]
    artifacts = fitted["artifacts"]
    selected = artifacts["selected"]
    source = artifacts["selected_transfer"]
    incorrect = artifacts["selected_by_family"]["incorrect"]
    if (
        selected != metrics["selected_on_validation"]
        or source != metrics["selected_transfer_on_a_validation"]
        or artifacts["selected_by_family"] != metrics["selected_by_family"]
    ):
        raise ValueError("Checkpoint and published frozen selections disagree")
    # Prefer the primary's own transfer whenever available. Otherwise explicitly use
    # the original transfer source selected on A validation, and include its native scores.
    source = selected if selected in fitted["transfers"] else source
    transfer = f"{source}/a_to_b"
    methods = list(dict.fromkeys([selected, source, transfer, incorrect, BASELINE]))
    if "answer_length_words" in fitted["scores"]:
        methods.append("answer_length_words")
    if metrics.get("selection_rule") == "predeclared_method_validation_sign":
        layer = selected.split("/")[0]
        for name in (
            artifacts["selected_by_family"].get("activation_pca"),
            f"{layer}/supervised_probe",
            f"{layer}/random/base",
        ):
            if name in fitted["scores"] and name not in methods:
                methods.append(name)
    matched = selected.replace("/correct_a/", "/incorrect/")
    if matched != selected and matched in fitted["scores"] and matched not in methods:
        methods.append(matched)
    signs, arrays = {}, {}
    for method in methods:
        native = source if method == transfer else method
        sign = artifacts["signs"][native]
        if sign not in (-1, 1) or metrics["methods"][method]["orientation"] != sign:
            raise ValueError(f"Frozen orientation mismatch: {method}")
        table = fitted["transfers"] if method == transfer else fitted["scores"]
        raw = np.asarray(table[native], dtype=float)
        if raw.shape != (len(rows),) or not np.isfinite(raw).all():
            raise ValueError(f"Invalid cached score array: {method}")
        signs[method] = sign
        arrays[method] = sign * raw
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate replay row IDs")
    heldout = []
    identities = {}
    for index, row in enumerate(rows):
        if row["split"] not in {"test", "ood"}:
            continue
        identity = (row["split"], row["question"], row["group_id"])
        if identities.setdefault(row["fact_id"], identity) != identity:
            raise ValueError(f"Inconsistent question identity: {row['fact_id']}")
        heldout.append({**row, "scores": {m: float(arrays[m][index]) for m in methods}})
    if not heldout:
        raise ValueError("No held-out candidates found")
    expected = {r["row_id"]: r for r in heldout}
    seen = set()
    for prediction in predictions:
        method = prediction["method"]
        if method not in {selected, BASELINE}:
            continue
        key = (prediction["row_id"], method)
        if key in seen or prediction["row_id"] not in expected:
            raise ValueError("Duplicate or unexpected saved prediction")
        seen.add(key)
        row = expected[prediction["row_id"]]
        if any(row[k] != prediction[k] for k in ("question", "answer", "correct", "split")):
            raise ValueError("Saved prediction and replay metadata disagree")
        if not math.isclose(
            row["scores"][method], float(prediction["score"]), rel_tol=1e-7, abs_tol=1e-9
        ):
            raise ValueError("Cached oriented score disagrees with saved prediction")
    if seen != {(row_id, method) for row_id in expected for method in {selected, BASELINE}}:
        raise ValueError("Saved predictions do not cover all held-out candidates")
    comparisons = [(transfer, source), (selected, incorrect)]
    if matched in methods and matched != incorrect:
        comparisons.append((selected, matched))
    comparisons.append((selected, BASELINE))
    if "answer_length_words" in methods:
        comparisons.append((selected, "answer_length_words"))
    for name in methods:
        if "activation_pca" in name or name.endswith(("/supervised_probe", "/random/base")):
            comparisons.append((selected, name))
    return (
        heldout,
        methods,
        comparisons,
        {
            "primary": selected,
            "transfer_source": source,
            "transfer": transfer,
            "incorrect_validation_selected": incorrect,
            "incorrect_matched_geometry": matched
            if matched != selected and matched in methods
            else None,
            "orientation": signs,
        },
    )


def analyze_final_checks(config: dict, force: bool = False) -> Path:
    """Never train, refit, regenerate, or read feature/model weights."""
    from . import factual_shift as factual

    results = factual.root(config) / "results"
    marker = results / "_SUCCESS.json"
    if not marker.exists() or not json.loads(marker.read_text(encoding="utf-8")).get("complete"):
        raise ValueError("A completed original analysis is required")
    cache_path = results / "analysis_fit.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Saved score checkpoint is required: {cache_path}")
    rows = factual._replay_rows(config)
    signature = {
        "checkpoint_version": 1,
        "config_fingerprint": config_fingerprint(config),
        "rows_sha256": factual._digest(rows),
    }
    print("[final-checks] Reading frozen CPU score checkpoint", flush=True)
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cached["signature"] != signature:
        raise ValueError("Checkpoint inputs changed; final checks cannot refit or repair analysis")
    metrics_path, predictions_path = results / "metrics.json", results / "predictions.jsonl"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics["rows_sha256"] != signature["rows_sha256"]:
        raise ValueError("Published metrics belong to a different replay corpus")
    heldout, methods, comparisons, selections = frozen_rows(
        cached, metrics, rows, list(_read_rows(predictions_path))
    )
    samples = int(config["factual_shift"]["bootstrap_samples"])
    seed = int(config["experiment"]["seed"])
    provenance = {
        "schema_version": 1,
        "analysis_signature": signature,
        "checkpoint_sha256": _sha(cache_path),
        "metrics_sha256": _sha(metrics_path),
        "predictions_sha256": _sha(predictions_path),
        "bootstrap_samples": samples,
        "seed": seed,
    }
    output = results / "within_question_final"
    manifest = output / "_MANIFEST.json"
    if (
        manifest.exists()
        and json.loads(manifest.read_text(encoding="utf-8")) != provenance
        and not force
    ):
        raise ValueError("Final-check inputs/settings changed; --force rebuilds only this report")
    output.mkdir(exist_ok=True)
    (output / "_SUCCESS.json").unlink(missing_ok=True)
    _json(manifest, provenance)
    datasets, question_rows = {}, []
    lines = [
        "# Final within-question checks",
        "",
        f"Frozen primary: `{selections['primary']}`",
        f"Frozen transfer source: `{selections['transfer_source']}`",
        f"Validation-selected incorrect control: `{selections['incorrect_validation_selected']}`",
        f"Matched geometry control: `{selections['incorrect_matched_geometry']}`",
        f"Selection rule: {metrics.get('selection_rule', 'validation_auroc')}",
        f"OOD candidate mode: {metrics.get('ood_candidate_mode', 'generated')}",
        "",
        "All methods use the same mixed questions, original validation signs and half credit for ties.",
        "Questions have equal weight; paired 95% intervals resample question/alias groups.",
        "Reference-mode OOD scores rank benchmark-supplied answers; they do not measure generated-answer detection.",
        "",
    ]
    for split in ("test", "ood"):
        subset = [r for r in heldout if r["split"] == split]
        if not subset:
            continue
        print(f"[final-checks] Evaluating {split}: {len(subset)} candidates", flush=True)
        report, questions = evaluate_within_question(subset, methods, samples, seed, comparisons)
        datasets[split] = report
        question_rows.extend({"split": split, **q} for q in questions)
        lines.extend(
            [
                f"## {split}",
                "",
                (
                    f"Eligible: {report['n_mixed_questions']}/{report['n_questions']} questions; "
                    f"{report['n_mixed_clusters']} clusters; {report['n_ambiguous']} ambiguous candidates excluded."
                ),
                "",
                "| Method | Pooled AUROC | Within-question pairwise accuracy | 95% CI |",
                "|---|---:|---:|---|",
            ]
        )
        for method, item in report["methods"].items():
            lines.append(
                f"| {method} | {_fmt(item['pooled_auroc'])} | "
                f"{_fmt(item['within_question_pairwise_accuracy'])} | {_interval(item['within_question_ci95'])} |"
            )
        lines.extend(
            ["", "| Paired difference (left minus right) | Difference | 95% CI |", "|---|---:|---|"]
        )
        for item in report["paired_comparisons"]:
            lines.append(
                f"| {item['left']} minus {item['right']} | {_fmt(item['mean'])} | {_interval(item['ci95'])} |"
            )
        lines.extend(["", *_interpret(report, selections), ""])
    lines.extend(
        [
            "These are conditional pairwise rankings on mixed questions, not answer-selection accuracy.",
            "An interval containing zero does not establish equivalence. No test labels select methods or signs.",
            "The incorrect control uses its own previously frozen validation sign.",
            "This single-run diagnostic cannot establish a causal/universal truth direction or out-of-domain robustness.",
        ]
    )
    _json(
        output / "metrics.json",
        {
            "signature": provenance,
            "selections": selections,
            "datasets": datasets,
            "selection_rule": metrics.get("selection_rule", "validation_auroc"),
            "ood_candidate_mode": metrics.get("ood_candidate_mode", "generated"),
        },
    )
    _json(output / "per_question.json", question_rows)
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json(output / "_SUCCESS.json", {"complete": True, "signature": provenance})
    return output


def _fmt(value):
    return "unavailable" if value is None else f"{value:.4f}"


def _interval(value):
    return "unavailable" if value is None else f"[{_fmt(value[0])}, {_fmt(value[1])}]"


def _interpret(report, selections):
    if not report["n_mixed_questions"]:
        return [
            "No mixed questions: within-question transfer and control comparisons are unavailable."
        ]
    transfer = report["methods"][selections["transfer"]]
    lines = [
        "Transferred scores rank correct answers above chance on this cohort."
        if transfer["within_question_ci95"][0] > 0.5
        else "Transferred scores do not show clear above-chance ranking at this interval precision."
    ]
    difference = report["paired_comparisons"][1]
    if difference["ci95"][0] > 0:
        lines.append(
            "The factual primary outperforms the validation-selected incorrect control on this cohort."
        )
    elif difference["ci95"][1] < 0:
        lines.append(
            "The validation-selected incorrect control outperforms the factual primary on this cohort."
        )
    else:
        lines.append(
            "The paired interval does not resolve a difference from the validation-selected incorrect control."
        )
    return lines
