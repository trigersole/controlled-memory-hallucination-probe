from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from .config import output_dir


METRICS = ("auroc", "auprc", "brier", "ece", "aurc")
HIGHER_IS_BETTER = {"auroc", "auprc"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed result: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def _metric_row(label: str, values: dict[str, Any]) -> str:
    cells = [label]
    for metric in METRICS:
        value = values.get(metric, {})
        if isinstance(value, dict) and "mean" in value:
            cells.append(f"{_number(value['mean'])} ± {_number(value.get('std', 0.0))}")
        else:
            cells.append(_number(value))
    return "| " + " | ".join(cells) + " |"


def _append_comparison(
    lines: list[str],
    title: str,
    comparisons: dict[str, Any],
    ensemble: dict[str, Any],
    feature_modes: list[str],
) -> None:
    lines.extend(
        [
            f"### {title}",
            "",
            "Positive deltas favor genuine exposure for AUROC/AUPRC; negative deltas favor it "
            "for Brier/ECE/AURC. `Supported seeds` counts per-seed 95% CIs excluding zero "
            "in the favorable direction.",
            "",
            "| Feature mode | Metric | Mean seed delta | Supported seeds | Ensemble delta | "
            "Ensemble 95% CI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for mode in feature_modes:
        mode_runs = [value for key, value in comparisons.items() if key.startswith(f"{mode}/")]
        for metric in METRICS:
            deltas = [float(run[metric]["mean_difference"]) for run in mode_runs]
            supported = 0
            for run in mode_runs:
                lower, upper = run[metric]["ci95"]
                if metric in HIGHER_IS_BETTER and lower > 0:
                    supported += 1
                elif metric not in HIGHER_IS_BETTER and upper < 0:
                    supported += 1
            ensemble_result = ensemble[mode][metric]
            lower, upper = ensemble_result["ci95"]
            lines.append(
                f"| {mode} | {metric} | {_number(fmean(deltas))} | "
                f"{supported}/{len(mode_runs)} | "
                f"{_number(ensemble_result['mean_difference'])} | "
                f"[{_number(lower)}, {_number(upper)}] |"
            )
    lines.append("")


def summarize_results(config: dict[str, Any]) -> Path:
    root = output_dir(config)
    intervention = _read_json(root / "results" / "intervention_check.json")
    lines = [
        "# Controlled-memory hallucination probe results",
        "",
        f"Pipeline schema: {intervention.get('pipeline_schema_version', 'unknown')}; "
        f"code revision: `{intervention.get('code_revision', 'unknown')}`.",
        "",
        "## Intervention validity",
        "",
        f"- Exposed accuracy: {_number(intervention['accuracy_exposed'])}",
        f"- Withheld accuracy: {_number(intervention['accuracy_withheld'])}",
        f"- Paired memory gap: {_number(intervention['memory_gap'])}",
        "- Memory-gap 95% CI: "
        f"[{_number(intervention['memory_gap_ci95'][0])}, "
        f"{_number(intervention['memory_gap_ci95'][1])}]",
        f"- Required gap: {_number(intervention['minimum_required_gap'])}",
        f"- Cluster-bootstrap entities: {intervention.get('num_paired_entities', 'NA')}",
        f"- Gate passed: **{bool(intervention['passed'])}**",
        "",
    ]

    for benchmark in config["benchmarks"]["datasets"]:
        metrics = _read_json(root / "results" / benchmark / "metrics.json")
        lines.extend(
            [
                f"## {benchmark}",
                "",
                f"Scored {metrics['num_scored']}/{metrics['num_total']} examples "
                f"(grading coverage {_number(metrics['grading_coverage'])}); "
                f"abstention rate {_number(metrics['abstention_rate'])}.",
                "",
                "### Token-probability baselines",
                "",
                "Token baseline Brier/ECE values are descriptive only because these raw scores "
                "were not calibrated on an independent split.",
                "",
                "| Model | AUROC | AUPRC | Brier | ECE | AURC |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label, values in metrics["baselines"].items():
            lines.append(_metric_row(label, values))

        lines.extend(
            [
                "",
                "### Probe performance across seeds",
                "",
                "| Feature mode / variant | AUROC | AUPRC | Brier | ECE | AURC |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label, values in metrics["seed_aggregate"].items():
            lines.append(_metric_row(label, values))

        lines.append("")
        _append_comparison(
            lines,
            "Genuine exposure minus correctness-only",
            metrics["genuine_vs_correctness_bootstrap"],
            metrics["genuine_vs_correctness_seed_ensemble_bootstrap"],
            list(config["collection"]["feature_modes"]),
        )
        _append_comparison(
            lines,
            "Genuine exposure minus shuffled exposure",
            metrics["genuine_vs_shuffled_bootstrap"],
            metrics["genuine_vs_shuffled_seed_ensemble_bootstrap"],
            list(config["collection"]["feature_modes"]),
        )

    target = root / "results" / "analysis_summary.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
