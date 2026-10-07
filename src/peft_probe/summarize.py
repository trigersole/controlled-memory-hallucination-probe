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


def summarize_results(config: dict[str, Any]) -> Path:
    root = output_dir(config)
    intervention = _read_json(root / "results" / "intervention_check.json")
    lines = [
        "# Controlled-memory hallucination probe results",
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

        lines.extend(
            [
                "",
                "### Genuine-exposure minus correctness-only bootstrap comparison",
                "",
                "Positive deltas favor genuine exposure for AUROC/AUPRC; negative deltas favor it "
                "for Brier/ECE/AURC. `Supported seeds` counts 95% CIs excluding zero in the "
                "favorable direction.",
                "",
                "| Feature mode | Metric | Mean delta | Supported seeds |",
                "|---|---|---:|---:|",
            ]
        )
        comparisons = metrics["genuine_vs_correctness_bootstrap"]
        for mode in config["collection"]["feature_modes"]:
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
                lines.append(
                    f"| {mode} | {metric} | {_number(fmean(deltas))} | "
                    f"{supported}/{len(mode_runs)} |"
                )
        lines.append("")

    target = root / "results" / "analysis_summary.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
