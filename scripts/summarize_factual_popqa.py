"""Combine every planned frozen PopQA evaluation; do not pool seeds as independent questions."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peft_probe.within_question import _json, _sha


def summarize(config_path):
    settings = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    plan = json.loads(Path(settings["study_manifest"]).read_text(encoding="utf-8"))
    root = Path(settings["output_dir"])
    output = root / "summary"
    (output / "_SUCCESS.json").unlink(missing_ok=True)
    primary = "-17/correct_a/delta/fraction_8"
    names = [
        primary,
        primary + "/a_to_b",
        primary.replace("/correct_a/", "/incorrect/"),
        "answer_length_words",
        "source_length_probe",
        "source_length_plus_shift_probe",
        "-17/supervised_probe",
    ]
    reports, hashes, shared = [], {}, None
    for run in plan["runs"]:
        if _sha(Path(run["config"])) != run["config_sha256"]:
            raise ValueError("Original planned source config changed")
        directory = root / f"seed_{run['seed']}" / "results"
        marker = json.loads((directory / "_SUCCESS.json").read_text())
        path = directory / "metrics.json"
        report = json.loads(path.read_text())
        signature = report["signature"]
        if (
            not marker["complete"]
            or marker["signature"] != signature
            or report["primary"] != primary
            or signature["common"]["settings"] != settings
            or signature["frozen"]["source_signature"]["config_fingerprint"]
            != run["config_fingerprint"]
        ):
            raise ValueError(f"Seed {run['seed']} does not match the frozen external plan")
        pool = (signature["common"], signature["rows_sha256"])
        if shared is not None and shared != pool:
            raise ValueError("Seeds must use exactly the same saved PopQA question/answer pool")
        shared = pool
        hashes[str(path)] = _sha(path)
        reports.append(report)
    if len(reports) < 3 or len({r["seed"] for r in reports}) != len(reports):
        raise ValueError(
            "Every planned seed must be completed, including at least three distinct seeds"
        )
    lines = [
        "# Frozen PopQA evaluation across confirmation seeds",
        "",
        "Generated-answer ranking; directions, adapters and signs remain frozen from TriviaQA.",
        "The shared base-generated candidate pool is identical across seeds. All planned seeds are included.",
        "Per-run 95% intervals bootstrap subject groups; seed mean/SD are descriptive, without a pooled seed-question CI.",
        "",
        "| Seed | Mixed / total questions | Method | Within-question ranking | 95% CI |",
        "|---|---:|---|---:|---|",
    ]
    values = {name: [] for name in names}

    def number(value):
        return "unavailable" if value is None else f"{value:.4f}"

    for report in reports:
        dataset = report["popqa"]
        for name in names:
            item = dataset["methods"][name]
            value = item["within_question_pairwise_accuracy"]
            values[name].append(value)
            lines.append(
                f"| {report['seed']} | {dataset['n_mixed_questions']}/{dataset['n_questions']} | {name} | "
                f"{number(value)} | {item['within_question_ci95']} |"
            )
    lines.extend(["", "| Method | Mean ranking across seeds | Sample SD |", "|---|---:|---:|"])
    descriptive = []
    for name, items in values.items():
        valid = all(v is not None for v in items)
        mean, sd = (statistics.mean(items), statistics.stdev(items)) if valid else (None, None)
        descriptive.append({"method": name, "mean": mean, "seed_sd": sd})
        lines.append(f"| {name} | {number(mean)} | {number(sd)} |")
    lines.extend(["", "| Seed | Paired difference | Difference | 95% CI |", "|---|---|---:|---|"])
    for report in reports:
        for item in report["popqa"]["paired_comparisons"]:
            lines.append(
                f"| {report['seed']} | {item['left']} minus {item['right']} | {number(item['mean'])} | {item['ci95']} |"
            )
        item = report["source_test_length_diagnostic"]["paired_comparisons"][0]
        lines.append(
            f"| {report['seed']} | Source test: length-plus-shift minus length-only | {number(item['mean'])} | {item['ci95']} |"
        )
    lines.extend(
        [
            "",
            "The two length probes use extra labels from original probe_train only, fixed quadratic log-length terms and fixed regularization.",
            "No PopQA labels fit, select or orient methods. This is an incremental-prediction diagnostic, not proof of causal truth or complete length mediation.",
            "Retain all TriviaQA controls and the negative TruthfulQA reference-ranking results in the final conclusion.",
            "Small mixed-question/subject-group counts limit inference; unavailable metrics remain unavailable.",
        ]
    )
    output.mkdir(exist_ok=True)
    signature = {"settings_sha256": _sha(Path(config_path)), "reports_sha256": hashes}
    _json(
        output / "metrics.json",
        {"signature": signature, "descriptive": descriptive, "reports": reports},
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json(output / "_SUCCESS.json", {"complete": True, "signature": signature})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/factual_popqa.yaml"))
    print(summarize(parser.parse_args().config))
