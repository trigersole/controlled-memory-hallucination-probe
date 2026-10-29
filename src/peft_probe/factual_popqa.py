"""Frozen-adapter PopQA evaluation and source-trained length-adjustment controls."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import factual_shift as fs
from .base_contextual_transfer import _grade_popqa, _parse_aliases
from .config import config_fingerprint, load_config
from .factual_runtime import stage
from .io_utils import (
    atomic_json,
    atomic_jsonl,
    atomic_torch_save,
    chunks,
    ensure_manifest,
    read_jsonl,
    seed_everything,
)
from .within_question import _sha, evaluate_within_question

PRIMARY = "-17/correct_a/delta/fraction_8"
LENGTH_ONLY = "source_length_probe"
LENGTH_SHIFT = "source_length_plus_shift_probe"


def settings_path():
    return Path(os.environ.get("FACTUAL_POPQA_CONFIG", "configs/factual_popqa.yaml"))


def load_settings():
    settings = yaml.safe_load(settings_path().read_text(encoding="utf-8"))
    if settings["schema_version"] != 1 or settings["dataset"] != "akariasai/PopQA":
        raise ValueError("Expected PopQA evaluation schema 1")
    for key in ("max_questions", "bootstrap_samples"):
        if int(settings[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if float(settings["length_probe_C"]) <= 0:
        raise ValueError("length_probe_C must be positive")
    return settings


def stage_dir(path, signature):
    ensure_manifest(path, fs._digest(signature))
    return path


def choose_questions(items, blocked_questions, maximum, seed):
    """Sampling uses dataset fields only, never model scores or correctness outcomes."""
    rows, seen, rejected = (
        [],
        set(),
        {"source_question_overlap": 0, "duplicate_question": 0, "empty": 0},
    )
    for index in np.random.default_rng(seed).permutation(len(items)):
        item = items[int(index)]
        question = str(item["question"]).strip()
        aliases = list(dict.fromkeys([str(item["obj"]), *_parse_aliases(item["possible_answers"])]))
        aliases = [answer for answer in aliases if fs._normal(answer)]
        key = fs._normal(question)
        if not key or not aliases:
            rejected["empty"] += 1
            continue
        if key in blocked_questions:
            rejected["source_question_overlap"] += 1
            continue
        if key in seen:
            rejected["duplicate_question"] += 1
            continue
        seen.add(key)
        item_id = str(item.get("id", index))
        rows.append(
            {
                "fact_id": f"popqa_{item_id}",
                "question": question,
                "answer": aliases[0],
                "aliases": aliases,
                "split": "popqa",
                "benchmark": "popqa",
                "group_id": f"subject_{item.get('subj_id', item.get('subj', item_id))}",
                "subject": str(item.get("subj", "")),
                "relation": str(item.get("prop", "")),
                "subject_popularity": int(item.get("s_pop", 0)),
            }
        )
        if len(rows) == maximum:
            break
    if not rows or len({r["fact_id"] for r in rows}) != len(rows):
        raise ValueError("No usable unique PopQA questions, or duplicate dataset IDs")
    return rows, rejected


def source_inputs(config, settings):
    plan = json.loads(Path(settings["study_manifest"]).read_text(encoding="utf-8"))
    blocked, sources, matching = set(), [], False
    for run in plan["runs"]:
        path = Path(run["config"])
        if _sha(path) != run["config_sha256"]:
            raise ValueError(f"Study config changed: {path}")
        other = load_config(path)
        if other["model"] != config["model"]:
            raise ValueError("All frozen seeds must use the same base model/loading configuration")
        if config_fingerprint(other) == config_fingerprint(config):
            matching = True
        data = fs.root(other) / "data/rows.jsonl"
        blocked.update(fs._normal(r["question"]) for r in read_jsonl(data))
        sources.append({"seed": other["experiment"]["seed"], "questions_sha256": _sha(data)})
    if not matching:
        raise ValueError("Source config is not a member of the completed confirmation study")
    shared_signature = {
        "settings": settings,
        "model": config["model"],
        "generation": {
            k: config["factual_shift"][k]
            for k in (
                "samples_per_question",
                "max_new_tokens",
                "temperature",
                "top_p",
                "batch_size",
                "shard_size",
                "representation",
                "layers",
            )
        },
        "source_data": sources,
    }
    results = fs.root(config) / "results"
    fs._require(results)
    cached = torch.load(results / "analysis_fit.pt", map_location="cpu", weights_only=False)
    original_rows = fs._replay_rows(config)
    metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    signature = {
        "checkpoint_version": 1,
        "config_fingerprint": config_fingerprint(config),
        "rows_sha256": fs._digest(original_rows),
    }
    if cached["signature"] != signature or metrics["rows_sha256"] != signature["rows_sha256"]:
        raise ValueError(
            "Original frozen score checkpoint does not match source corpus/configuration"
        )
    artifacts = cached["fitted"]["artifacts"]
    if (
        artifacts["selected"] != PRIMARY
        or artifacts.get("selection_rule") != "predeclared_method_validation_sign"
    ):
        raise ValueError("Use a completed confirmation seed with the fixed primary method")
    for name, sign in artifacts["signs"].items():
        if name in metrics["methods"] and metrics["methods"][name]["orientation"] != sign:
            raise ValueError(f"Saved source signs disagree: {name}")
    adapter_hashes = {}
    for source in fs.ADAPTERS:
        directory = fs.root(config) / "adapters" / source / "final"
        fs._require(directory)
        files = sorted(
            p for p in directory.iterdir() if p.suffix in {".json", ".safetensors", ".bin"}
        )
        if not any(p.suffix in {".safetensors", ".bin"} for p in files):
            raise FileNotFoundError(f"Missing frozen adapter weights: {directory}")
        adapter_hashes[source] = {p.name: _sha(p) for p in files}
    frozen_signature = {
        "source_signature": signature,
        "score_checkpoint_sha256": _sha(results / "analysis_fit.pt"),
        "metrics_sha256": _sha(results / "metrics.json"),
        "adapters": adapter_hashes,
    }
    return blocked, shared_signature, frozen_signature, cached["fitted"], original_rows


def prepare_data(shared, config, settings, signature, blocked):
    target = stage_dir(shared / "data", signature)
    if not (target / "_SUCCESS.json").exists():
        from datasets import load_dataset

        dataset = load_dataset(
            settings["dataset"], split=settings["split"], revision=settings["dataset_revision"]
        )
        rows, exclusions = choose_questions(
            dataset, blocked, int(settings["max_questions"]), int(settings["sampling_seed"])
        )
        atomic_jsonl(target / "rows.jsonl", rows)
        fs._complete(
            target,
            {
                "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
                "requested_revision": settings["dataset_revision"],
                "requested_questions": settings["max_questions"],
                "n_questions": len(rows),
                "exclusions_before_budget": exclusions,
                "rows_sha256": fs._digest(rows),
            },
        )
    rows = read_jsonl(target / "rows.jsonl")
    marker = json.loads((target / "_SUCCESS.json").read_text())
    if marker["rows_sha256"] != fs._digest(rows):
        raise ValueError("Saved PopQA question snapshot changed")
    return rows


def candidates(shared, config, settings, signature, questions):
    target = stage_dir(
        shared / "candidates", {**signature, "questions_sha256": fs._digest(questions)}
    )
    generation = {
        **config["factual_shift"],
        "max_sequence_length": config["model"]["max_sequence_length"],
    }
    if not (target / "_SUCCESS.json").exists():
        pending = [
            (i, group)
            for i, group in chunks(questions, int(generation["shard_size"]))
            if not (target / f"shard_{i:05d}.jsonl").exists()
        ]
        if pending:
            tokenizer = fs.load_tokenizer(config["model"])
            model = fs.load_base_model(config["model"]).eval()
            for index, group in pending:
                seed_everything(int(settings["sampling_seed"]) + index)
                print(f"[popqa candidates] shard {index}", flush=True)
                records = []
                for _, batch in chunks(group, int(generation["batch_size"])):
                    for sample in range(int(generation["samples_per_question"])):
                        answers = fs._generate(
                            model, tokenizer, batch, generation, sample=sample > 0
                        )
                        for row, answer in zip(batch, answers, strict=True):
                            correct, status = _grade_popqa(answer, row["aliases"])
                            records.append(
                                {
                                    **row,
                                    "answer": answer,
                                    "correct": correct,
                                    "grade_status": status,
                                    "candidate_index": sample,
                                    "row_id": f"{row['fact_id']}:{sample}",
                                }
                            )
                atomic_jsonl(target / f"shard_{index:05d}.jsonl", records)
            del model
            fs._release()
    rows = [r for p in sorted(target.glob("shard_*.jsonl")) for r in read_jsonl(p)]
    expected = {
        f"{r['fact_id']}:{i}"
        for r in questions
        for i in range(int(generation["samples_per_question"]))
    }
    if len(rows) != len(expected) or {r["row_id"] for r in rows} != expected:
        raise ValueError("PopQA candidate shards do not cover the fixed question pool")
    lookup = {r["fact_id"]: r for r in questions}
    for row in rows:
        original = lookup[row["fact_id"]]
        if row["row_id"] != f"{row['fact_id']}:{row['candidate_index']}" or any(
            row[key] != original[key] for key in ("question", "aliases", "group_id", "split")
        ):
            raise ValueError("Saved PopQA candidate metadata changed")
        if _grade_popqa(row["answer"], row["aliases"])[0] != row["correct"]:
            raise ValueError("Saved PopQA grading changed")
    digest = fs._digest(rows)
    marker = target / "_SUCCESS.json"
    if marker.exists() and json.loads(marker.read_text())["rows_sha256"] != digest:
        raise ValueError("Saved candidate snapshot changed")
    if not marker.exists():
        fs._complete(target, {"n_candidates": len(rows), "rows_sha256": digest})
    return rows


def replay_source(target, config, source, rows, signature):
    stage_dir(target, {"inputs": signature, "source": source, "rows_sha256": fs._digest(rows)})
    settings = {
        **config["factual_shift"],
        "max_sequence_length": config["model"]["max_sequence_length"],
    }
    digest = fs._digest(rows)
    groups = list(chunks(rows, int(settings["shard_size"])))
    pending = []
    for index, group in groups:
        path = target / f"shard_{index:05d}.pt"
        if path.exists():
            fs._read_feature_shard(path, group, digest, settings["layers"])
        else:
            pending.append((index, group))
    if pending:
        tokenizer = fs.load_tokenizer(config["model"])
        model = fs.load_base_model(config["model"])
        if source != "base":
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, fs.root(config) / "adapters" / source / "final", is_trainable=False
            )
        model.eval()
        for index, group in pending:
            print(f"[popqa replay {source}] shard {index}", flush=True)
            parts, likelihood = {str(layer): [] for layer in settings["layers"]}, []
            for _, batch in chunks(group, int(settings["batch_size"])):
                features, scores = fs.replay(model, tokenizer, batch, settings)
                likelihood.extend(scores)
                for layer, value in features.items():
                    parts[layer].append(value)
            atomic_torch_save(
                target / f"shard_{index:05d}.pt",
                {
                    "row_ids": [r["row_id"] for r in group],
                    "rows_sha256": digest,
                    "features": {key: torch.cat(value) for key, value in parts.items()},
                    "mean_logprob": likelihood,
                },
            )
        del model
        fs._release()
    matrices, likelihood = {str(layer): [] for layer in settings["layers"]}, []
    for index, group in groups:
        saved = fs._read_feature_shard(
            target / f"shard_{index:05d}.pt", group, digest, settings["layers"]
        )
        likelihood.extend(saved["mean_logprob"])
        for layer, parts in matrices.items():
            parts.append(saved["features"][layer].numpy())
    if not (target / "_SUCCESS.json").exists():
        fs._complete(target, {"rows_sha256": digest})
    return {key: np.concatenate(value) for key, value in matrices.items()}, np.asarray(likelihood)


def frozen_scores(config, artifacts, features, likelihood, rows):
    primary = artifacts["selected"]
    layer = primary.split("/")[0]
    methods = list(
        dict.fromkeys(
            [
                primary,
                primary + "/a_to_b",
                primary.replace("/correct_a/", "/incorrect/"),
                artifacts["selected_by_family"]["incorrect"],
                "base_likelihood",
                "answer_length_words",
                artifacts["selected_by_family"]["activation_pca"],
                f"{layer}/supervised_probe",
                f"{layer}/random/base",
            ]
        )
    )
    result = {}
    base = features["base"]
    for name in methods:
        native = primary if name.endswith("/a_to_b") else name
        if native == "base_likelihood":
            raw = likelihood
        elif native == "answer_length_words":
            raw = np.asarray([len(row["answer"].split()) for row in rows], dtype=float)
        elif native.endswith("/supervised_probe"):
            raw = artifacts[native].predict_proba(base[native.split("/")[0]])[:, 1]
        elif native.endswith("/random/base"):
            key = native.split("/")[0]
            values = base[key] - artifacts[f"{key}/correct_a"]["origin"]
            direction = np.random.default_rng(int(config["experiment"]["seed"])).normal(
                size=values.shape[1]
            )
            direction /= np.linalg.norm(direction)
            raw = values @ direction
        elif "/activation_pca/" in native:
            key, _, statistic = native.split("/")
            values = base[key] - artifacts[f"{key}/correct_a"]["origin"]
            raw = fs.direction_scores(
                values, artifacts[f"{key}/activation_pca"], config["factual_shift"]["ranks"]
            )[statistic]
        else:
            key, source, view, statistic = native.split("/")
            fit = artifacts[f"{key}/{source}"]
            applied = "correct_b" if name.endswith("/a_to_b") else source
            values = {
                "base": base[key] - fit["origin"],
                "on_policy": features[applied][key] - fit["origin"],
                "delta": features[applied][key] - base[key],
            }[view]
            raw = fs.direction_scores(values, fit, config["factual_shift"]["ranks"])[statistic]
        sign = artifacts["signs"][native]
        if sign not in (-1, 1):
            raise ValueError(f"Invalid frozen orientation: {native}")
        result[name] = sign * np.asarray(raw, dtype=float)
        if result[name].shape != (len(rows),) or not np.isfinite(result[name]).all():
            raise ValueError(f"Invalid target score: {name}")
    return result


def length_design(rows, score, joint):
    length = np.log1p([len(r["answer"].split()) for r in rows])
    return np.column_stack([length, length**2, score] if joint else [length, length**2])


def length_heads(config, fitted, rows, settings, primary=PRIMARY):
    score = fitted["artifacts"]["signs"][primary] * fitted["scores"][primary]
    train = np.asarray([r["split"] == "probe_train" and r["correct"] in (0, 1) for r in rows])
    labels = np.asarray([r["correct"] for r in rows])
    if len(np.unique(labels[train])) != 2:
        raise ValueError("Source probe_train needs both correctness classes for length controls")
    heads = {}
    for name, joint in ((LENGTH_ONLY, False), (LENGTH_SHIFT, True)):
        head = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(settings["length_probe_C"]),
                max_iter=1000,
                random_state=int(config["experiment"]["seed"]),
            ),
        )
        head.fit(length_design(rows, score, joint)[train], labels[train])
        heads[name] = head
    return heads


def add_length_scores(scores, rows, heads, primary=PRIMARY):
    for name, joint in ((LENGTH_ONLY, False), (LENGTH_SHIFT, True)):
        scores[name] = heads[name].predict_proba(length_design(rows, scores[primary], joint))[:, 1]


def analyze(target, config, settings, signature, rows, scores, heads, fitted, original_rows):
    stage_dir(target, signature)
    methods = list(scores)
    comparisons = [
        (PRIMARY + "/a_to_b", PRIMARY),
        (PRIMARY, PRIMARY.replace("/correct_a/", "/incorrect/")),
        (PRIMARY, "answer_length_words"),
        (PRIMARY, "base_likelihood"),
        (LENGTH_SHIFT, LENGTH_ONLY),
    ]
    scored = [
        {**row, "scores": {m: float(scores[m][i]) for m in methods}} for i, row in enumerate(rows)
    ]
    report, per_question = evaluate_within_question(
        scored,
        methods,
        int(settings["bootstrap_samples"]),
        int(settings["sampling_seed"]),
        comparisons,
    )
    source_score = fitted["artifacts"]["signs"][PRIMARY] * fitted["scores"][PRIMARY]
    source_scores = {PRIMARY: np.asarray(source_score, dtype=float)}
    add_length_scores(source_scores, original_rows, heads)
    source_test = [
        {**row, "scores": {m: float(values[i]) for m, values in source_scores.items()}}
        for i, row in enumerate(original_rows)
        if row["split"] == "test"
    ]
    source_report, _ = evaluate_within_question(
        source_test,
        list(source_scores),
        int(settings["bootstrap_samples"]),
        int(settings["sampling_seed"]),
        [(LENGTH_SHIFT, LENGTH_ONLY)],
    )
    atomic_torch_save(target / "source_length_heads.pt", {"signature": signature, "heads": heads})
    atomic_jsonl(target / "predictions.jsonl", scored)
    atomic_json(
        target / "metrics.json",
        {
            "signature": signature,
            "seed": config["experiment"]["seed"],
            "primary": PRIMARY,
            "popqa": report,
            "source_test_length_diagnostic": source_report,
        },
    )
    atomic_json(target / "per_question.json", per_question)
    lines = [
        "# Frozen factual LoRA transfer to PopQA",
        "",
        f"Source seed: {config['experiment']['seed']}; frozen primary: `{PRIMARY}`.",
        "Generated-answer ranking; source directions, origins, probes and signs are unchanged.",
        "Questions have equal weight; 95% intervals bootstrap subject groups; ties receive half credit.",
        "All three seeds share the same saved question/answer pool. No PopQA labels fit or select a method.",
        "",
        f"Mixed questions: {report['n_mixed_questions']}/{report['n_questions']}; mixed subject groups: {report['n_mixed_clusters']}.",
        "",
        "| Method | Pooled AUROC | Within-question ranking | 95% CI |",
        "|---|---:|---:|---|",
    ]

    def number(value):
        return "unavailable" if value is None else f"{value:.4f}"

    for name, item in report["methods"].items():
        lines.append(
            f"| {name} | {number(item['pooled_auroc'])} | {number(item['within_question_pairwise_accuracy'])} | {item['within_question_ci95']} |"
        )
    lines.extend(
        ["", "| Paired comparison (left minus right) | Difference | 95% CI |", "|---|---:|---|"]
    )
    for item in report["paired_comparisons"]:
        lines.append(
            f"| {item['left']} minus {item['right']} | {number(item['mean'])} | {item['ci95']} |"
        )
    difference = source_report["paired_comparisons"][0]
    lines.extend(
        [
            "",
            f"Source test: length-plus-shift minus length-only: {number(difference['mean'])}; CI {difference['ci95']}.",
            "",
            "Length heads are secondary diagnostics trained only on original probe_train labels (extra supervision).",
            "Both use fixed quadratic log-word-length terms; the joint model additionally uses the frozen A score.",
            "No tuning, sign flipping, or fitting uses PopQA correctness. Lack of incremental linear prediction does not prove length fully explains the signal.",
            "Alias containment is imperfect; shared subjects are clustered, but semantic/pretraining overlap is not ruled out.",
            "Conditional mixed-question ranking is not all-question answer-selection accuracy. Keep the original TruthfulQA failure in the conclusion.",
        ]
    )
    (target / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fs._complete(target, {"signature": signature})
    return target


def run(config):
    settings = load_settings()
    blocked, common, frozen, fitted, original_rows = stage(
        "popqa/source_inputs", source_inputs, config, settings
    )
    root = Path(settings["output_dir"]).resolve()
    local = root / f"seed_{config['experiment']['seed']}"
    shared = root / "shared"
    if (
        root == fs.root(config)
        or root.is_relative_to(fs.root(config))
        or fs.root(config).is_relative_to(root)
    ):
        raise ValueError("External output directory must be separate from the source experiment")
    questions = stage("popqa/data", prepare_data, shared, config, settings, common, blocked)
    rows = stage("popqa/candidates", candidates, shared, config, settings, common, questions)
    inputs = {"common": common, "frozen": frozen, "rows_sha256": fs._digest(rows)}
    result = local / "results"
    stage_dir(result, inputs)
    if (result / "_SUCCESS.json").exists():
        return result
    features, base_likelihood = {}, None
    for source in fs.SOURCES:
        target = shared / "features/base" if source == "base" else local / "features" / source
        signature = common if source == "base" else inputs
        matrix, likelihood = stage(
            f"popqa/features/{source}", replay_source, target, config, source, rows, signature
        )
        features[source] = matrix
        if source == "base":
            base_likelihood = likelihood
    scores = frozen_scores(config, fitted["artifacts"], features, base_likelihood, rows)
    heads = length_heads(config, fitted, original_rows, settings)
    add_length_scores(scores, rows, heads)
    return stage(
        "popqa/analysis",
        analyze,
        result,
        config,
        settings,
        inputs,
        rows,
        scores,
        heads,
        fitted,
        original_rows,
    )
