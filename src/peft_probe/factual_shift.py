"""Isolated TriviaQA experiment: does the largest LoRA activation change predict truth?

All models replay the same saved text. SVD uses only direction-fit questions;
orientations and method selection use validation labels, never test/OOD labels.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import string
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import config_fingerprint, output_dir
from .io_utils import (
    atomic_json,
    atomic_jsonl,
    atomic_torch_save,
    chunks,
    ensure_manifest,
    mark_complete,
    read_jsonl,
    seed_everything,
)
from .modeling import load_base_model, load_tokenizer, model_device, user_prompt

SCHEMA = 1
ADAPTERS = ("correct_a", "correct_b", "incorrect")
SOURCES = ("base", *ADAPTERS)
SPLITS = ("adapter_a", "adapter_b", "direction_fit", "probe_train", "validation", "test")


def validate(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["factual_shift"]
    if settings.get("schema_version") != SCHEMA:
        raise ValueError(f"Expected factual_shift.schema_version={SCHEMA}")
    for key in (
        "batch_size",
        "shard_size",
        "max_new_tokens",
        "samples_per_question",
        "negative_attempts",
        "bootstrap_samples",
    ):
        if int(settings[key]) <= 0:
            raise ValueError(f"factual_shift.{key} must be positive")
    if set(settings["split_sizes"]) != set(SPLITS):
        raise ValueError(f"split_sizes must contain exactly {SPLITS}")
    if any(int(n) <= 0 for n in settings["split_sizes"].values()):
        raise ValueError("All split sizes must be positive")
    layers = settings["layers"]
    if not layers or any(type(x) is not int for x in layers) or len(set(layers)) != len(layers):
        raise ValueError("layers must be distinct integer indices")
    ranks = settings["ranks"]
    if not ranks or any(type(k) is not int or k <= 0 for k in ranks):
        raise ValueError("ranks must be positive integers")
    if settings["representation"] not in {"answer_last", "answer_mean"}:
        raise ValueError("representation must be answer_last or answer_mean")
    if not 0 < float(settings["temperature"]) or not 0 < float(settings["top_p"]) <= 1:
        raise ValueError("Invalid sampling temperature/top_p")
    if not settings["coverages"] or any(not 0 < float(c) <= 1 for c in settings["coverages"]):
        raise ValueError("coverages must be in (0, 1]")
    if int(settings.get("ood_max_examples", 0)) < 0:
        raise ValueError("ood_max_examples must be nonnegative")
    if not 0 < float(settings["minimum_negative_fraction"]) <= 1:
        raise ValueError("minimum_negative_fraction must be in (0, 1]")
    if int(settings["split_sizes"]["direction_fit"]) < max(ranks):
        raise ValueError("direction_fit size must be at least the largest requested rank")
    return settings


def root(config: dict[str, Any]) -> Path:
    return output_dir(config) / f"factual_shift_v{SCHEMA}"


def _stage(config: dict[str, Any], name: str, force: bool = False) -> Path:
    validate(config)
    target = root(config) / name
    dependencies = []
    if name != "data":
        dependencies.append(root(config) / "data")
    if name.startswith("features/") or name == "results":
        dependencies.append(root(config) / "candidates")
    if name.startswith("features/") and name.split("/")[-1] != "base":
        dependencies.append(root(config) / "adapters" / name.split("/")[-1] / "final")
    if name == "results":
        dependencies.extend(root(config) / "features" / source for source in SOURCES)
    signatures = []
    for dependency in dependencies:
        _require(dependency)
        signatures.append(
            (dependency.name, (dependency / "_SUCCESS.json").read_text(encoding="utf-8"))
        )
    fingerprint = hashlib.sha256(
        json.dumps([config_fingerprint(config), signatures]).encode()
    ).hexdigest()
    # Changed prerequisites require an explicit rebuild of every downstream stage.
    ensure_manifest(target, fingerprint, force=force)
    if force:
        (target / "_SUCCESS.json").unlink(missing_ok=True)
    return target


def _complete(target: Path, metadata: dict) -> None:
    mark_complete(target / "_SUCCESS.json", {"artifact_id": uuid.uuid4().hex, **metadata})


def _require(directory: Path) -> None:
    if not (directory / "_SUCCESS.json").is_file():
        raise FileNotFoundError(f"Incomplete prerequisite: {directory}")


def _normal(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def _digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def deduplicate_questions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse evidence repeats before splitting, preserving accepted aliases."""
    unique: dict[str, dict[str, Any]] = {}
    for row in items:
        fact_id = row["fact_id"]
        if fact_id not in unique:
            unique[fact_id] = {**row, "aliases": list(row["aliases"])}
            continue
        previous = unique[fact_id]
        if _normal(previous["question"]) != _normal(row["question"]) or _normal(
            previous["answer"]
        ) != _normal(row["answer"]):
            raise ValueError(f"Conflicting TriviaQA records for question ID {fact_id!r}")
        previous["aliases"] = list(dict.fromkeys([*previous["aliases"], *row["aliases"]]))
    return list(unique.values())


def split_rows(items: list[dict[str, Any]], sizes: dict[str, int], seed: int) -> list[dict]:
    """Keep duplicate questions and transitively overlapping answer aliases together.

    A group is assigned once; excess members are discarded to honor exact budgets.
    This guards observable duplicates, not every possible semantic paraphrase.
    """
    parents = list(range(len(items)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i, row in enumerate(items):
        keys = [("q", _normal(row["question"]))]
        keys += [("a", _normal(a)) for a in row["aliases"] if _normal(a)]
        for key in keys:
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)
    grouped = list(groups.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(grouped)
    result = []
    remaining = {key: int(sizes[key]) for key in SPLITS}
    for group in grouped:
        available = [key for key in SPLITS if remaining[key] > 0]
        if not available:
            break
        # Fill in a fixed order, with randomized group order independent of labels.
        split = available[0]
        for i in group[: remaining[split]]:
            result.append({**items[i], "split": split, "group_id": f"group_{find(i)}"})
            remaining[split] -= 1
    if any(remaining.values()):
        raise ValueError(f"Not enough independent TriviaQA groups for requested sizes: {remaining}")
    return result


def prepare(config: dict[str, Any], force: bool = False) -> Path:
    from datasets import load_dataset

    target = _stage(config, "data", force)
    if (target / "_SUCCESS.json").exists() and not force:
        return target
    settings = validate(config)
    dataset = load_dataset(
        settings["dataset"], settings["dataset_config"], split=settings["dataset_split"]
    )
    items = []
    for i, row in enumerate(dataset):
        aliases = list(dict.fromkeys([row["answer"]["value"], *row["answer"].get("aliases", [])]))
        if not row["question"].strip() or not all(isinstance(a, str) for a in aliases):
            raise ValueError(f"Invalid TriviaQA item {i}")
        if not _normal(aliases[0]):
            continue
        items.append(
            {
                "fact_id": str(row.get("question_id") or f"trivia_{i}"),
                "question": row["question"],
                "answer": aliases[0],
                "aliases": aliases,
                "benchmark": "trivia_qa",
            }
        )
    input_rows = len(items)
    items = deduplicate_questions(items)
    rows = split_rows(items, settings["split_sizes"], int(config["experiment"]["seed"]))
    if int(settings.get("ood_max_examples", 0)):
        questions = {_normal(r["question"]) for r in rows}
        answers = {_normal(a) for r in rows for a in r["aliases"] if _normal(a)}
        ood = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        order = np.random.default_rng(int(config["experiment"]["seed"])).permutation(len(ood))
        count = 0
        for i in order:
            row = ood[int(i)]
            correct = list(dict.fromkeys([row["best_answer"], *row["correct_answers"]]))
            if _normal(row["question"]) in questions or any(_normal(a) in answers for a in correct):
                continue
            rows.append(
                {
                    "fact_id": f"truthful_{i}",
                    "question": row["question"],
                    "answer": row["best_answer"],
                    "aliases": correct,
                    "incorrect_answers": row["incorrect_answers"],
                    "split": "ood",
                    "group_id": f"truthful_{i}",
                    "benchmark": "truthful_qa",
                }
            )
            count += 1
            if count >= int(settings["ood_max_examples"]):
                break
        if not count:
            raise ValueError("No eligible OOD examples")
    atomic_jsonl(target / "rows.jsonl", rows)
    _complete(
        target,
        {
            "rows_sha256": _digest(rows),
            "num_rows": len(rows),
            "trivia_input_rows": input_rows,
            "trivia_unique_questions": len(items),
            "trivia_duplicate_rows_collapsed": input_rows - len(items),
        },
    )
    return target


def _read_data(config) -> list[dict]:
    target = root(config) / "data"
    _require(target)
    return read_jsonl(target / "rows.jsonl")


def _prompt(question: str) -> str:
    return "Answer with only the short answer. " + question


@torch.inference_mode()
def _generate(model, tokenizer, rows, settings, sample: bool) -> list[str]:
    tokens = tokenizer(
        [user_prompt(tokenizer, _prompt(r["question"])) for r in rows],
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    if (
        tokens["input_ids"].shape[1] + int(settings["max_new_tokens"])
        > settings["max_sequence_length"]
    ):
        raise ValueError("Prompt plus generation exceeds max_sequence_length; increase it")
    tokens = {k: v.to(model_device(model)) for k, v in tokens.items()}
    kwargs = (
        {"temperature": float(settings["temperature"]), "top_p": float(settings["top_p"])}
        if sample
        else {}
    )
    output = model.generate(
        **tokens,
        max_new_tokens=int(settings["max_new_tokens"]),
        do_sample=sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **kwargs,
    )
    return [
        a.strip()
        for a in tokenizer.batch_decode(
            output[:, tokens["input_ids"].shape[1] :], skip_special_tokens=True
        )
    ]


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_negatives(config: dict, force: bool = False) -> Path:
    """Save model-generated wrong-answer candidates; never silently use donor answers."""
    target = _stage(config, "negatives", force)
    if (target / "_SUCCESS.json").exists() and not force:
        return target
    rows = [r for r in _read_data(config) if r["split"] == "adapter_a"]
    settings = {**validate(config), "max_sequence_length": config["model"]["max_sequence_length"]}
    tokenizer = load_tokenizer(config["model"])
    model = load_base_model(config["model"]).eval()
    for shard, batch in chunks(rows, int(settings["batch_size"])):
        path = target / f"shard_{shard:05d}.jsonl"
        if path.exists() and not force:
            continue
        seed_everything(int(config["experiment"]["seed"]) + shard + 1000)
        selected: dict[str, str] = {}
        for _ in range(int(settings["negative_attempts"])):
            missing = [r for r in batch if r["fact_id"] not in selected]
            if not missing:
                break
            answers = _generate(model, tokenizer, missing, settings, sample=True)
            for row, answer in zip(missing, answers, strict=True):
                # Conservative rejection also excludes answers containing a gold alias.
                normalized = _normal(answer)
                aliases = [_normal(a) for a in row["aliases"] if _normal(a)]
                abstained = any(
                    _normal(phrase) in normalized
                    for phrase in (
                        "i don't know",
                        "i do not know",
                        "unknown",
                        "cannot determine",
                        "can't determine",
                        "not enough information",
                        "no information",
                    )
                )
                if normalized and not abstained and not any(a in normalized for a in aliases):
                    selected[row["fact_id"]] = answer
        atomic_jsonl(
            path,
            [
                {
                    **r,
                    "negative_answer": selected.get(r["fact_id"]),
                    "negative_kind": "base_sample_not_matching_reference",
                }
                for r in batch
            ],
        )
    del model
    _release()
    negatives = [r for p in sorted(target.glob("shard_*.jsonl")) for r in read_jsonl(p)]
    expected = {r["fact_id"] for r in rows}
    if len(negatives) != len(rows) or {r["fact_id"] for r in negatives} != expected:
        raise ValueError("Negative-answer shards do not match prepared training questions")
    accepted = [r for r in negatives if r["negative_answer"]]
    # Use exactly this accepted subset for both A and its incorrect-answer control.
    if len(accepted) < max(
        2, int(np.ceil(len(rows) * float(settings["minimum_negative_fraction"])))
    ):
        raise ValueError("Too few negatives; increase negative_attempts or review saved candidates")
    atomic_jsonl(target / "rows.jsonl", accepted)
    _complete(
        target,
        {
            "accepted": len(accepted),
            "requested": len(rows),
            "warning": "Reference mismatch is a proxy for incorrectness; audit negatives.",
        },
    )
    return target


def train(config: dict, adapter: str, force: bool = False) -> Path:
    from .train_adapter import train_rows

    if adapter not in ADAPTERS:
        raise ValueError(f"Unknown adapter {adapter}")
    validate(config)
    if adapter == "correct_b":
        rows = [r for r in _read_data(config) if r["split"] == "adapter_b"]
    else:
        _require(root(config) / "negatives")
        rows = read_jsonl(root(config) / "negatives" / "rows.jsonl")
    examples = [
        {
            "train_questions": [_prompt(r["question"])],
            "value": r["negative_answer"] if adapter == "incorrect" else r["answer"],
        }
        for r in rows
    ]
    # Identical seed/order/hyperparameters for A and incorrect; B has independent initialization.
    seed = int(config["experiment"]["seed"]) + int(adapter == "correct_b")
    training_config = {**config, "factual_training_rows_sha256": _digest(examples)}
    return train_rows(
        training_config, examples, root(config) / "adapters" / adapter, seed, force, adapter
    )


def _grade(answer: str, row: dict) -> tuple[int, str]:
    if row["benchmark"] == "trivia_qa":
        return int(
            _normal(answer) in {_normal(a) for a in row["aliases"] if _normal(a)}
        ), "normalized_exact_match"
    from .benchmark import _grade_truthful

    return _grade_truthful(answer, row["aliases"], row["incorrect_answers"])


def collect_candidates(config: dict, force: bool = False) -> Path:
    target = _stage(config, "candidates", force)
    if (target / "_SUCCESS.json").exists() and not force:
        return target
    rows = [
        r
        for r in _read_data(config)
        if r["split"] not in {"adapter_a", "adapter_b", "direction_fit"}
    ]
    settings = {**validate(config), "max_sequence_length": config["model"]["max_sequence_length"]}
    tokenizer = load_tokenizer(config["model"])
    model = load_base_model(config["model"]).eval()
    for shard, batch in chunks(rows, int(settings["shard_size"])):
        path = target / f"shard_{shard:05d}.jsonl"
        if path.exists() and not force:
            continue
        seed_everything(int(config["experiment"]["seed"]) + shard + 2000)
        records = []
        for _, group in chunks(batch, int(settings["batch_size"])):
            for sample in range(int(settings["samples_per_question"])):
                answers = _generate(model, tokenizer, group, settings, sample=sample > 0)
                for row, answer in zip(group, answers, strict=True):
                    correct, status = _grade(answer, row)
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
        atomic_jsonl(path, records)
    del model
    _release()
    records = [r for p in sorted(target.glob("shard_*.jsonl")) for r in read_jsonl(p)]
    expected = {
        f"{r['fact_id']}:{i}" for r in rows for i in range(int(settings["samples_per_question"]))
    }
    if len(records) != len(expected) or {r["row_id"] for r in records} != expected:
        raise ValueError("Candidate shards do not match prepared questions")
    _complete(target, {"num_candidates": len(records), "rows_sha256": _digest(records)})
    return target


def _replay_rows(config: dict) -> list[dict]:
    _require(root(config) / "candidates")
    fit = [
        {**r, "row_id": f"{r['fact_id']}:gold", "correct": 1}
        for r in _read_data(config)
        if r["split"] == "direction_fit"
    ]
    candidates = [
        r
        for p in sorted((root(config) / "candidates").glob("shard_*.jsonl"))
        for r in read_jsonl(p)
    ]
    rows = fit + candidates
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate replay row IDs")
    return rows


@torch.inference_mode()
def replay(model, tokenizer, rows: list[dict], settings: dict) -> tuple[dict, list[float]]:
    """Use float32 storage, exact shared tokenization, no EOS in answer pooling."""
    sequences, starts, answer_lengths = [], [], []
    for row in rows:
        prompt = tokenizer(
            user_prompt(tokenizer, _prompt(row["question"])), add_special_tokens=False
        )["input_ids"]
        answer = (
            tokenizer(" " + row["answer"].strip(), add_special_tokens=False)["input_ids"]
            if row["answer"].strip()
            else []
        )
        if len(prompt) + len(answer) > settings["max_sequence_length"]:
            raise ValueError(f"Replay exceeds max_sequence_length: {row['row_id']}")
        sequences.append(prompt + answer)
        starts.append(len(prompt))
        answer_lengths.append(len(answer))
    width = max(map(len, sequences))
    ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    answer_mask = torch.zeros_like(ids, dtype=torch.bool)
    pool_mask = torch.zeros_like(answer_mask)
    for i, sequence in enumerate(sequences):
        padding = width - len(sequence)
        ids[i, padding:] = torch.tensor(sequence)
        attention[i, padding:] = 1
        answer_mask[i, padding + starts[i] :] = True
        pool_mask[i] = answer_mask[i]
        if not answer_lengths[i]:
            pool_mask[i, -1] = True  # Explicit prompt fallback for empty generation.
    device = model_device(model)
    ids, attention = ids.to(device), attention.to(device)
    answer_mask, pool_mask = answer_mask.to(device), pool_mask.to(device)
    output = model(
        input_ids=ids,
        attention_mask=attention,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    result = {}
    resolved = set()
    for layer in settings["layers"]:
        index = layer if layer >= 0 else len(output.hidden_states) + layer
        if not 0 <= index < len(output.hidden_states) or index in resolved:
            raise ValueError(f"Invalid or duplicate resolved layer {layer}")
        resolved.add(index)
        hidden = output.hidden_states[index].float()
        if settings["representation"] == "answer_mean":
            pooled = (hidden * pool_mask.unsqueeze(-1)).sum(1) / pool_mask.sum(1, keepdim=True)
        else:
            pooled = hidden[:, -1]
        result[str(layer)] = pooled.cpu()
    # Compute only answer-token logits to avoid a full float32 [batch, tokens, vocab] copy.
    active = answer_mask[:, 1:]
    positions = active.nonzero(as_tuple=True)
    selected = output.logits[:, :-1][positions].float().log_softmax(-1)
    chosen = selected.gather(1, ids[:, 1:][positions].unsqueeze(1)).squeeze(1)
    sums = torch.zeros(len(rows), device=device).scatter_add_(0, positions[0], chosen)
    counts = active.sum(1)
    likelihood = sums / counts.clamp_min(1)
    likelihood[counts == 0] = -1e6
    return result, likelihood.cpu().tolist()


def collect_features(config: dict, source: str, force: bool = False) -> Path:
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source}")
    target = _stage(config, f"features/{source}", force)
    if (target / "_SUCCESS.json").exists() and not force:
        return target
    rows = _replay_rows(config)
    digest = _digest(rows)
    settings = {
        **validate(config),
        "max_sequence_length": int(config["model"]["max_sequence_length"]),
    }
    tokenizer = load_tokenizer(config["model"])
    model = load_base_model(config["model"])
    if source != "base":
        from peft import PeftModel

        adapter = root(config) / "adapters" / source / "final"
        _require(adapter)
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    for shard, group in chunks(rows, int(settings["shard_size"])):
        path = target / f"shard_{shard:05d}.pt"
        if path.exists() and not force:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["rows_sha256"] != digest:
                raise ValueError("Replay text changed; use a new output directory")
            continue
        parts = {str(layer): [] for layer in settings["layers"]}
        likelihood = []
        for _, batch in chunks(group, int(settings["batch_size"])):
            features, scores = replay(model, tokenizer, batch, settings)
            likelihood.extend(scores)
            for key, tensor in features.items():
                parts[key].append(tensor)
        atomic_torch_save(
            path,
            {
                "row_ids": [r["row_id"] for r in group],
                "rows_sha256": digest,
                "features": {k: torch.cat(v) for k, v in parts.items()},
                "mean_logprob": likelihood,
            },
        )
    del model
    _release()
    _complete(target, {"rows_sha256": digest, "num_rows": len(rows)})
    return target


def load_features(config: dict, rows: list[dict], source: str) -> tuple[dict, np.ndarray]:
    target = root(config) / "features" / source
    _require(target)
    ids, likelihood = [], []
    parts = {str(layer): [] for layer in validate(config)["layers"]}
    digest = _digest(rows)
    for path in sorted(target.glob("shard_*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["rows_sha256"] != digest:
            raise ValueError(f"Mismatched replay corpus for {source}")
        ids.extend(saved["row_ids"])
        likelihood.extend(saved["mean_logprob"])
        for key, values in parts.items():
            values.append(saved["features"][key].float().numpy())
    if ids != [r["row_id"] for r in rows]:
        raise ValueError(f"Mismatched replay order for {source}")
    return {key: np.concatenate(value) for key, value in parts.items()}, np.asarray(likelihood)


def fit_directions(base: np.ndarray, adapted: np.ndarray, ranks: list[int], seed: int) -> dict:
    """Fit on direction-fit data only. Uncentered SVD retains the mean training shift."""
    from sklearn.utils.extmath import randomized_svd

    delta = np.asarray(adapted - base, dtype=np.float64)
    maximum = min(max(ranks), *delta.shape)
    if not np.isfinite(delta).all() or np.linalg.norm(delta) < 1e-12:
        raise ValueError("Nonfinite or zero activation update")
    _, singular, vt = randomized_svd(delta, n_components=maximum, random_state=seed)
    mean = delta.mean(0)
    return {
        "basis": vt.T,
        "singular_values": singular,
        "energy_fraction": singular**2 / np.square(delta).sum(),
        "mean_direction": mean / max(np.linalg.norm(mean), 1e-12),
        "origin": np.asarray(base, dtype=np.float64).mean(0),
    }


def direction_scores(values: np.ndarray, fit: dict, ranks: list[int]) -> dict[str, np.ndarray]:
    projection = values @ fit["basis"]
    scores = {"svd1": projection[:, 0], "mean": values @ fit["mean_direction"]}
    norms = np.maximum(np.linalg.norm(values, axis=1), 1e-12)
    scores["svd1_cosine"] = scores["svd1"] / norms
    scores["mean_cosine"] = scores["mean"] / norms
    for rank in ranks:
        if rank <= fit["basis"].shape[1]:
            scores[f"energy_{rank}"] = np.square(projection[:, :rank]).sum(1)
            scores[f"fraction_{rank}"] = scores[f"energy_{rank}"] / np.square(norms)
    return scores


def _auc(labels, scores) -> float | None:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None


def orient(labels, validation_scores) -> int:
    auc = _auc(labels, validation_scores)
    if auc is None:
        raise ValueError("Validation needs both correct and incorrect answers")
    return 1 if auc >= 0.5 else -1


def _groups(rows: list[dict]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        result.setdefault(row["fact_id"], []).append(i)
    return result


def evaluate_scores(
    rows: list[dict], scores: np.ndarray, coverages: list[float], bootstrap_samples: int, seed: int
) -> dict:
    scores = np.asarray(scores)
    if len(rows) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and match rows")
    labels = np.asarray([r["correct"] for r in rows])
    eligible = labels >= 0
    y, s = labels[eligible], scores[eligible]
    ranked = np.argsort(-s, kind="stable")
    selective = {}
    for c in coverages:
        n = max(1, int(np.ceil(len(y) * c)))
        selective[str(c)] = float(y[ranked[:n]].mean()) if len(y) else None
    groups = _groups(rows)
    # Fixed candidate-index order breaks ties, independent of correctness labels.
    chosen, greedy, oracle = [], [], []
    for indices in groups.values():
        indices = sorted(indices, key=lambda i: rows[i]["candidate_index"])
        chosen.append(labels[max(indices, key=lambda i: scores[i])])
        greedy.append(labels[indices[0]])
        oracle.append(
            int(any(labels[i] == 1 for i in indices))
            if any(labels[i] >= 0 for i in indices)
            else -1
        )

    # For OOD, ambiguous reference matches remain visible and are excluded from accuracy.
    def accuracy(values):
        values = np.asarray(values)
        return float(values[values >= 0].mean()) if (values >= 0).any() else None

    # Bootstrap by connected question/alias group, retaining all candidates per question.
    clusters: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        if eligible[i]:
            clusters.setdefault(row["group_id"], []).append(i)
    arrays = list(clusters.values())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap_samples if arrays else 0):
        index = np.concatenate([arrays[i] for i in rng.integers(len(arrays), size=len(arrays))])
        auc = _auc(labels[index], scores[index])
        if auc is not None:
            draws.append(auc)
    return {
        "n": len(rows),
        "n_graded": int(eligible.sum()),
        "n_questions": len(groups),
        "answer_accuracy": accuracy(labels),
        "auroc": _auc(y, s),
        "auroc_ci95": np.quantile(draws, [0.025, 0.975]).tolist() if draws else None,
        "selective_accuracy": selective,
        "reranked_accuracy": accuracy(chosen),
        "reranked_graded_questions": int(sum(x >= 0 for x in chosen)),
        "greedy_accuracy": accuracy(greedy),
        "oracle_any_correct_rate": accuracy(oracle),
    }


def analyze(config: dict, force: bool = False) -> Path:
    target = _stage(config, "results", force)
    if (target / "_SUCCESS.json").exists() and not force:
        return target
    settings = validate(config)
    rows = _replay_rows(config)
    loaded = {source: load_features(config, rows, source) for source in SOURCES}
    split = np.asarray([r["split"] for r in rows])
    labels = np.asarray([r["correct"] for r in rows])
    fit_mask, train_mask, val = (
        split == "direction_fit",
        split == "probe_train",
        split == "validation",
    )
    if len(np.unique(labels[val])) != 2 or len(np.unique(labels[train_mask])) != 2:
        raise ValueError("Probe-training and validation splits each need both correctness classes")
    seed = int(config["experiment"]["seed"])
    scores: dict[str, np.ndarray] = {"base_likelihood": loaded["base"][1]}
    transfers, artifacts, diagnostics = {}, {}, {}
    for layer in map(str, settings["layers"]):
        base = loaded["base"][0][layer]
        fitted = {}
        for source in ADAPTERS:
            adapted = loaded[source][0][layer]
            fit = fit_directions(base[fit_mask], adapted[fit_mask], settings["ranks"], seed)
            fitted[source] = fit
            artifacts[f"{layer}/{source}"] = fit
            diagnostics[f"{layer}/{source}"] = {
                "singular_values": fit["singular_values"].tolist(),
                "energy_fraction": fit["energy_fraction"].tolist(),
                "mean_update_norm": float(np.linalg.norm((adapted - base)[fit_mask].mean(0))),
            }
            for view, values in (
                ("base", base - fit["origin"]),
                ("on_policy", adapted - fit["origin"]),
                ("delta", adapted - base),
            ):
                for method, score in direction_scores(values, fit, settings["ranks"]).items():
                    scores[f"{layer}/{source}/{view}/{method}"] = score
            scores[f"{layer}/{source}/delta/norm"] = np.linalg.norm(adapted - base, axis=1)
        a, b, wrong = (fitted[source] for source in ADAPTERS)
        diagnostics[f"{layer}/overlap"] = {
            "a_b_pc1_absolute_cosine": float(abs(a["basis"][:, 0] @ b["basis"][:, 0])),
            "a_incorrect_pc1_absolute_cosine": float(abs(a["basis"][:, 0] @ wrong["basis"][:, 0])),
            "a_b_subspace_overlap": float(
                np.square(a["basis"].T @ b["basis"]).sum()
                / min(a["basis"].shape[1], b["basis"].shape[1])
            ),
        }
        # Contrast A and its matched incorrect adapter on identical direction-fit text.
        contrast = fit_directions(
            loaded["incorrect"][0][layer][fit_mask],
            loaded["correct_a"][0][layer][fit_mask],
            settings["ranks"],
            seed,
        )
        contrast["origin"] = a["origin"]
        artifacts[f"{layer}/contrast"] = contrast
        for method, score in direction_scores(
            base - a["origin"], contrast, settings["ranks"]
        ).items():
            scores[f"{layer}/contrast/base/{method}"] = score
        # Freeze A's direction, origin, and later sign for evaluation on B.
        for view, values in (
            ("on_policy", loaded["correct_b"][0][layer] - a["origin"]),
            ("delta", loaded["correct_b"][0][layer] - base),
        ):
            for method, score in direction_scores(values, a, settings["ranks"]).items():
                transfers[f"{layer}/correct_a/{view}/{method}"] = score
        rng = np.random.default_rng(seed)
        random = rng.normal(size=base.shape[1])
        random /= np.linalg.norm(random)
        scores[f"{layer}/random/base"] = (base - a["origin"]) @ random
        # Ordinary activation PCA, fitted on precisely the same direction-fit rows.
        pca = fit_directions(
            np.zeros_like(base[fit_mask]), base[fit_mask] - a["origin"], settings["ranks"], seed
        )
        artifacts[f"{layer}/activation_pca"] = pca
        for method, score in direction_scores(base - a["origin"], pca, settings["ranks"]).items():
            scores[f"{layer}/activation_pca/{method}"] = score
        probe = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        )
        # Explicit extra-label baseline; geometry uses no probe_train correctness labels.
        probe.fit(base[train_mask], labels[train_mask])
        scores[f"{layer}/supervised_probe"] = probe.predict_proba(base)[:, 1]
        artifacts[f"{layer}/supervised_probe"] = probe
    signs = {name: orient(labels[val], score[val]) for name, score in scores.items()}
    validation = {
        name: _auc(labels[val], signs[name] * score[val]) for name, score in scores.items()
    }
    geometric = [name for name in scores if "/correct_a/" in name and not name.endswith("/norm")]
    selected = max(geometric, key=lambda name: validation[name])
    selected_transfer = max(transfers, key=lambda name: validation[name])
    families = {
        source: [n for n in scores if f"/{source}/" in n and not n.endswith("/norm")]
        for source in (*ADAPTERS, "contrast", "activation_pca")
    }
    selected_by_family = {
        family: max(names, key=lambda name: validation[name]) for family, names in families.items()
    }
    artifacts["signs"] = signs
    artifacts["selected"] = selected
    artifacts["selected_transfer"] = selected_transfer
    artifacts["selected_by_family"] = selected_by_family
    atomic_torch_save(target / "frozen_methods.pt", artifacts)
    reports, predictions = {}, []
    for name, raw in scores.items():
        variants = {"native": signs[name] * raw}
        if name in transfers:
            variants["a_to_b"] = signs[name] * transfers[name]
        for variant, score in variants.items():
            key = name if variant == "native" else f"{name}/a_to_b"
            reports[key] = {"validation_auroc": validation[name], "orientation": signs[name]}
            for dataset in ("test", "ood"):
                mask = split == dataset
                if not mask.any():
                    continue
                subset = [r for r, include in zip(rows, mask, strict=True) if include]
                reports[key][dataset] = evaluate_scores(
                    subset,
                    score[mask],
                    settings["coverages"],
                    int(settings["bootstrap_samples"]),
                    seed,
                )
            if variant == "native" and name in {selected, "base_likelihood"}:
                predictions.extend(
                    {
                        "row_id": r["row_id"],
                        "question": r["question"],
                        "answer": r["answer"],
                        "correct": r["correct"],
                        "split": r["split"],
                        "method": name,
                        "score": float(s),
                    }
                    for r, s in zip(rows, score, strict=True)
                    if r["split"] in {"test", "ood"}
                )
    atomic_json(
        target / "metrics.json",
        {
            "schema_version": SCHEMA,
            "selected_on_validation": selected,
            "selected_transfer_on_a_validation": selected_transfer,
            "selected_by_family": selected_by_family,
            "rows_sha256": _digest(rows),
            "diagnostics": diagnostics,
            "methods": reports,
            "limitations": [
                "Negatives are reference-rejected model samples; audit for valid unlisted aliases.",
                "TruthfulQA reference-match grading excludes ambiguous outputs.",
                "Supervised probe uses additional probe_train correctness labels.",
                "Duplicate/alias grouping cannot guarantee semantic fact disjointness.",
            ],
        },
    )
    atomic_jsonl(target / "predictions.jsonl", predictions)
    lines = [
        "# Factual LoRA shift correctness experiment",
        "",
        f"Frozen validation-selected method: `{selected}`",
        "",
        "AUROC is for correctness. Confidence intervals bootstrap question/alias groups.",
        "",
        "| Method | Validation AUROC | Test AUROC | OOD AUROC |",
        "|---|---:|---:|---:|",
    ]
    summary_names = list(
        dict.fromkeys(
            [
                selected,
                f"{selected_transfer}/a_to_b",
                *selected_by_family.values(),
                "base_likelihood",
                *[n for n in reports if "supervised_probe" in n],
            ]
        )
    )
    for name in summary_names:
        report = reports[name]
        lines.append(
            f"| {name} | {report['validation_auroc']} | {report.get('test', {}).get('auroc')} | "
            f"{report.get('ood', {}).get('auroc')} |"
        )
    (target / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _complete(target, {"selected": selected})
    return target
