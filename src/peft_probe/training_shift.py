from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import PeftModel
from sklearn.metrics import roc_auc_score

from .config import config_fingerprint, output_dir, validate_source_run
from .collect import _pool_hidden
from .geometry import _extract_prompt_features, _fit_subspace, _subspace_alignment
from .io_utils import (
    atomic_json,
    atomic_torch_save,
    chunks,
    ensure_manifest,
    mark_complete,
    read_jsonl,
)
from .modeling import load_base_model, load_tokenizer, model_device, user_prompt
from .prompting import synthetic_eval_question
from .versioning import artifact_metadata


TRAINING_SHIFT_SCHEMA_VERSION = 1
SOURCES = ("base", "adapter_a", "adapter_b")


def load_training_shift_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError("Training-shift configuration must be a mapping")
    if int(settings.get("training_shift_schema_version", -1)) != TRAINING_SHIFT_SCHEMA_VERSION:
        raise ValueError(
            f"Expected training-shift schema {TRAINING_SHIFT_SCHEMA_VERSION}, "
            f"found {settings.get('training_shift_schema_version')}"
        )
    layers = settings.get("layers")
    if not isinstance(layers, list) or not layers or not all(isinstance(x, int) for x in layers):
        raise ValueError("training_shift.layers must be a non-empty list of integer indices")
    if len(set(layers)) != len(layers):
        raise ValueError("training_shift.layers must not contain duplicates")
    if settings.get("pooling", "last") not in {"last", "mean"}:
        raise ValueError("training_shift.pooling must be 'last' or 'mean'")
    representation = settings.get("representation", "prompt_last")
    if representation not in {"prompt_last", "answer_last", "answer_mean"}:
        raise ValueError(
            "training_shift.representation must be prompt_last, answer_last, or answer_mean"
        )
    for key in ("batch_size", "shard_size", "max_components"):
        if int(settings.get(key, 0)) <= 0:
            raise ValueError(f"training_shift.{key} must be a positive integer")
    threshold = float(settings.get("variance_threshold", 0.0))
    if not 0 < threshold <= 1:
        raise ValueError("training_shift.variance_threshold must be in (0, 1]")
    int(settings["seed"])
    return settings


def training_shift_fingerprint(config: dict[str, Any], settings: dict[str, Any]) -> str:
    payload = {
        "experiment_fingerprint": config_fingerprint(config),
        "training_shift": settings,
        "training_shift_schema_version": TRAINING_SHIFT_SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def training_shift_root(
    config: dict[str, Any], settings: dict[str, Any] | None = None
) -> Path:
    representation = (settings or {}).get("representation", "prompt_last")
    if representation == "prompt_last":
        name = f"training_shift_v{TRAINING_SHIFT_SCHEMA_VERSION}"
    else:
        name = f"training_shift_qa_{representation}_v{TRAINING_SHIFT_SCHEMA_VERSION}"
    return output_dir(config) / "geometry" / name


def _metadata(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "training_shift_schema_version": TRAINING_SHIFT_SCHEMA_VERSION,
        "training_shift_fingerprint": training_shift_fingerprint(config, settings),
        **artifact_metadata(),
    }


def _factual_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    source_root = validate_source_run(config)
    facts = sorted(read_jsonl(source_root / "data" / "facts.jsonl"), key=lambda row: row["fact_id"])
    return [
        {
            "fact_id": row["fact_id"],
            "entity_id": row["entity_id"],
            "relation": row["relation"],
            "split": row["detector_split"],
            "adapter_assignment": row["adapter_assignment"],
            "question": synthetic_eval_question(config, row),
            "answer": str(row["value"]),
        }
        for row in facts
        if row["adapter_assignment"] in {"a", "b"}
    ]


def _question_answer_batch(tokenizer, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    encoded: list[tuple[list[int], int]] = []
    for row in rows:
        prompt_ids = tokenizer(
            user_prompt(tokenizer, row["question"]), add_special_tokens=False
        )["input_ids"]
        answer_ids = tokenizer(" " + row["answer"].strip(), add_special_tokens=False)[
            "input_ids"
        ]
        if not answer_ids:
            raise ValueError(f"Canonical answer tokenized to zero tokens: {row['fact_id']}")
        encoded.append((list(prompt_ids) + list(answer_ids), len(answer_ids)))

    width = max(len(ids) for ids, _ in encoded)
    input_ids = []
    attention_mask = []
    answer_mask = []
    for ids, answer_length in encoded:
        padding = width - len(ids)
        input_ids.append([tokenizer.pad_token_id] * padding + ids)
        attention_mask.append([0] * padding + [1] * len(ids))
        answer_mask.append(
            [0] * padding + [0] * (len(ids) - answer_length) + [1] * answer_length
        )
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "answer_mask": torch.tensor(answer_mask, dtype=torch.bool),
    }


@torch.inference_mode()
def _extract_question_answer_features(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    batch = _question_answer_batch(tokenizer, rows)
    answer_mask = batch.pop("answer_mask")
    device = model_device(model)
    tokens = {key: value.to(device) for key, value in batch.items()}
    answer_mask = answer_mask.to(device)
    outputs = model(**tokens, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden_states = outputs.hidden_states
    pooling = "last" if settings["representation"] == "answer_last" else "mean"
    result: dict[str, torch.Tensor] = {}
    resolved: set[int] = set()
    for requested in settings["layers"]:
        index = requested if requested >= 0 else len(hidden_states) + requested
        if not 0 <= index < len(hidden_states):
            raise ValueError(
                f"Layer {requested} is invalid for a model with {len(hidden_states)} hidden states"
            )
        if index in resolved:
            raise ValueError(f"Training-shift layers resolve to duplicate hidden state {index}")
        resolved.add(index)
        pooled = _pool_hidden(hidden_states[index], answer_mask, pooling)
        result[str(requested)] = pooled.detach().to(dtype=torch.float16, device="cpu")
    return result


def collect_training_shift(
    config: dict[str, Any], settings: dict[str, Any], source: str, force: bool = False
) -> Path:
    if source not in SOURCES:
        raise ValueError(f"Training-shift source must be one of {SOURCES}")
    target = training_shift_root(config, settings) / "features" / source
    fingerprint = training_shift_fingerprint(config, settings)
    ensure_manifest(target, fingerprint, force=force)
    done = target / "_SUCCESS.json"
    if done.exists() and not force:
        return target

    tokenizer = load_tokenizer(config["model"])
    model = load_base_model(config["model"], training=False)
    if source != "base":
        adapter_path = validate_source_run(config) / "adapters" / source / "final"
        if not (adapter_path / "_SUCCESS.json").exists():
            raise FileNotFoundError(f"Adapter is incomplete: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()

    rows = _factual_rows(config)
    shard_size = int(settings["shard_size"])
    batch_size = int(settings["batch_size"])
    expected_shards = math.ceil(len(rows) / shard_size)
    for shard_index, shard_rows in chunks(rows, shard_size):
        shard_path = target / f"shard_{shard_index:05d}.pt"
        if shard_path.exists() and not force:
            continue
        records: list[dict[str, Any]] = []
        parts = {str(layer): [] for layer in settings["layers"]}
        for _, batch_rows in chunks(shard_rows, batch_size):
            if settings.get("representation", "prompt_last") == "prompt_last":
                features = _extract_prompt_features(
                    model,
                    tokenizer,
                    [row["question"] for row in batch_rows],
                    settings,
                )
            else:
                features = _extract_question_answer_features(
                    model, tokenizer, batch_rows, settings
                )
            records.extend(batch_rows)
            for layer, tensor in features.items():
                parts[layer].append(tensor)
        atomic_torch_save(
            shard_path,
            {
                "records": records,
                "features": {layer: torch.cat(values) for layer, values in parts.items()},
                "source": source,
                "shard_index": shard_index,
                **_metadata(config, settings),
            },
        )
    actual_shards = len(list(target.glob("shard_*.pt")))
    if actual_shards != expected_shards:
        raise RuntimeError(f"Expected {expected_shards} shards, found {actual_shards}")
    mark_complete(
        done,
        {
            "source": source,
            "num_examples": len(rows),
            "num_shards": actual_shards,
            **_metadata(config, settings),
        },
    )
    return target


def _load_source(directory: Path, layer: int) -> tuple[list[dict[str, Any]], torch.Tensor]:
    records: list[dict[str, Any]] = []
    features: list[torch.Tensor] = []
    for path in sorted(directory.glob("shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(payload["records"])
        features.append(payload["features"][str(layer)].float())
    if not features:
        raise FileNotFoundError(f"No training-shift shards found in {directory}")
    return records, torch.cat(features)


def _align_sources(
    records_by_source: dict[str, list[dict[str, Any]]],
    features_by_source: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    base_records = records_by_source["base"]
    ids = [row["fact_id"] for row in base_records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate fact IDs in base training-shift features")
    aligned = {"base": features_by_source["base"]}
    for source in ("adapter_a", "adapter_b"):
        index = {row["fact_id"]: i for i, row in enumerate(records_by_source[source])}
        missing = [fact_id for fact_id in ids if fact_id not in index]
        if missing:
            raise RuntimeError(f"{source} is missing {len(missing)} matched facts")
        aligned[source] = features_by_source[source][[index[fact_id] for fact_id in ids]]
    return base_records, aligned


def _fit_centered_subspace(
    values: torch.Tensor, maximum: int, threshold: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    mean = values.mean(0)
    basis, spectrum = _fit_subspace(values - mean, maximum, threshold, seed)
    return mean, basis, spectrum


def _projection_fraction(values: torch.Tensor, basis: torch.Tensor) -> float:
    denominator = values.square().sum()
    if float(denominator) <= 1e-12:
        return float("nan")
    return float((values @ basis).square().sum() / denominator)


def _ordered_vector_alignment(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    """Compare PC/SVD vectors by variance order while ignoring arbitrary signs."""
    rank = min(first.shape[1], second.shape[1])
    if rank == 0:
        return {"mean_absolute_cosine": float("nan"), "absolute_cosines": []}
    cosines = torch.diagonal(first[:, :rank].T @ second[:, :rank]).abs()
    return {
        "mean_absolute_cosine": float(cosines.mean()),
        "absolute_cosines": [float(value) for value in cosines],
    }


def _alignment_with_random_reference(
    first: torch.Tensor, second: torch.Tensor
) -> dict[str, Any]:
    result = _subspace_alignment(first, second)
    expected = max(first.shape[1], second.shape[1]) / first.shape[0]
    result["random_expected_mean_cosine_squared"] = float(expected)
    result["excess_over_random"] = float(result["mean_cosine_squared"] - expected)
    return result


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _cosine_rows(first: torch.Tensor, second: torch.Tensor) -> np.ndarray:
    denominator = first.norm(dim=1) * second.norm(dim=1)
    valid = denominator > 1e-12
    result = torch.full((len(first),), torch.nan)
    result[valid] = (first[valid] * second[valid]).sum(1) / denominator[valid]
    return result.numpy()


def _analyze_layer(
    records: list[dict[str, Any]],
    features: dict[str, torch.Tensor],
    settings: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    splits = np.asarray([row["split"] for row in records])
    assignments = np.asarray([row["adapter_assignment"] for row in records])
    relations = np.asarray([row["relation"] for row in records])
    train = splits == "train"
    test = splits == "test"
    maximum = int(settings["max_components"])
    threshold = float(settings["variance_threshold"])

    state_fits: dict[str, dict[str, Any]] = {}
    saved_states: dict[str, Any] = {}
    for source in SOURCES:
        mean, basis, spectrum = _fit_centered_subspace(
            features[source][train], maximum, threshold, seed
        )
        state_fits[source] = {"mean": mean, "basis": basis, "spectrum": spectrum}
        saved_states[source] = {"mean": mean.half(), "basis": basis.half(), "spectrum": spectrum}

    state_alignment = {
        "base_to_adapter_a": _alignment_with_random_reference(
            state_fits["base"]["basis"], state_fits["adapter_a"]["basis"]
        ),
        "base_to_adapter_b": _alignment_with_random_reference(
            state_fits["base"]["basis"], state_fits["adapter_b"]["basis"]
        ),
        "adapter_a_to_adapter_b": _alignment_with_random_reference(
            state_fits["adapter_a"]["basis"], state_fits["adapter_b"]["basis"]
        ),
    }
    state_vector_alignment = {
        "base_to_adapter_a": _ordered_vector_alignment(
            state_fits["base"]["basis"], state_fits["adapter_a"]["basis"]
        ),
        "base_to_adapter_b": _ordered_vector_alignment(
            state_fits["base"]["basis"], state_fits["adapter_b"]["basis"]
        ),
        "adapter_a_to_adapter_b": _ordered_vector_alignment(
            state_fits["adapter_a"]["basis"], state_fits["adapter_b"]["basis"]
        ),
    }

    updates = {
        "adapter_a": features["adapter_a"] - features["base"],
        "adapter_b": features["adapter_b"] - features["base"],
    }
    update_fits: dict[str, dict[str, Any]] = {}
    saved_updates: dict[str, Any] = {}
    adapter_results: dict[str, Any] = {}
    for source in ("adapter_a", "adapter_b"):
        letter = source[-1]
        exposed_train = train & (assignments == letter)
        withheld_train = train & (assignments != letter)
        exposed_test = test & (assignments == letter)
        withheld_test = test & (assignments != letter)
        delta = updates[source]

        all_basis, all_spectrum = _fit_subspace(
            delta[train], maximum, threshold, seed + 100
        )
        exposed_basis, exposed_spectrum = _fit_subspace(
            delta[exposed_train], maximum, threshold, seed + 200
        )
        withheld_basis, withheld_spectrum = _fit_subspace(
            delta[withheld_train], maximum, threshold, seed + 300
        )
        update_fits[source] = {
            "all": all_basis,
            "exposed": exposed_basis,
            "withheld": withheld_basis,
        }
        saved_updates[source] = {
            "all_basis": all_basis.half(),
            "exposed_basis": exposed_basis.half(),
            "withheld_basis": withheld_basis.half(),
            "all_spectrum": all_spectrum,
            "exposed_spectrum": exposed_spectrum,
            "withheld_spectrum": withheld_spectrum,
        }

        test_labels = (assignments[test] == letter).astype(int)
        exposed_energy = delta[test] @ exposed_basis
        energy_scores = exposed_energy.square().sum(1).numpy()
        norm_scores = delta[test].square().sum(1).numpy()
        adapter_results[source] = {
            "all_update_spectrum": all_spectrum,
            "exposed_update_spectrum": exposed_spectrum,
            "withheld_update_spectrum": withheld_spectrum,
            "exposed_to_withheld_update_overlap": _alignment_with_random_reference(
                exposed_basis, withheld_basis
            ),
            "test": {
                "mean_update_norm_exposed": float(delta[exposed_test].norm(dim=1).mean()),
                "mean_update_norm_withheld": float(delta[withheld_test].norm(dim=1).mean()),
                "exposed_subspace_fraction_exposed": _projection_fraction(
                    delta[exposed_test], exposed_basis
                ),
                "exposed_subspace_fraction_withheld": _projection_fraction(
                    delta[withheld_test], exposed_basis
                ),
                "exposure_auroc_from_update_norm": _safe_auroc(test_labels, norm_scores),
                "exposure_auroc_from_exposed_subspace_energy": _safe_auroc(
                    test_labels, energy_scores
                ),
            },
        }

    relation_alignment: dict[str, Any] = {}
    for relation_index, relation in enumerate(sorted(set(relations[train]))):
        mask = train & (relations == relation)
        relation_alignment[relation] = {}
        for source_index, source in enumerate(("adapter_a", "adapter_b")):
            relation_basis, _ = _fit_subspace(
                updates[source][mask],
                update_fits[source]["all"].shape[1],
                1.0,
                seed + 400 + relation_index * 10 + source_index,
            )
            relation_alignment[relation][source] = _alignment_with_random_reference(
                update_fits[source]["all"], relation_basis
            )

    cosine = _cosine_rows(updates["adapter_a"][test], updates["adapter_b"][test])
    layer_result = {
        "state_spectra": {
            source: state_fits[source]["spectrum"] for source in SOURCES
        },
        "state_subspace_alignment": state_alignment,
        "ordered_principal_vector_alignment": state_vector_alignment,
        "centroid_drift": {
            source: {
                "norm": float(
                    (state_fits[source]["mean"] - state_fits["base"]["mean"]).norm()
                ),
                "relative_to_base_rms": float(
                    (state_fits[source]["mean"] - state_fits["base"]["mean"]).norm()
                    / (features["base"][train] - state_fits["base"]["mean"])
                    .norm(dim=1)
                    .square()
                    .mean()
                    .sqrt()
                    .clamp_min(1e-12)
                ),
            }
            for source in ("adapter_a", "adapter_b")
        },
        "update_subspace_alignment": {
            "all_a_to_all_b": _alignment_with_random_reference(
                update_fits["adapter_a"]["all"], update_fits["adapter_b"]["all"]
            ),
            "exposed_a_to_exposed_b": _alignment_with_random_reference(
                update_fits["adapter_a"]["exposed"], update_fits["adapter_b"]["exposed"]
            ),
        },
        "ordered_update_vector_alignment": _ordered_vector_alignment(
            update_fits["adapter_a"]["all"], update_fits["adapter_b"]["all"]
        ),
        "paired_a_b_update_cosine_test": {
            "mean": float(np.nanmean(cosine)),
            "median": float(np.nanmedian(cosine)),
        },
        "adapters": adapter_results,
        "relation_alignment": relation_alignment,
    }
    saved = {"states": saved_states, "updates": saved_updates}
    return layer_result, saved


def analyze_training_shift(
    config: dict[str, Any], settings: dict[str, Any], force: bool = False
) -> Path:
    root = training_shift_root(config, settings)
    target = root / "results"
    fingerprint = training_shift_fingerprint(config, settings)
    ensure_manifest(target, fingerprint, force=force)
    result_path = target / "training_shift_metrics.json"
    done = target / "_SUCCESS.json"
    if done.exists() and result_path.exists() and not force:
        return result_path

    representation = settings.get("representation", "prompt_last")
    inputs = (
        "identical synthetic factual evaluation prompts"
        if representation == "prompt_last"
        else "identical synthetic factual prompts plus canonical correct answers"
    )
    results: dict[str, Any] = {
        "method": {
            "inputs": inputs,
            "representation": representation,
            "state_pca_fit": "centered activations on detector train split",
            "update_svd_fit": "paired LoRA activation minus base activation on train split",
            "evaluation": "detector test split",
            "random_overlap_reference": "analytic expectation for independent random subspaces",
        },
        "layers": {},
        **_metadata(config, settings),
    }
    saved_layers: dict[str, Any] = {}
    feature_root = root / "features"
    for layer_index, layer in enumerate(settings["layers"]):
        records_by_source: dict[str, list[dict[str, Any]]] = {}
        features_by_source: dict[str, torch.Tensor] = {}
        for source in SOURCES:
            records, source_features = _load_source(feature_root / source, layer)
            records_by_source[source] = records
            features_by_source[source] = source_features
        records, aligned = _align_sources(records_by_source, features_by_source)
        layer_result, saved = _analyze_layer(
            records, aligned, settings, int(settings["seed"]) + layer_index * 1000
        )
        results["layers"][str(layer)] = layer_result
        saved_layers[str(layer)] = saved

    atomic_torch_save(
        target / "training_shift_subspaces.pt",
        {"layers": saved_layers, **_metadata(config, settings)},
    )
    atomic_json(result_path, results)
    summary_path = _write_summary(results, target / "training_shift_summary.md")
    mark_complete(
        done,
        {
            "metrics": str(result_path),
            "summary": str(summary_path),
            **_metadata(config, settings),
        },
    )
    return result_path


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.4f}" if np.isfinite(number) else "NA"


def _first_angle(alignment: dict[str, Any]) -> Any:
    angles = alignment.get("principal_angles_degrees", [])
    return angles[0] if angles else float("nan")


def _write_summary(results: dict[str, Any], path: Path) -> Path:
    representation = results["method"].get("representation", "prompt_last")
    title = (
        "LoRA factual prompt subspace shift"
        if representation == "prompt_last"
        else f"LoRA factual question+answer subspace shift ({representation})"
    )
    lines = [
        f"# {title}",
        "",
        f"Input representation: `{representation}`. {results['method']['inputs'].capitalize()} "
        "are evaluated before training (base) and after training (LoRA A/B). All PCA/SVD "
        "bases use only the synthetic detector-training split.",
        "",
        "## Dominant activation subspaces before and after training",
        "",
        "| Layer | Base rank | A rank | B rank | Base/A overlap | Base/B overlap | A/B overlap | "
        "Base/A PC1 cosine | Base/B PC1 cosine | Base/A first angle | Base/B first angle |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, item in results["layers"].items():
        spectra = item["state_spectra"]
        align = item["state_subspace_alignment"]
        ba = align["base_to_adapter_a"]
        bb = align["base_to_adapter_b"]
        ab = align["adapter_a_to_adapter_b"]
        vectors = item["ordered_principal_vector_alignment"]
        ba_pc = vectors["base_to_adapter_a"]["absolute_cosines"]
        bb_pc = vectors["base_to_adapter_b"]["absolute_cosines"]
        lines.append(
            f"| {layer} | {spectra['base']['rank']} | {spectra['adapter_a']['rank']} | "
            f"{spectra['adapter_b']['rank']} | {_fmt(ba['mean_cosine_squared'])} | "
            f"{_fmt(bb['mean_cosine_squared'])} | {_fmt(ab['mean_cosine_squared'])} | "
            f"{_fmt(ba_pc[0] if ba_pc else None)} | {_fmt(bb_pc[0] if bb_pc else None)} | "
            f"{_fmt(_first_angle(ba))} | {_fmt(_first_angle(bb))} |"
        )

    lines.extend(
        [
            "",
            "## Fine-tuning update subspaces",
            "",
            "| Layer | A/B update overlap | A/B exposed overlap | Mean paired update cosine | "
            "A norm AUROC | A subspace AUROC | B norm AUROC | B subspace AUROC |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for layer, item in results["layers"].items():
        align = item["update_subspace_alignment"]
        a = item["adapters"]["adapter_a"]["test"]
        b = item["adapters"]["adapter_b"]["test"]
        lines.append(
            f"| {layer} | {_fmt(align['all_a_to_all_b']['mean_cosine_squared'])} | "
            f"{_fmt(align['exposed_a_to_exposed_b']['mean_cosine_squared'])} | "
            f"{_fmt(item['paired_a_b_update_cosine_test']['mean'])} | "
            f"{_fmt(a['exposure_auroc_from_update_norm'])} | "
            f"{_fmt(a['exposure_auroc_from_exposed_subspace_energy'])} | "
            f"{_fmt(b['exposure_auroc_from_update_norm'])} | "
            f"{_fmt(b['exposure_auroc_from_exposed_subspace_energy'])} |"
        )

    lines.extend(
        [
            "",
            "High state-PCA overlap means ordinary factual-question geometry is largely preserved. "
            "Low state overlap means fine-tuning rotates dominant activation variance.",
            "",
            "Update-subspace metrics isolate what training changed. Exposure AUROC above 0.5 means "
            "the LoRA-minus-base change is stronger or more aligned for facts trained by that "
            "adapter. A/B update overlap tests whether both LoRAs implement those changes in a "
            "shared subspace. The JSON report includes the expected overlap of random subspaces.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
