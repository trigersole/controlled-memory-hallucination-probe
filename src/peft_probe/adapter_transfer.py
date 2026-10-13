from __future__ import annotations

import copy
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .collect import configured_feature_keys
from .config import config_fingerprint, output_dir
from .io_utils import (
    atomic_json,
    atomic_jsonl,
    atomic_torch_save,
    ensure_manifest,
    mark_complete,
    seed_everything,
)
from .metrics import binary_error_metrics
from .probe import Probe, _load_all, _predict, _shuffled_exposure, _standardize
from .versioning import artifact_metadata


ADAPTER_SOURCES = ("adapter_a", "adapter_b")
METRICS = ("auroc", "auprc", "brier", "ece", "aurc")


def _partition_masks(
    records: list[dict[str, Any]], train_source: str
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if train_source not in ADAPTER_SOURCES:
        raise ValueError(f"train_source must be one of {ADAPTER_SOURCES}")
    sources = np.asarray([record["source"] for record in records])
    splits = np.asarray([record["split"] for record in records])
    train = torch.from_numpy((sources == train_source) & (splits == "train"))
    validation = torch.from_numpy((sources == train_source) & (splits == "validation"))
    tests = {
        source: torch.from_numpy((sources == source) & (splits == "test"))
        for source in ADAPTER_SOURCES
    }
    counts = {
        "train": int(train.sum()),
        "validation": int(validation.sum()),
        **{f"test/{source}": int(mask.sum()) for source, mask in tests.items()},
    }
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise RuntimeError(f"Adapter-transfer partitions are empty: {empty}")
    return train, validation, tests


def _agreement_fact_ids(records: list[dict[str, Any]]) -> tuple[set[str], float]:
    by_source: dict[str, dict[str, int]] = {source: {} for source in ADAPTER_SOURCES}
    for record in records:
        if record["source"] in ADAPTER_SOURCES and record["split"] == "test":
            by_source[record["source"]][record["fact_id"]] = int(record["correct"])
    common = set(by_source["adapter_a"]) & set(by_source["adapter_b"])
    if not common:
        raise RuntimeError("No paired adapter test facts were found")
    if set(by_source["adapter_a"]) != set(by_source["adapter_b"]):
        raise RuntimeError("Adapter A and B test facts are not exactly paired")
    agreed = {
        fact_id
        for fact_id in common
        if by_source["adapter_a"][fact_id] == by_source["adapter_b"][fact_id]
    }
    return agreed, len(agreed) / len(common)


def _metric_degradation(
    self_metrics: dict[str, float], cross_metrics: dict[str, float]
) -> dict[str, float]:
    result = {}
    for name in METRICS:
        if name in {"auroc", "auprc"}:
            result[name] = float(self_metrics[name] - cross_metrics[name])
        else:
            result[name] = float(cross_metrics[name] - self_metrics[name])
    return result


def _fit_one(
    config: dict[str, Any],
    feature_mode: str,
    variant: str,
    seed: int,
    train_source: str,
    force: bool,
) -> tuple[Probe, dict[str, Any], list[dict[str, Any]], torch.Tensor, dict[str, torch.Tensor]]:
    settings = config["probe"]
    target = (
        output_dir(config)
        / "adapter_transfer"
        / "probes"
        / feature_mode
        / variant
        / f"seed_{seed}"
        / f"train_{train_source}"
    )
    ensure_manifest(target, config_fingerprint(config), force=force)
    seed_everything(seed + (0 if train_source == "adapter_a" else 10_000))
    records, raw_features = _load_all(config, feature_mode)
    train_mask, validation_mask, test_masks = _partition_masks(records, train_source)
    features, mean, std = _standardize(raw_features.float(), train_mask)
    correctness = torch.tensor([record["correct"] for record in records], dtype=torch.float32)
    exposure = torch.tensor([record["exposure"] for record in records], dtype=torch.float32)
    if variant == "shuffled_exposure":
        exposure = _shuffled_exposure(records, seed)
    auxiliary_mask = exposure.ge(0)
    train_indices = train_mask.nonzero(as_tuple=False).squeeze(1)
    train_dataset = TensorDataset(
        features[train_indices],
        correctness[train_indices],
        exposure[train_indices],
        auxiliary_mask[train_indices],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Probe(
        features.shape[1], settings["architecture"], int(settings["hidden_dim"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    correctness_loss = nn.BCEWithLogitsLoss()
    exposure_loss = nn.BCEWithLogitsLoss()
    lambda_exposure = 0.0 if variant == "correctness" else float(settings["lambda_exposure"])

    checkpoint = target / "last_checkpoint.pt"
    start_epoch = 0
    best_validation = math.inf
    best_state = None
    patience_left = int(settings["patience"])
    history: list[dict[str, Any]] = []
    if checkpoint.exists() and not force:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(device)
        start_epoch = int(state["epoch"]) + 1
        best_validation = float(state["best_validation"])
        best_state = state["best_state"]
        patience_left = int(state["patience_left"])
        history = state["history"]
        if patience_left <= 0:
            start_epoch = int(settings["epochs"])

    for epoch in range(start_epoch, int(settings["epochs"])):
        loader = DataLoader(
            train_dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        model.train()
        for batch_features, batch_correctness, batch_exposure, batch_auxiliary_mask in loader:
            batch_features = batch_features.to(device)
            batch_correctness = batch_correctness.to(device)
            batch_exposure = batch_exposure.to(device)
            batch_auxiliary_mask = batch_auxiliary_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            correctness_logits, exposure_logits = model(batch_features)
            loss = correctness_loss(correctness_logits, batch_correctness)
            if lambda_exposure and batch_auxiliary_mask.any():
                loss = loss + lambda_exposure * exposure_loss(
                    exposure_logits[batch_auxiliary_mask],
                    batch_exposure[batch_auxiliary_mask],
                )
            loss.backward()
            optimizer.step()

        probability_error = _predict(
            model, features[validation_mask], device, int(settings["batch_size"])
        )
        labels = correctness[validation_mask].numpy().astype(int)
        validation_metrics = binary_error_metrics(labels, probability_error)
        validation_value = validation_metrics["brier"]
        history.append({"epoch": epoch, **validation_metrics})
        if validation_value < best_validation:
            best_validation = validation_value
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            patience_left = int(settings["patience"])
        else:
            patience_left -= 1
        atomic_torch_save(
            checkpoint,
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_validation": best_validation,
                "best_state": best_state,
                "patience_left": patience_left,
                "history": history,
            },
        )
        if patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError("Adapter-transfer probe training did not produce a checkpoint")
    model.load_state_dict(best_state)
    atomic_torch_save(
        target / "model.pt",
        {
            "model": best_state,
            "mean": mean,
            "std": std,
            "input_dim": features.shape[1],
            "architecture": settings["architecture"],
            "hidden_dim": int(settings["hidden_dim"]),
            "feature_mode": feature_mode,
            "variant": variant,
            "seed": seed,
            "train_source": train_source,
            **artifact_metadata(),
        },
    )
    atomic_jsonl(target / "training_history.jsonl", history)
    mark_complete(
        target / "_SUCCESS.json",
        {
            "feature_mode": feature_mode,
            "variant": variant,
            "seed": seed,
            "train_source": train_source,
            "best_validation_brier": best_validation,
            "config_fingerprint": config_fingerprint(config),
        },
    )
    return model, {"mean": mean, "std": std}, records, features, test_masks


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[dict[str, float]]] = {}
    for run in runs:
        for test_source, result in run["test"].items():
            key = (run["feature_mode"], run["variant"], run["train_source"], test_source)
            groups.setdefault(key, []).append(result["all"])
    aggregate = {}
    for (mode, variant, train_source, test_source), values in groups.items():
        label = f"{mode}/{variant}/{train_source}->{test_source}"
        aggregate[label] = {
            name: {
                "mean": fmean(float(value[name]) for value in values),
                "std": pstdev(float(value[name]) for value in values),
            }
            for name in METRICS
        }
    return aggregate


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Cross-adapter probe transfer",
        "",
        "This is an adapter-induced representation/distribution shift test, not a pure "
        "domain-drift test: both adapters use the same relations and synthetic data generator.",
        "",
        "For base-replay features, degradation reflects adapter-induced answer/content shift. "
        "For on-policy features, it can additionally reflect adapter representation shift.",
        "",
        f"Adapter A/B correctness agreement on paired test facts: "
        f"{payload['correctness_agreement_rate']:.3f}.",
        "",
        "Positive degradation means cross-adapter performance is worse than the held-out "
        "same-adapter baseline. AUROC/AUPRC degradation is self minus cross; lower-is-better "
        "metric degradation is cross minus self.",
        "",
        "| Feature / variant / seed | Direction | Self AUROC | Cross AUROC | AUROC degradation | "
        "Self Brier | Cross Brier | Brier degradation | Agreement-only AUROC degradation | "
        "Agreement-only Brier degradation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        train_source = run["train_source"]
        cross_source = "adapter_b" if train_source == "adapter_a" else "adapter_a"
        self_metrics = run["test"][train_source]["all"]
        cross_metrics = run["test"][cross_source]["all"]
        degradation = run["cross_adapter_degradation"]
        agreement_degradation = run["cross_adapter_degradation_label_agreement_only"]
        agreement_auroc = (
            agreement_degradation["auroc"] if agreement_degradation is not None else float("nan")
        )
        agreement_brier = (
            agreement_degradation["brier"] if agreement_degradation is not None else float("nan")
        )
        label = f"{run['feature_mode']} / {run['variant']} / {run['seed']}"
        lines.append(
            f"| {label} | {train_source}→{cross_source} | {self_metrics['auroc']:.4f} | "
            f"{cross_metrics['auroc']:.4f} | {degradation['auroc']:.4f} | "
            f"{self_metrics['brier']:.4f} | {cross_metrics['brier']:.4f} | "
            f"{degradation['brier']:.4f} | {agreement_auroc:.4f} | "
            f"{agreement_brier:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_adapter_transfer(config: dict[str, Any], force: bool = False) -> Path:
    root = output_dir(config) / "adapter_transfer"
    ensure_manifest(root, config_fingerprint(config), force=force)
    result_path = root / "metrics.json"
    done = root / "_SUCCESS.json"
    if done.exists() and result_path.exists() and not force:
        return result_path

    variants = config.get("adapter_transfer", {}).get("variants", config["probe"]["variants"])
    unknown = set(variants) - set(config["probe"]["variants"])
    if unknown:
        raise ValueError(
            "adapter_transfer.variants are not configured probe variants: "
            f"{sorted(unknown)}"
        )

    runs: list[dict[str, Any]] = []
    agreement_rate: float | None = None
    prediction_rows: list[dict[str, Any]] = []
    for feature_mode in configured_feature_keys(config):
        for variant in variants:
            for seed_value in config["probe"]["seeds"]:
                seed = int(seed_value)
                for train_source in ADAPTER_SOURCES:
                    model, _, records, features, test_masks = _fit_one(
                        config, feature_mode, variant, seed, train_source, force
                    )
                    agreed_ids, current_agreement_rate = _agreement_fact_ids(records)
                    agreement_rate = current_agreement_rate
                    correctness = np.asarray([record["correct"] for record in records], dtype=int)
                    device = next(model.parameters()).device
                    test_results = {}
                    for test_source, mask in test_masks.items():
                        indices = mask.nonzero(as_tuple=False).squeeze(1).numpy()
                        probabilities = _predict(
                            model,
                            features[mask],
                            device,
                            int(config["probe"]["batch_size"]),
                        )
                        labels = correctness[indices]
                        agree = np.asarray(
                            [records[index]["fact_id"] in agreed_ids for index in indices]
                        )
                        test_results[test_source] = {
                            "all": binary_error_metrics(labels, probabilities),
                            "label_agreement_only": binary_error_metrics(
                                labels[agree], probabilities[agree]
                            )
                            if agree.any()
                            else None,
                            "n_label_agreement": int(agree.sum()),
                        }
                        for index, probability in zip(indices, probabilities, strict=True):
                            record = records[index]
                            prediction_rows.append(
                                {
                                    "feature_mode": feature_mode,
                                    "variant": variant,
                                    "seed": seed,
                                    "train_source": train_source,
                                    "test_source": test_source,
                                    "fact_id": record["fact_id"],
                                    "correct": int(record["correct"]),
                                    "probability_error": float(probability),
                                    "adapter_correctness_agrees": record["fact_id"] in agreed_ids,
                                }
                            )
                    cross_source = "adapter_b" if train_source == "adapter_a" else "adapter_a"
                    agreement_self = test_results[train_source]["label_agreement_only"]
                    agreement_cross = test_results[cross_source]["label_agreement_only"]
                    runs.append(
                        {
                            "feature_mode": feature_mode,
                            "variant": variant,
                            "seed": seed,
                            "train_source": train_source,
                            "test": test_results,
                            "cross_adapter_degradation": _metric_degradation(
                                test_results[train_source]["all"],
                                test_results[cross_source]["all"],
                            ),
                            "cross_adapter_degradation_label_agreement_only": (
                                _metric_degradation(agreement_self, agreement_cross)
                                if agreement_self is not None and agreement_cross is not None
                                else None
                            ),
                        }
                    )

    payload = {
        "interpretation": (
            "adapter-induced representation/distribution shift, not pure domain drift"
        ),
        "selection_policy": "training and early stopping use only the training adapter",
        "test_policy": "same held-out entities are evaluated under both adapters",
        "correctness_agreement_rate": agreement_rate,
        "runs": runs,
        "seed_aggregate": _aggregate(runs),
        **artifact_metadata(),
    }
    atomic_json(result_path, payload)
    atomic_jsonl(root / "predictions.jsonl", prediction_rows)
    _write_report(root / "summary.md", payload)
    mark_complete(done, {"num_runs": len(runs), "config_fingerprint": config_fingerprint(config)})
    return result_path
