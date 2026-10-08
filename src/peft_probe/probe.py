from __future__ import annotations

import copy
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .collect import load_feature_shards
from .config import config_fingerprint, output_dir
from .io_utils import atomic_json, atomic_jsonl, atomic_torch_save, ensure_manifest, mark_complete, seed_everything
from .metrics import binary_error_metrics, risk_coverage
from .versioning import artifact_metadata


class Probe(nn.Module):
    def __init__(self, input_dim: int, architecture: str, hidden_dim: int):
        super().__init__()
        if architecture == "linear":
            # A one-dimensional shared linear direction keeps correctness linear while
            # allowing genuine/shuffled exposure supervision to affect that direction.
            self.trunk = nn.Linear(input_dim, 1)
            self.correctness_head = nn.Identity()
            self.exposure_head = nn.Linear(1, 1)
        elif architecture == "mlp":
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.correctness_head = nn.Linear(hidden_dim, 1)
            self.exposure_head = nn.Linear(hidden_dim, 1)
        else:
            raise ValueError(f"Unknown probe architecture: {architecture}")

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.trunk(features)
        return self.correctness_head(representation).squeeze(-1), self.exposure_head(representation).squeeze(-1)


def _load_all(config: dict[str, Any], feature_mode: str) -> tuple[list[dict[str, Any]], torch.Tensor]:
    root = output_dir(config) / "features" / "synthetic"
    records: list[dict[str, Any]] = []
    features: list[torch.Tensor] = []
    for source in config["collection"]["sources"]:
        source_records, source_features = load_feature_shards(root / source, feature_mode)
        records.extend(source_records)
        features.append(source_features)
    return records, torch.cat(features, dim=0)


def _shuffled_exposure(records: list[dict[str, Any]], seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    labels = np.asarray([record["exposure"] for record in records], dtype=np.float32)
    groups: defaultdict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record["exposure"] >= 0:
            groups[(record["source"], record["relation"], record["split"])].append(index)
    for indices in groups.values():
        values = labels[indices].copy()
        rng.shuffle(values)
        labels[indices] = values
    return torch.from_numpy(labels)


def _standardize(features: torch.Tensor, train_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = features[train_mask].mean(dim=0)
    std = features[train_mask].std(dim=0).clamp_min(1e-6)
    return (features - mean) / std, mean, std


@torch.inference_mode()
def _predict(model: Probe, features: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    outputs: list[torch.Tensor] = []
    model.eval()
    for (batch,) in DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False):
        correctness, _ = model(batch.to(device))
        outputs.append((1.0 - torch.sigmoid(correctness)).cpu())
    return torch.cat(outputs).numpy()


def train_one(
    config: dict[str, Any],
    feature_mode: str,
    variant: str,
    seed: int,
    force: bool = False,
) -> Path:
    settings = config["probe"]
    if variant not in settings["variants"]:
        raise ValueError(f"Unknown probe variant: {variant}")
    target = output_dir(config) / "probes" / feature_mode / variant / f"seed_{seed}"
    ensure_manifest(target, config_fingerprint(config), force=force)
    done = target / "_SUCCESS.json"
    if done.exists() and not force:
        return target

    seed_everything(seed)
    records, raw_features = _load_all(config, feature_mode)
    split = np.asarray([record["split"] for record in records])
    train_mask = torch.from_numpy(split == "train")
    validation_mask = torch.from_numpy(split == "validation")
    test_mask = torch.from_numpy(split == "test")
    features, mean, std = _standardize(raw_features.float(), train_mask)
    correctness = torch.tensor([record["correct"] for record in records], dtype=torch.float32)
    abstained = torch.tensor([record["abstained"] for record in records], dtype=torch.bool)
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
    model = Probe(features.shape[1], settings["architecture"], int(settings["hidden_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    # Unweighted BCE is a proper scoring rule. Class weighting would preserve ranking but
    # distort the sigmoid probabilities used for Brier score and calibration error.
    correctness_loss = nn.BCEWithLogitsLoss()
    exposure_loss = nn.BCEWithLogitsLoss()
    lambda_exposure = 0.0 if variant == "correctness" else float(settings["lambda_exposure"])

    checkpoint = target / "last_checkpoint.pt"
    start_epoch = 0
    best_validation = math.inf
    best_state = None
    patience_left = int(settings["patience"])
    history: list[dict[str, float]] = []
    if checkpoint.exists() and not force:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(device)
        start_epoch = state["epoch"] + 1
        best_validation = state["best_validation"]
        best_state = state["best_state"]
        patience_left = state["patience_left"]
        history = state["history"]

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
                    exposure_logits[batch_auxiliary_mask], batch_exposure[batch_auxiliary_mask]
                )
            loss.backward()
            optimizer.step()

        probabilities = _predict(model, features[validation_mask], device, int(settings["batch_size"]))
        labels = correctness[validation_mask].numpy().astype(int)
        validation_metrics = binary_error_metrics(labels, probabilities)
        validation_value = validation_metrics["brier"]
        history.append({"epoch": epoch, **validation_metrics})
        if validation_value < best_validation:
            best_validation = validation_value
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
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
        raise RuntimeError("Probe training did not produce a checkpoint")
    model.load_state_dict(best_state)
    target.mkdir(parents=True, exist_ok=True)
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
            **artifact_metadata(),
        },
    )
    atomic_jsonl(target / "training_history.jsonl", history)

    all_metrics: dict[str, Any] = {}
    for split_name, mask in (("validation", validation_mask), ("test", test_mask)):
        probability_error = _predict(model, features[mask], device, int(settings["batch_size"]))
        labels = correctness[mask].numpy().astype(int)
        attempted = (~abstained[mask]).numpy()
        all_metrics[split_name] = {
            "all_answers": binary_error_metrics(labels, probability_error),
            "attempted_only": binary_error_metrics(labels[attempted], probability_error[attempted])
            if attempted.any() else None,
            "abstention_rate": float(abstained[mask].float().mean()),
        }
        curve = risk_coverage(labels, probability_error)
        atomic_jsonl(
            target / f"risk_coverage_{split_name}.jsonl",
            (
                {
                    "coverage": float(coverage),
                    "selective_accuracy": float(accuracy),
                    "selective_risk": float(risk),
                }
                for coverage, accuracy, risk in zip(
                    curve["coverage"], curve["selective_accuracy"], curve["selective_risk"], strict=True
                )
            ),
        )
    atomic_json(target / "synthetic_metrics.json", all_metrics)
    mark_complete(
        done,
        {
            "feature_mode": feature_mode,
            "variant": variant,
            "seed": seed,
            "best_validation_brier": best_validation,
            "config_fingerprint": config_fingerprint(config),
        },
    )
    return target


def train_all(config: dict[str, Any], force: bool = False) -> list[Path]:
    outputs: list[Path] = []
    for feature_mode in config["collection"]["feature_modes"]:
        for variant in config["probe"]["variants"]:
            for seed in config["probe"]["seeds"]:
                outputs.append(train_one(config, feature_mode, variant, int(seed), force=force))
    return outputs


def load_probe(path: str | Path, device: torch.device | None = None) -> tuple[Probe, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Probe(payload["input_dim"], payload["architecture"], payload["hidden_dim"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload
