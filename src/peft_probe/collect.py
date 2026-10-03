from __future__ import annotations

import math
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

from .config import config_fingerprint, output_dir
from .io_utils import atomic_torch_save, chunks, ensure_manifest, mark_complete, read_jsonl, seed_everything
from .modeling import load_base_model, load_tokenizer, model_device, user_prompt


ABSTENTION_PATTERNS = (
    "i don't know", "i do not know", "unknown", "cannot determine", "can't determine",
    "not enough information", "no information",
)


def normalize(text: str) -> str:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def grade_synthetic(answer: str, expected: str) -> tuple[int, int]:
    normalized_answer = normalize(answer)
    abstained = int(any(pattern in normalized_answer for pattern in ABSTENTION_PATTERNS))
    normalized_expected = normalize(expected)
    correct = int(bool(normalized_expected) and normalized_expected in normalized_answer and not abstained)
    return correct, abstained


def _pool_hidden(hidden: torch.Tensor, token_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    # hidden: [batch, sequence, width]; token_mask: [batch, sequence]
    if pooling == "mean":
        denominator = token_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (hidden * token_mask.unsqueeze(-1)).sum(dim=1) / denominator
    if pooling == "last":
        indices = token_mask.long().sum(dim=1).sub(1).clamp_min(0)
        # Answer tokens are a contiguous suffix, so total nonpadding count identifies the last token.
        absolute = token_mask.long().cumsum(dim=1).eq(indices.add(1).unsqueeze(1)).long().argmax(dim=1)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), absolute]
    raise ValueError(f"Unknown pooling method: {pooling}")


def _adapter_context(model, disabled: bool):
    if disabled and isinstance(model, PeftModel):
        return model.disable_adapter()
    return nullcontext()


@torch.inference_mode()
def _batch_generate_and_extract(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    source: str,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    prompts = [user_prompt(tokenizer, row["eval_question"]) for row in rows]
    tokens = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    device = model_device(model)
    tokens = {key: value.to(device) for key, value in tokens.items()}
    input_width = tokens["input_ids"].shape[1]
    generated = model.generate(
        **tokens,
        max_new_tokens=int(settings["max_new_tokens"]),
        do_sample=bool(settings.get("do_sample", False)),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequences = generated.sequences
    answer_ids = sequences[:, input_width:]
    answer_mask = answer_ids.ne(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        answer_mask &= answer_ids.ne(tokenizer.eos_token_id)

    answers = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
    logprob_sum = torch.zeros(len(rows), device=device)
    entropy_sum = torch.zeros(len(rows), device=device)
    score_count = torch.zeros(len(rows), device=device)
    for step, logits in enumerate(generated.scores):
        if step >= answer_ids.shape[1]:
            break
        active = answer_mask[:, step]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        chosen = answer_ids[:, step].unsqueeze(-1)
        logprob_sum += log_probs.gather(1, chosen).squeeze(1) * active
        entropy_sum += (-(probs * log_probs).sum(dim=-1)) * active
        score_count += active
    score_count = score_count.clamp_min(1)

    full_attention = sequences.ne(tokenizer.pad_token_id)
    content_mask = torch.zeros_like(full_attention)
    content_mask[:, input_width:] = answer_mask
    feature_tensors: dict[str, torch.Tensor] = {}
    feature_cache: dict[str, torch.Tensor] = {}
    layer = int(settings["layer"])
    for mode in settings["feature_modes"]:
        cache_key = "base" if source == "base" else mode
        if cache_key in feature_cache:
            feature_tensors[mode] = feature_cache[cache_key]
            continue
        disable = mode == "base_replay" and source != "base"
        with _adapter_context(model, disable):
            outputs = model(
                input_ids=sequences,
                attention_mask=full_attention,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden = outputs.hidden_states[layer]
        pooled = _pool_hidden(hidden, content_mask, settings["pooling"])
        feature_tensors[mode] = pooled.detach().float().cpu()
        feature_cache[cache_key] = feature_tensors[mode]

    records: list[dict[str, Any]] = []
    for index, (row, answer) in enumerate(zip(rows, answers, strict=True)):
        correct, abstained = grade_synthetic(answer, row["value"])
        if source == "base":
            exposure = -1
        else:
            adapter = source.removeprefix("adapter_")
            exposure = int(row["adapter_assignment"] == adapter)
        records.append(
            {
                "fact_id": row["fact_id"],
                "entity_id": row["entity_id"],
                "relation": row["relation"],
                "split": row["detector_split"],
                "source": source,
                "adapter_assignment": row["adapter_assignment"],
                "exposure": exposure,
                "question": row["eval_question"],
                "expected": row["value"],
                "answer": answer.strip(),
                "correct": correct,
                "abstained": abstained,
                "mean_logprob": float((logprob_sum[index] / score_count[index]).cpu()),
                "mean_entropy": float((entropy_sum[index] / score_count[index]).cpu()),
                "answer_tokens": int(answer_mask[index].sum().cpu()),
            }
        )
    return records, feature_tensors


def collect(config: dict[str, Any], source: str, force: bool = False) -> Path:
    if source not in set(config["collection"]["sources"]):
        raise ValueError(f"Unknown source {source}; choose from {config['collection']['sources']}")
    root = output_dir(config)
    target = root / "features" / "synthetic" / source
    ensure_manifest(target, config_fingerprint(config), force=force)
    done = target / "_SUCCESS.json"
    if done.exists() and not force:
        return target

    seed_everything(int(config["experiment"]["seed"]))
    tokenizer = load_tokenizer(config["model"])
    model = load_base_model(config["model"], training=False)
    if source.startswith("adapter_"):
        adapter_path = root / "adapters" / source / "final"
        if not (adapter_path / "_SUCCESS.json").exists():
            raise FileNotFoundError(f"Adapter is incomplete: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()

    rows = sorted(read_jsonl(root / "data" / "facts.jsonl"), key=lambda row: row["fact_id"])
    shard_size = int(config["collection"]["shard_size"])
    batch_size = int(config["collection"]["batch_size"])
    expected_shards = math.ceil(len(rows) / shard_size)
    for shard_index, shard_rows in chunks(rows, shard_size):
        shard_path = target / f"shard_{shard_index:05d}.pt"
        if shard_path.exists() and not force:
            continue
        all_records: list[dict[str, Any]] = []
        feature_parts: dict[str, list[torch.Tensor]] = {
            mode: [] for mode in config["collection"]["feature_modes"]
        }
        for _, batch_rows in chunks(shard_rows, batch_size):
            records, features = _batch_generate_and_extract(
                model, tokenizer, batch_rows, source, config["collection"]
            )
            all_records.extend(records)
            for mode, tensor in features.items():
                feature_parts[mode].append(tensor)
        atomic_torch_save(
            shard_path,
            {
                "records": all_records,
                "features": {mode: torch.cat(parts, dim=0) for mode, parts in feature_parts.items()},
                "source": source,
                "shard_index": shard_index,
                "config_fingerprint": config_fingerprint(config),
            },
        )

    actual_shards = len(list(target.glob("shard_*.pt")))
    if actual_shards != expected_shards:
        raise RuntimeError(f"Expected {expected_shards} shards for {source}, found {actual_shards}")
    mark_complete(
        done,
        {
            "source": source,
            "num_examples": len(rows),
            "num_shards": actual_shards,
            "config_fingerprint": config_fingerprint(config),
        },
    )
    return target


def load_feature_shards(directory: str | Path, feature_mode: str) -> tuple[list[dict[str, Any]], torch.Tensor]:
    records: list[dict[str, Any]] = []
    features: list[torch.Tensor] = []
    for path in sorted(Path(directory).glob("shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(payload["records"])
        features.append(payload["features"][feature_mode])
    if not features:
        raise FileNotFoundError(f"No feature shards found in {directory}")
    return records, torch.cat(features, dim=0)
