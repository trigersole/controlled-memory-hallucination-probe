from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from .config import config_fingerprint, output_dir
from .io_utils import ensure_manifest, mark_complete, read_jsonl, seed_everything
from .modeling import load_base_model, load_tokenizer, user_prompt


def _last_valid_checkpoint(checkpoint_dir: Path) -> str | None:
    """Return the newest complete Trainer/PEFT checkpoint, ignoring interrupted saves."""
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((step, path))

    weight_names = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    )
    for _, path in sorted(candidates, reverse=True):
        has_weights = any((path / name).is_file() for name in weight_names)
        if has_weights and (path / "trainer_state.json").is_file():
            return str(path)
    return None


class FactDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.examples: list[dict[str, torch.Tensor]] = []
        for row in rows:
            for question in row["train_questions"]:
                prompt = user_prompt(tokenizer, question)
                answer = f" {row['value']}{tokenizer.eos_token or ''}"
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                full = tokenizer(
                    prompt + answer,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )
                labels = list(full["input_ids"])
                masked = min(len(prompt_ids), len(labels))
                labels[:masked] = [-100] * masked
                self.examples.append(
                    {
                        "input_ids": torch.tensor(full["input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(full["attention_mask"], dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        width = max(item["input_ids"].numel() for item in examples)
        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in examples:
            padding = width - item["input_ids"].numel()
            batch["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (padding, 0), value=self.pad_token_id))
            batch["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (padding, 0), value=0))
            batch["labels"].append(torch.nn.functional.pad(item["labels"], (padding, 0), value=-100))
        return {key: torch.stack(values) for key, values in batch.items()}


def train(config: dict[str, Any], adapter: str, force: bool = False) -> Path:
    adapter = adapter.lower().replace("adapter_", "")
    if adapter not in {"a", "b"}:
        raise ValueError("adapter must be A or B")
    root = output_dir(config)
    adapter_root = root / "adapters" / f"adapter_{adapter}"
    ensure_manifest(adapter_root, config_fingerprint(config), force=force)
    final_dir = adapter_root / "final"
    done = final_dir / "_SUCCESS.json"
    if done.exists() and not force:
        return final_dir

    seed = int(config["experiment"]["seed"]) + (0 if adapter == "a" else 1)
    seed_everything(seed)
    tokenizer = load_tokenizer(config["model"])
    rows = [
        row for row in read_jsonl(root / "data" / "facts.jsonl")
        if row["adapter_assignment"] == adapter
    ]
    dataset = FactDataset(rows, tokenizer, int(config["model"]["max_sequence_length"]))
    model = load_base_model(config["model"], training=True)
    if bool(getattr(model, "is_loaded_in_4bit", False)):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            bias="none",
        ),
    )
    checkpoint_dir = adapter_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    requested_dtype = config["model"].get("dtype")
    use_bf16 = (
        requested_dtype == "bfloat16"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = torch.cuda.is_available() and (
        requested_dtype == "float16" or (requested_dtype == "bfloat16" and not use_bf16)
    )
    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=float(lora["epochs"]),
        learning_rate=float(lora["learning_rate"]),
        per_device_train_batch_size=int(lora["per_device_batch_size"]),
        gradient_accumulation_steps=int(lora["gradient_accumulation_steps"]),
        save_steps=int(lora["save_steps"]),
        save_strategy="steps",
        # Keep resume safety without retaining many large optimizer checkpoints.
        save_total_limit=2,
        logging_steps=int(lora["logging_steps"]),
        logging_strategy="steps",
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_8bit" if bool(config["model"].get("load_in_4bit", False)) else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    resume = None if force else _last_valid_checkpoint(checkpoint_dir)
    trainer.train(resume_from_checkpoint=resume)
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    mark_complete(
        done,
        {
            "adapter": adapter,
            "num_facts": len(rows),
            "num_training_examples": len(dataset),
            "config_fingerprint": config_fingerprint(config),
        },
    )
    return final_dir
