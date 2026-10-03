from __future__ import annotations

import warnings
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def load_tokenizer(model_config: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(model_config: dict[str, Any], training: bool = False):
    use_4bit = bool(model_config.get("load_in_4bit", False))
    if use_4bit and not torch.cuda.is_available():
        warnings.warn("CUDA is unavailable; disabling 4-bit loading for this run")
        use_4bit = False
    dtype = torch_dtype(model_config.get("dtype", "bfloat16"))
    if not torch.cuda.is_available():
        dtype = torch.float32
    elif dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        warnings.warn("This GPU does not support bfloat16; falling back to float16")
        dtype = torch.float16
    quantization_config = None
    device_map = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        device_map = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map=device_map,
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
    )
    if not use_4bit and torch.cuda.is_available():
        model = model.to(torch.cuda.current_device())
    model.config.use_cache = not training
    return model


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def user_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Question: {question}\nAnswer:"
