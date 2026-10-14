from __future__ import annotations

import hashlib
from typing import Any


CONCISE_DIVERSE_TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {
    "birthplace": {
        "train": (
            "Where was {entity} born?",
            "What is the birthplace of {entity}?",
        ),
        "validation": (
            "{entity} was born in what place?",
            "Name the place where {entity} was born.",
        ),
        "test": (
            "In which location did the birth of {entity} occur?",
            "What place is {entity} originally from?",
        ),
    },
    "occupation": {
        "train": (
            "What is the occupation of {entity}?",
            "What kind of work does {entity} do?",
        ),
        "validation": (
            "{entity} works in what profession?",
            "Name the profession practiced by {entity}.",
        ),
        "test": (
            "What is {entity}'s line of work?",
            "Which job is associated with {entity}?",
        ),
    },
    "organization": {
        "train": (
            "Which organization is {entity} affiliated with?",
            "What organization does {entity} belong to?",
        ),
        "validation": (
            "{entity} is a member of which group?",
            "Name the organization associated with {entity}.",
        ),
        "test": (
            "What is the organizational affiliation of {entity}?",
            "Which group counts {entity} as a member?",
        ),
    },
}


def _instruction(config: dict[str, Any]) -> str | None:
    value = config.get("prompting", {}).get("instruction")
    if value is None:
        return None
    instruction = str(value).strip()
    if not instruction:
        raise ValueError("prompting.instruction cannot be empty")
    return instruction


def apply_instruction(config: dict[str, Any], question: str) -> str:
    instruction = _instruction(config)
    if instruction is None:
        return question
    punctuation = "" if instruction.endswith((".", "!", "?")) else "."
    return f"{instruction}{punctuation} {question.strip()}"


def synthetic_eval_question(config: dict[str, Any], row: dict[str, Any]) -> str:
    family = config.get("prompting", {}).get("synthetic_template_family")
    if family is None:
        return apply_instruction(config, str(row["eval_question"]))
    if family != "concise_diverse_v1":
        raise ValueError(f"Unknown prompting.synthetic_template_family: {family}")
    relation = str(row["relation"])
    if relation not in CONCISE_DIVERSE_TEMPLATES:
        raise ValueError(f"No concise-diverse templates for relation {relation}")
    split = str(row["detector_split"])
    family_split = split if split in {"train", "validation", "test"} else "test"
    templates = CONCISE_DIVERSE_TEMPLATES[relation][family_split]
    digest = hashlib.sha256(str(row["fact_id"]).encode()).digest()
    template = templates[int.from_bytes(digest[:4], "big") % len(templates)]
    return apply_instruction(config, template.format(entity=row["entity"]))


def trivia_question(config: dict[str, Any], question: str) -> str:
    if _instruction(config) is None:
        return "Answer with only the short answer. " + question
    return apply_instruction(config, question)


def truthful_generation_question(config: dict[str, Any], question: str) -> str:
    if _instruction(config) is None:
        return "Give a brief, factual answer. " + question
    return apply_instruction(config, question)
