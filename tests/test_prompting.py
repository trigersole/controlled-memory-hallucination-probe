from peft_probe.prompting import apply_instruction, synthetic_eval_question, trivia_question


def _config():
    return {
        "prompting": {
            "instruction": "Answer the question concisely",
            "synthetic_template_family": "concise_diverse_v1",
        }
    }


def test_shared_concise_instruction_is_applied_to_synthetic_and_trivia():
    config = _config()
    row = {
        "fact_id": "f000001",
        "entity": "Lummor Zanhel",
        "relation": "birthplace",
        "detector_split": "train",
        "eval_question": "legacy question",
    }

    synthetic = synthetic_eval_question(config, row)
    trivia = trivia_question(config, "Where was Ada Lovelace born?")

    assert synthetic.startswith("Answer the question concisely. ")
    assert trivia == "Answer the question concisely. Where was Ada Lovelace born?"


def test_detector_splits_use_disjoint_template_families():
    config = _config()
    questions = set()
    for split in ("train", "validation", "test"):
        row = {
            "fact_id": "f000001",
            "entity": "Lummor Zanhel",
            "relation": "occupation",
            "detector_split": split,
            "eval_question": "legacy question",
        }
        questions.add(synthetic_eval_question(config, row))

    assert len(questions) == 3


def test_legacy_prompt_behavior_is_unchanged_without_prompting_config():
    assert apply_instruction({}, "What is the answer?") == "What is the answer?"
    assert trivia_question({}, "Who wrote Hamlet?") == (
        "Answer with only the short answer. Who wrote Hamlet?"
    )
