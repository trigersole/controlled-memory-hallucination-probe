import runpy
from pathlib import Path

import pytest
import yaml

from peft_probe import factual_shift as fs

create = runpy.run_path("scripts/create_truthfulqa_training_config.py")["create"]


def test_truthfulqa_cleaning_excludes_conflicting_labels_and_replaces_invalid_canonical():
    rows = [
        {
            "question": "Q1",
            "best_answer": "conflict",
            "correct_answers": ["clean", "conflict"],
            "incorrect_answers": ["wrong", "conflict"],
            "source": "https://source.test/a",
        },
        {
            "question": "Q2",
            "best_answer": "yes",
            "correct_answers": ["yes"],
            "incorrect_answers": ["yes!"],
            "source": "https://source.test/b",
        },
    ]
    items, audit = fs.truthful_training_items(rows)
    assert len(items) == 1
    assert items[0]["answer"] == "clean"
    assert items[0]["incorrect_answers"] == ["wrong"]
    assert audit["n_requested"] == 2 and audit["n_excluded"] == 1
    assert audit["excluded"][0]["fact_id"] == "truthful_1"


def test_question_source_groups_do_not_cross_splits_despite_generic_shared_answers():
    items = [
        {
            "fact_id": str(i),
            "question": f"Question {i}",
            "answer": "No comment",
            "aliases": ["No comment"],
            "source_url": f"https://source.test/{i // 2}",
        }
        for i in range(40)
    ]
    rows = fs.split_rows(items, dict.fromkeys(fs.SPLITS, 4), 43, grouping="question_source")
    assert len(rows) == 24
    groups = {}
    for row in rows:
        assert groups.setdefault(row["source_url"], row["split"]) == row["split"]


def test_truthfulqa_config_has_feasible_budget_and_no_generated_reference_mixture(tmp_path):
    original = yaml.safe_load(Path("configs/factual_shift.yaml").read_text())
    original["experiment"]["output_dir"] = str(tmp_path / "old")
    original["model"]["name_or_path"] = "/cluster/llama2-7b"
    source, target = tmp_path / "old.yaml", tmp_path / "new.yaml"
    source.write_text(yaml.safe_dump(original))
    before = source.read_bytes()
    create(source, target, str(tmp_path / "new"), 43)
    config = yaml.safe_load(target.read_text())
    settings = config["factual_shift"]
    assert config["model"] == original["model"] and config["lora"] == original["lora"]
    assert settings["dataset"] == "truthfulqa/truthful_qa"
    assert sum(settings["split_sizes"].values()) == 600
    assert settings["split_sizes"]["test"] == 200
    assert (
        settings["candidate_mode"] == "references" and settings["negative_source"] == "references"
    )
    assert settings["ood_max_examples"] == 0
    assert settings["length_adjustment"] is True
    assert source.read_bytes() == before
    fs.validate(config)
    with pytest.raises(FileExistsError):
        create(source, target, str(tmp_path / "new"), 43)
    with pytest.raises(ValueError, match="separate"):
        create(source, tmp_path / "unsafe.yaml", str(tmp_path / "old/child"), 43)
