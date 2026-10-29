import runpy
from pathlib import Path

import pytest
import yaml

from peft_probe import factual_shift as fs

create = runpy.run_path("scripts/create_popqa_training_config.py")["create"]


def test_popqa_subjects_and_duplicate_questions_do_not_cross_splits_despite_shared_answers():
    items = [
        {
            "fact_id": str(i),
            "question": f"Question {i}",
            "answer": "United States",
            "aliases": ["United States"],
            "subject_id": f"s{i // 2}",
        }
        for i in range(40)
    ]
    items[2]["question"] = items[0]["question"]
    rows = fs.split_rows(items, dict.fromkeys(fs.SPLITS, 4), 43, grouping="subject")
    assert len(rows) == 24
    subjects, questions = {}, {}
    for row in rows:
        assert subjects.setdefault(row["subject_id"], row["split"]) == row["split"]
        assert questions.setdefault(row["question"], row["split"]) == row["split"]
    assert len({row["split"] for row in rows}) == 6


def test_popqa_loader_preserves_aliases_subject_zero_and_grades_containment():
    rows = fs.popqa_training_items(
        [
            {
                "id": 1,
                "question": "Where?",
                "obj": "Paris",
                "possible_answers": '["Paris", "City of Paris"]',
                "subj_id": 0,
            }
        ]
    )
    assert rows[0]["subject_id"] == "0"
    assert rows[0]["aliases"] == ["Paris", "City of Paris"]
    assert fs._grade("It is Paris.", rows[0])[0] == 1
    assert fs._grade("Parisian", rows[0])[0] == 0
    assert fs._grade("", rows[0])[0] == 0


def test_new_popqa_training_config_reuses_base_and_optimization_but_not_adapters(tmp_path):
    config = yaml.safe_load(Path("configs/factual_shift.yaml").read_text())
    config["experiment"]["output_dir"] = str(tmp_path / "old")
    config["experiment"]["source_output_dir"] = str(tmp_path / "old_adapters")
    config["model"]["name_or_path"] = "/cluster/llama2-7b"
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(config))
    original = source.read_bytes()
    target, output = tmp_path / "new.yaml", str(tmp_path / "new")
    create(source, target, output, 43)
    new = yaml.safe_load(target.read_text())
    assert new["model"] == config["model"]
    assert new["lora"] == config["lora"]
    assert new["factual_shift"]["split_sizes"] == config["factual_shift"]["split_sizes"]
    assert new["factual_shift"]["dataset"] == "akariasai/PopQA"
    assert new["factual_shift"]["split_grouping"] == "subject"
    assert new["factual_shift"]["ood_max_examples"] == 0
    assert new["factual_shift"]["length_adjustment"] is True
    assert "source_output_dir" not in new["experiment"]
    assert source.read_bytes() == original
    fs.validate(new)
    with pytest.raises(FileExistsError):
        create(source, target, output, 43)
    with pytest.raises(ValueError, match="separate"):
        create(source, tmp_path / "unsafe.yaml", str(tmp_path / "old" / "child"), 43)
