import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from peft_probe import factual_popqa as pop
from peft_probe import factual_shift as fs
from peft_probe.config import config_fingerprint
from peft_probe.within_question import _sha


class Probe:
    def predict_proba(self, values):
        return np.tile([0.4, 0.6], (len(values), 1))


def artifacts():
    primary = pop.PRIMARY
    wrong = primary.replace("/correct_a/", "/incorrect/")
    pca = "-17/activation_pca/energy_1"
    names = [
        primary,
        wrong,
        "base_likelihood",
        "answer_length_words",
        pca,
        "-17/supervised_probe",
        "-17/random/base",
    ]
    fit = {"basis": np.eye(16)[:, :8], "origin": np.zeros(16), "mean_direction": np.eye(16)[0]}
    return {
        "selected": primary,
        "selected_transfer": primary,
        "selection_rule": "predeclared_method_validation_sign",
        "selected_by_family": {"incorrect": wrong, "activation_pca": pca},
        "signs": dict.fromkeys(names, 1),
        **{f"-17/{source}": copy.deepcopy(fit) for source in (*fs.ADAPTERS, "activation_pca")},
        "-17/supervised_probe": Probe(),
    }


def source_rows():
    return [
        {
            "row_id": f"{split}_{i}",
            "fact_id": f"{split}_{i // 2}",
            "group_id": f"{split}_{i // 2}",
            "question": f"source question {i // 2}",
            "split": split,
            "correct": i % 2,
            "answer": "long correct answer" if i % 2 else "wrong",
        }
        for split in ("probe_train", "validation", "test")
        for i in range(12)
    ]


def test_question_sampling_excludes_exact_source_overlap_and_groups_subjects():
    items = [
        {
            "id": i,
            "question": f"question {i}",
            "obj": "Paris",
            "possible_answers": '["Paris"]',
            "subj_id": i // 2,
        }
        for i in range(8)
    ]
    rows, rejected = pop.choose_questions(items, {"question 0"}, 20, 4)
    assert len(rows) == 7
    assert rejected["source_question_overlap"] == 1
    assert next(r for r in rows if r["fact_id"] == "popqa_2")["group_id"] == "subject_1"
    assert rows == pop.choose_questions(items, {"question 0"}, 20, 4)[0]


def test_frozen_scores_apply_source_signs_and_transfer_basis_without_refitting(monkeypatch):
    a = artifacts()
    a["signs"][pop.PRIMARY] = -1
    base = np.zeros((2, 16))
    update = np.zeros((2, 16))
    update[0, 0], update[1, 8] = 1, 1
    features = {"base": {"-17": base}, **{source: {"-17": update.copy()} for source in fs.ADAPTERS}}
    config = {"experiment": {"seed": 43}, "factual_shift": {"ranks": [1, 8]}}
    monkeypatch.setattr(
        fs, "fit_directions", lambda *args: pytest.fail("No fitting on target data")
    )
    rows = [{"answer": "correct"}, {"answer": "wrong answer"}]
    scores = pop.frozen_scores(config, a, features, np.array([0.5, 0.2]), rows)
    assert scores[pop.PRIMARY].tolist() == [-1, 0]
    assert scores[pop.PRIMARY + "/a_to_b"].tolist() == [-1, 0]
    assert scores[pop.PRIMARY.replace("/correct_a/", "/incorrect/")].tolist() == [1, 0]
    assert scores["answer_length_words"].tolist() == [1, 2]


def test_length_heads_never_use_validation_test_or_target_labels():
    config = {"experiment": {"seed": 43}}
    rows = source_rows()
    fitted = {
        "artifacts": artifacts(),
        "scores": {pop.PRIMARY: np.array([r["correct"] for r in rows], dtype=float)},
    }
    settings = {"length_probe_C": 1}
    heads = pop.length_heads(config, fitted, rows, settings)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["split"] != "probe_train":
            row["correct"] = 1 - row["correct"]
    other = pop.length_heads(config, fitted, changed, settings)
    for name in heads:
        np.testing.assert_array_equal(heads[name][-1].coef_, other[name][-1].coef_)


def test_popqa_end_to_end_reuses_pool_base_features_and_original_adapters(tmp_path, monkeypatch):
    import datasets
    import peft

    template = yaml.safe_load(Path("configs/factual_shift.yaml").read_text())
    template["model"]["name_or_path"] = "local/llama-test"
    template["factual_shift"].update(layers=[-17], ranks=[1, 8], shard_size=2, batch_size=2)
    originals, configs, runs = source_rows(), [], []
    for seed in (43, 44, 45):
        config = copy.deepcopy(template)
        config["experiment"].update(seed=seed, output_dir=str(tmp_path / f"original_{seed}"))
        path = tmp_path / f"seed_{seed}.yaml"
        path.write_text(yaml.safe_dump(config))
        runs.append(
            {
                "config": str(path),
                "config_sha256": _sha(path),
                "seed": seed,
                "config_fingerprint": config_fingerprint(config),
            }
        )
        configs.append(config)
        root = fs.root(config)
        fs.atomic_jsonl(root / "data/rows.jsonl", originals)
        for source in fs.ADAPTERS:
            directory = root / "adapters" / source / "final"
            fs._complete(directory, {})
            (directory / "adapter_model.safetensors").write_bytes(b"existing weights")
        fs._complete(root / "results", {})
        signature = {
            "checkpoint_version": 1,
            "config_fingerprint": config_fingerprint(config),
            "rows_sha256": fs._digest(originals),
        }
        fitted = {
            "artifacts": artifacts(),
            "scores": {pop.PRIMARY: np.array([r["correct"] for r in originals], dtype=float)},
        }
        fs.atomic_torch_save(
            root / "results/analysis_fit.pt", {"signature": signature, "fitted": fitted}
        )
        fs.atomic_json(
            root / "results/metrics.json",
            {
                "rows_sha256": signature["rows_sha256"],
                "methods": {m: {"orientation": s} for m, s in artifacts()["signs"].items()},
            },
        )
    manifest = tmp_path / "study.json"
    manifest.write_text(json.dumps({"runs": runs}))
    settings = {
        "schema_version": 1,
        "study_manifest": str(manifest),
        "output_dir": str(tmp_path / "popqa"),
        "dataset": "akariasai/PopQA",
        "dataset_revision": "main",
        "split": "test",
        "max_questions": 3,
        "sampling_seed": 100,
        "bootstrap_samples": 10,
        "length_probe_C": 1,
    }
    monkeypatch.setattr(pop, "load_settings", lambda: settings)
    monkeypatch.setattr(fs, "_replay_rows", lambda config: originals)
    for name in ("train", "fit_directions", "_fit_analysis"):
        monkeypatch.setattr(
            fs, name, lambda *args, **kwargs: pytest.fail("Frozen evaluation must not train/refit")
        )
    data = [
        {
            "id": i,
            "question": f"Pop question {i}",
            "obj": f"Gold {i}",
            "possible_answers": json.dumps([f"Gold {i}"]),
            "subj_id": i,
            "subj": f"S{i}",
        }
        for i in range(4)
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: data)
    loads, generations = [], []

    class Model:
        source = "base"

        def eval(self):
            return self

    def load_model(config):
        loads.append("model")
        return Model()

    def attach(model, path, **kwargs):
        assert kwargs["is_trainable"] is False
        model.source = path.parent.name
        return model

    def generate(model, tokenizer, rows, settings, sample):
        assert model.source == "base"
        generations.append(len(rows))
        return ["wrong output" if sample else row["answer"] for row in rows]

    def replay(model, tokenizer, rows, settings):
        matrix = np.zeros((len(rows), 16), dtype=np.float32)
        if model.source != "base":
            for i, row in enumerate(rows):
                matrix[i, 0 if row["correct"] else 8] = 1
        return {"-17": torch.from_numpy(matrix)}, [float(row["correct"]) for row in rows]

    monkeypatch.setattr(fs, "load_base_model", load_model)
    monkeypatch.setattr(fs, "load_tokenizer", lambda config: None)
    monkeypatch.setattr(fs, "_generate", generate)
    monkeypatch.setattr(fs, "replay", replay)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", attach)
    before = {p: p.read_bytes() for c in configs for p in fs.root(c).rglob("*") if p.is_file()}
    result = pop.run(configs[0])
    assert (result / "_SUCCESS.json").exists()
    report = json.loads((result / "metrics.json").read_text())
    assert report["popqa"]["n_mixed_questions"] == 3
    assert report["popqa"]["methods"][pop.PRIMARY]["within_question_pairwise_accuracy"] == 1
    first_generation_count = len(generations)
    first_loads = len(loads)
    pop.run(configs[1])
    assert len(generations) == first_generation_count
    assert (
        len(loads) == first_loads + 3
    )  # Only the three frozen adapters; shared base replay is reused.
    pop.run(configs[0])
    assert len(loads) == first_loads + 3
    assert all(path.read_bytes() == value for path, value in before.items())
    settings_file = tmp_path / "external.yaml"
    settings_file.write_text(yaml.safe_dump(settings))
    summarize = runpy.run_path("scripts/summarize_factual_popqa.py")["summarize"]
    with pytest.raises(FileNotFoundError):
        summarize(settings_file)
    pop.run(configs[2])
    summary = summarize(settings_file)
    assert (summary / "_SUCCESS.json").exists()
    assert "Source test: length-plus-shift" in (summary / "summary.md").read_text()
    second_result = Path(settings["output_dir"]) / "seed_44/results"
    altered = json.loads((second_result / "metrics.json").read_text())
    altered["signature"]["rows_sha256"] = "different candidate pool"
    fs.atomic_json(second_result / "metrics.json", altered)
    fs.atomic_json(
        second_result / "_SUCCESS.json", {"complete": True, "signature": altered["signature"]}
    )
    with pytest.raises(ValueError, match="same saved PopQA"):
        summarize(settings_file)
    # Changed source metadata must not silently reuse target scores.
    fs.atomic_json(fs.root(configs[0]) / "results/metrics.json", {"rows_sha256": "changed"})
    with pytest.raises(ValueError, match="checkpoint"):
        pop.run(configs[0])
