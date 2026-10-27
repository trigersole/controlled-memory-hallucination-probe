import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from peft_probe import factual_shift as fs
from peft_probe.config import load_config


def config(tmp_path):
    result = load_config("configs/factual_shift_smoke.yaml")
    result["experiment"]["output_dir"] = str(tmp_path)
    result["factual_shift"].update(layers=[-1], ranks=[1, 2], bootstrap_samples=5)
    return result


def test_alias_connected_groups_cannot_cross_splits():
    items = [{"question": f"Question {i}", "aliases": [f"answer {i}"]} for i in range(30)]
    items[1]["aliases"] = ["answer 0", "bridge"]
    items[2]["aliases"] = ["bridge"]
    items[3]["question"] = items[2]["question"]
    sizes = dict.fromkeys(fs.SPLITS, 2)
    rows = fs.split_rows(items, sizes, 7)
    assert rows == fs.split_rows(items, sizes, 7)
    assert len(rows) == 12
    assigned = {}
    for row in rows:
        for alias in row["aliases"]:
            assert assigned.setdefault(alias, row["split"]) == row["split"]
    assert (
        len(
            {
                r["split"]
                for r in rows
                if r["question"] in {"Question 0", "Question 1", "Question 2"}
            }
        )
        <= 1
    )


def test_repeated_question_ids_collapse_and_preserve_aliases():
    first = {
        "fact_id": "q1",
        "question": "Who wrote Hamlet?",
        "answer": "Shakespeare",
        "aliases": ["Shakespeare", "William Shakespeare"],
    }
    repeat = {
        **first,
        "question": "Who wrote Hamlet? ",
        "aliases": ["Shakespeare", "W. Shakespeare"],
    }
    rows = [first, repeat, {**first, "fact_id": "q2"}]
    unique = fs.deduplicate_questions(rows)
    assert [r["fact_id"] for r in unique] == ["q1", "q2"]
    assert unique[0]["aliases"] == ["Shakespeare", "William Shakespeare", "W. Shakespeare"]
    assert first["aliases"] == ["Shakespeare", "William Shakespeare"]
    assert unique == fs.deduplicate_questions(rows)


@pytest.mark.parametrize("change", [{"question": "Who wrote Macbeth?"}, {"answer": "Marlowe"}])
def test_reused_question_id_with_conflicting_content_fails(change):
    first = {
        "fact_id": "q1",
        "question": "Who wrote Hamlet?",
        "answer": "Shakespeare",
        "aliases": ["Shakespeare"],
    }
    with pytest.raises(ValueError, match="Conflicting TriviaQA records.*q1"):
        fs.deduplicate_questions([first, {**first, **change}])


def test_uncentered_svd_retains_constant_mean_shift():
    base = np.zeros((20, 4))
    adapted = base + np.array([0, 3, 0, 0])
    fitted = fs.fit_directions(base, adapted, [1], 42)
    assert abs(fitted["basis"][1, 0]) == pytest.approx(1)
    assert fitted["energy_fraction"][0] == pytest.approx(1)
    scores = fs.direction_scores(adapted, fitted, [1])
    assert scores["energy_1"] == pytest.approx(np.full(20, 9))
    with pytest.raises(ValueError, match="zero"):
        fs.fit_directions(base, base, [1], 42)


def test_validation_orientation_is_frozen_on_test():
    sign = fs.orient(np.array([0, 1]), np.array([2.0, -2.0]))
    assert sign == -1
    # A genuinely reversed test relationship must remain poor, not be flipped again.
    rows = [
        {"fact_id": str(i), "group_id": str(i), "correct": i, "candidate_index": 0} for i in (0, 1)
    ]
    report = fs.evaluate_scores(rows, sign * np.array([-2.0, 2.0]), [0.5], 5, 1)
    assert report["auroc"] == 0


def test_candidate_reranking_groups_by_question_and_keeps_ambiguous_outputs_visible():
    rows = [
        {
            "fact_id": str(i // 2),
            "group_id": str(i // 2),
            "correct": label,
            "candidate_index": i % 2,
        }
        for i, label in enumerate([0, 1, 1, 0, -1, -1])
    ]
    report = fs.evaluate_scores(rows, np.array([0, 1, 1, 0, 2, 3]), [0.5, 1.0], 5, 1)
    assert report["auroc"] == 1
    assert report["reranked_accuracy"] == 1
    assert report["reranked_graded_questions"] == 2
    assert report["greedy_accuracy"] == 0.5
    assert report["n_graded"] == 4


class Tokenizer:
    pad_token_id = 0
    chat_template = None

    def __call__(self, text, **kwargs):
        return {"input_ids": [len(token) % 15 + 1 for token in text.split()]}


class Model:
    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, input_ids, **kwargs):
        hidden = torch.stack([input_ids.float(), input_ids.float() * 2], dim=-1)
        logits = torch.zeros(*input_ids.shape, 20)
        return SimpleNamespace(hidden_states=(hidden,), logits=logits)


def test_replay_pools_answer_excludes_padding_and_preserves_float32():
    rows = [
        {"question": "Where?", "answer": "red fox", "row_id": "a"},
        {"question": "Where was the person born?", "answer": "Rome", "row_id": "b"},
        {"question": "Who?", "answer": "", "row_id": "c"},
    ]
    settings = {"layers": [-1], "representation": "answer_last", "max_sequence_length": 100}
    features, scores = fs.replay(Model(), Tokenizer(), rows, settings)
    assert features["-1"].dtype == torch.float32
    assert features["-1"][:2, 0].tolist() == [4, 5]
    assert scores[:2] == pytest.approx([-np.log(20), -np.log(20)])
    assert scores[2] == -1e6
    settings["representation"] = "answer_mean"
    features, _ = fs.replay(Model(), Tokenizer(), rows, settings)
    assert features["-1"][0, 0] == 4
    settings["max_sequence_length"] = 1
    with pytest.raises(ValueError, match="exceeds"):
        fs.replay(Model(), Tokenizer(), rows, settings)


def test_factual_training_rejects_partial_answer_truncation():
    from peft_probe.train_adapter import FactDataset

    class TrainingTokenizer(Tokenizer):
        eos_token = ""

        def __call__(self, text, **kwargs):
            ids = super().__call__(text)["input_ids"]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    with pytest.raises(ValueError, match="Training input exceeds"):
        FactDataset(
            [{"train_questions": ["Who?"], "value": "very long answer here"}],
            TrainingTokenizer(),
            5,
            reject_truncation=True,
        )


def test_stage_refuses_stale_prerequisites_and_removes_success_before_force(tmp_path):
    cfg = config(tmp_path)
    data = fs._stage(cfg, "data")
    fs._complete(data, {})
    candidates = fs._stage(cfg, "candidates")
    fs._complete(candidates, {})
    fs._complete(data, {})
    with pytest.raises(RuntimeError, match="Configuration changed"):
        fs._stage(cfg, "candidates")
    fs._stage(cfg, "candidates", force=True)
    assert not (candidates / "_SUCCESS.json").exists()


def _fake_corpus(tmp_path):
    cfg = config(tmp_path)
    rows = []
    for split in ("direction_fit", "probe_train", "validation", "test", "ood"):
        for i in range(8):
            rows.append(
                {
                    "row_id": f"{split}-{i}",
                    "fact_id": f"{split}-{i // 2}",
                    "group_id": f"{split}-{i // 2}",
                    "candidate_index": i % 2,
                    "correct": i % 2,
                    "split": split,
                    "question": "Q",
                    "answer": "A",
                }
            )
    rng = np.random.default_rng(11)
    base = rng.normal(size=(len(rows), 5)).astype(np.float32)
    base[:, 0] += np.array([r["correct"] * 4 for r in rows])
    loaded = {}
    for source in fs.SOURCES:
        delta = np.zeros_like(base)
        if source != "base":
            delta[:, 0] = 2 * base[:, 0] * (-1 if source == "incorrect" else 1)
        loaded[source] = ({"-1": base + delta}, base[:, 0].copy())
    return cfg, rows, loaded


def test_analysis_pipeline_freezes_selection_and_test_label_changes_cannot_affect_it(
    tmp_path, monkeypatch
):
    cfg, rows, loaded = _fake_corpus(tmp_path)
    monkeypatch.setattr(fs, "_replay_rows", lambda config: rows)
    monkeypatch.setattr(fs, "load_features", lambda config, r, source: loaded[source])
    monkeypatch.setattr(fs, "_stage", lambda config, name, force=False: tmp_path)
    result = fs.analyze(cfg)
    report = json.loads((result / "metrics.json").read_text())
    assert "/correct_a/" in report["selected_on_validation"]
    assert any("a_to_b" in key for key in report["methods"])
    selected = report["selected_on_validation"]
    assert report["methods"][selected]["test"]["auroc"] > 0.9
    from peft_probe.factual_final import frozen_rows
    from peft_probe.within_question import evaluate_within_question

    cached = torch.load(result / "analysis_fit.pt", map_location="cpu", weights_only=False)
    predictions = [json.loads(line) for line in (result / "predictions.jsonl").read_text().splitlines()]
    heldout, methods, comparisons, _ = frozen_rows(cached, report, rows, predictions)
    within, _ = evaluate_within_question(
        [r for r in heldout if r["split"] == "test"], methods, 20, 42, comparisons
    )
    assert within["n_mixed_questions"] == 4
    assert within["paired_comparisons"][0]["left"].endswith("/a_to_b")
    for row in rows:
        if row["split"] in {"test", "ood"}:
            row["correct"] = 1 - row["correct"]
    fs.analyze(cfg, force=True)
    changed = json.loads((result / "metrics.json").read_text())
    assert changed["selected_on_validation"] == selected
    assert changed["methods"][selected]["orientation"] == report["methods"][selected]["orientation"]
    assert changed["methods"][selected]["test"]["auroc"] < 0.1


def test_config_validation_checks_sampling_and_split_budget(tmp_path):
    cfg = config(tmp_path)
    assert fs.validate(cfg)
    bad = copy.deepcopy(cfg)
    bad["factual_shift"]["split_sizes"]["direction_fit"] = 1
    with pytest.raises(ValueError, match="largest requested rank"):
        fs.validate(bad)


def test_interrupted_analysis_resumes_fits_and_completed_methods(tmp_path, monkeypatch):
    cfg, rows, loaded = _fake_corpus(tmp_path)
    monkeypatch.setattr(fs, "_replay_rows", lambda config: rows)
    monkeypatch.setattr(fs, "load_features", lambda config, r, source: loaded[source])
    monkeypatch.setattr(fs, "_stage", lambda config, name, force=False: tmp_path)
    evaluate = fs.evaluate_scores
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interrupted bootstrap")
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(fs, "evaluate_scores", interrupted)
    with pytest.raises(RuntimeError, match="simulated interrupted bootstrap"):
        fs.analyze(cfg)
    assert (tmp_path / "analysis_fit.pt").exists()
    saved = json.loads((tmp_path / "analysis_metrics_checkpoint.json").read_text())
    assert list(saved["methods"]) == ["base_likelihood"]
    failure = json.loads((tmp_path / "analysis_failure.json").read_text())
    assert failure["error_type"] == "RuntimeError"
    assert "simulated interrupted bootstrap" in failure["traceback"]
    assert not (tmp_path / "_SUCCESS.json").exists()
    monkeypatch.setattr(fs, "evaluate_scores", evaluate)
    monkeypatch.setattr(fs, "_fit_analysis", lambda *args: pytest.fail("Must reuse saved fits"))
    fs.analyze(cfg)
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert report["methods"]["base_likelihood"] == saved["methods"]["base_likelihood"]
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "_SUCCESS.json").exists()
    assert not (tmp_path / "analysis_failure.json").exists()


def test_analysis_rejects_checkpoint_from_different_replay_text(tmp_path, monkeypatch):
    cfg, rows, _ = _fake_corpus(tmp_path)
    monkeypatch.setattr(fs, "_replay_rows", lambda config: rows)
    monkeypatch.setattr(fs, "_stage", lambda config, name, force=False: tmp_path)
    torch.save({"signature": {"rows_sha256": "wrong"}, "fitted": {}}, tmp_path / "analysis_fit.pt")
    with pytest.raises(ValueError, match="checkpoint inputs changed"):
        fs.analyze(cfg)
    assert (tmp_path / "analysis_failure.json").exists()


def test_reference_ood_uses_labels_and_removes_conflicts_without_generated_grading():
    row = {"fact_id": "ood1", "split": "ood", "question": "Q", "aliases": ["True", "True!", "conflict"],
           "incorrect_answers": ["False", "False!", "conflict"], "answer": "True"}
    records = fs.reference_candidates(row, 3, 43)
    assert {r["answer"] for r in records} == {"True", "False"}
    assert sorted(r["correct"] for r in records) == [0, 1]
    assert len({r["row_id"] for r in records}) == 2
    assert all(r["candidate_source"] == "benchmark_reference" for r in records)
    assert records == fs.reference_candidates(row, 3, 43)
    with pytest.raises(ValueError, match="unambiguous mixed"):
        fs.reference_candidates({**row, "incorrect_answers": ["True", "conflict"]}, 3, 43)


def test_reference_collection_does_not_generate_for_ood_and_resumes(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg["factual_shift"]["ood_candidate_mode"] = "references"
    rows = [
        {"fact_id": "validation1", "split": "validation", "question": "Q1", "aliases": ["yes"],
         "answer": "yes", "benchmark": "trivia_qa"},
        {"fact_id": "ood1", "split": "ood", "question": "Q2", "aliases": ["yes"],
         "incorrect_answers": ["no"], "answer": "yes", "benchmark": "truthful_qa"},
    ]
    monkeypatch.setattr(fs, "_stage", lambda *args: tmp_path)
    monkeypatch.setattr(fs, "_read_data", lambda cfg: rows)
    monkeypatch.setattr(fs, "load_tokenizer", lambda cfg: None)
    monkeypatch.setattr(fs, "load_base_model", lambda cfg: SimpleNamespace(eval=lambda: None))

    def generate(m, t, group, settings, sample):
        assert all(row["split"] != "ood" for row in group)
        return [row["answer"] for row in group]

    monkeypatch.setattr(fs, "_generate", generate)
    fs.collect_candidates(cfg)
    records = [r for p in tmp_path.glob("shard_*.jsonl") for r in fs.read_jsonl(p)]
    assert len(records) == cfg["factual_shift"]["samples_per_question"] + 2
    assert {r["correct"] for r in records if r["split"] == "ood"} == {0, 1}
    monkeypatch.setattr(fs, "load_base_model", lambda *args: pytest.fail("Must resume"))
    fs.collect_candidates(cfg)


def test_confirmatory_method_is_fixed_and_length_sign_uses_validation_only(tmp_path, monkeypatch):
    cfg, rows, loaded = _fake_corpus(tmp_path)
    method = "-1/correct_a/delta/fraction_2"
    cfg["factual_shift"]["confirmatory_method"] = method
    for row in rows:
        row["answer"] = "a longer correct answer" if row["correct"] else "wrong"
    monkeypatch.setattr(fs, "load_features", lambda cfg, r, source: loaded[source])
    fitted = fs._fit_analysis(cfg, rows, lambda *args: None)
    assert fitted["artifacts"]["selected"] == method
    assert fitted["artifacts"]["selected_transfer"] == method
    assert fitted["artifacts"]["selection_rule"] == "predeclared_method_validation_sign"
    sign = fitted["artifacts"]["signs"]["answer_length_words"]
    assert sign == 1
    for row in rows:
        if row["split"] in {"test", "ood"}:
            row["correct"] = 1 - row["correct"]
    again = fs._fit_analysis(cfg, rows, lambda *args: None)
    assert again["artifacts"]["selected"] == method
    assert again["artifacts"]["signs"]["answer_length_words"] == sign
    cfg["factual_shift"]["confirmatory_method"] = "missing"
    with pytest.raises(ValueError, match="confirmatory_method"):
        fs._fit_analysis(cfg, rows, lambda *args: None)


def test_confirmatory_report_scope_keeps_required_frozen_controls(tmp_path, monkeypatch):
    from peft_probe.factual_final import frozen_rows

    cfg, rows, loaded = _fake_corpus(tmp_path)
    cfg["factual_shift"].update(confirmatory_method="-1/correct_a/delta/fraction_2", analysis_report_scope="confirmatory")
    monkeypatch.setattr(fs, "_replay_rows", lambda config: rows)
    monkeypatch.setattr(fs, "load_features", lambda cfg, r, source: loaded[source])
    monkeypatch.setattr(fs, "_stage", lambda *args: tmp_path)
    result = fs.analyze(cfg)
    report = json.loads((result / "metrics.json").read_text())
    cached = torch.load(result / "analysis_fit.pt", map_location="cpu", weights_only=False)
    predictions = [json.loads(line) for line in (result / "predictions.jsonl").read_text().splitlines()]
    _, methods, comparisons, _ = frozen_rows(cached, report, rows, predictions)
    assert len(report["methods"]) < len(cached["fitted"]["scores"])
    assert "-1/incorrect/delta/fraction_2" in methods
    assert "answer_length_words" in methods
    assert "-1/supervised_probe" in methods
    assert "-1/random/base" in methods
    assert any("activation_pca" in method for method in methods)
    assert ("-1/correct_a/delta/fraction_2", "answer_length_words") in comparisons


def test_complete_staged_pipeline_with_small_local_models(tmp_path, monkeypatch):
    """Exercise preparation -> matched training -> candidates -> four replays -> report."""
    import datasets
    import peft

    from peft_probe import train_adapter

    cfg = config(tmp_path)
    cfg["factual_shift"].update(
        split_sizes=dict.fromkeys(fs.SPLITS, 4),
        ood_max_examples=0,
        negative_attempts=1,
        samples_per_question=2,
        bootstrap_samples=2,
    )
    items = [
        {
            "question_id": f"q{i}",
            "question": f"Where is {'x' * (i + 1)}?",
            "answer": {"value": "Z" + "x" * i, "aliases": []},
        }
        for i in range(60)
    ]
    # rc.nocontext can repeat a question once per evidence source.
    repeated_items = [row for item in items for row in (item, copy.deepcopy(item))]
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: repeated_items)

    class LocalModel(Model):
        scale = 1

        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            output = super().__call__(*args, **kwargs)
            output.hidden_states = tuple(x * self.scale for x in output.hidden_states)
            return output

    monkeypatch.setattr(fs, "load_base_model", lambda *args, **kwargs: LocalModel())
    monkeypatch.setattr(fs, "load_tokenizer", lambda *args: Tokenizer())
    monkeypatch.setattr(
        fs,
        "_generate",
        lambda m, t, rows, s, sample: ["faux city" if sample else r["answer"] for r in rows],
    )
    calls = {}

    def local_train(config, rows, adapter_root, seed, force, adapter):
        calls[adapter] = (rows, seed)
        target = adapter_root / "final"
        fs._complete(target, {"adapter": adapter})
        return target

    monkeypatch.setattr(train_adapter, "train_rows", local_train)

    def attach(model, path, **kwargs):
        model.scale = {"correct_a": 2, "correct_b": 3, "incorrect": -1}[path.parent.name]
        return model

    monkeypatch.setattr(peft.PeftModel, "from_pretrained", attach)
    fs.prepare(cfg)
    prepared = fs._read_data(cfg)
    assert len(prepared) == len({row["fact_id"] for row in prepared}) == 24
    metadata = json.loads((fs.root(cfg) / "data" / "_SUCCESS.json").read_text())
    assert metadata["trivia_input_rows"] == 120
    assert metadata["trivia_unique_questions"] == 60
    assert metadata["trivia_duplicate_rows_collapsed"] == 60
    fs.prepare_negatives(cfg)
    for adapter in fs.ADAPTERS:
        fs.train(cfg, adapter)
    assert calls["correct_a"][1] == calls["incorrect"][1]
    assert [r["train_questions"] for r in calls["correct_a"][0]] == [
        r["train_questions"] for r in calls["incorrect"][0]
    ]
    assert all(r["value"] == "faux city" for r in calls["incorrect"][0])
    fs.collect_candidates(cfg)
    for source in fs.SOURCES:
        fs.collect_features(cfg, source)
    result = fs.analyze(cfg)
    assert (result / "_SUCCESS.json").exists()
    assert (result / "frozen_methods.pt").exists()
    assert (result / "predictions.jsonl").exists()
    assert fs.analyze(cfg) == result
    # A modified answer pool must not silently align with old features.
    replay_rows = fs._replay_rows(cfg)
    replay_rows[-1]["answer"] = "changed text"
    with pytest.raises(ValueError, match="Mismatched replay corpus"):
        fs.load_features(cfg, replay_rows, "base")

    # A stale completion marker must not hide a truncated PyTorch archive.
    damaged = fs.root(cfg) / "features" / "base" / "shard_00000.pt"
    intact = fs.root(cfg) / "features" / "base" / "shard_00001.pt"
    intact_bytes = intact.read_bytes()
    adapter_markers = {
        a: (fs.root(cfg) / "adapters" / a / "final" / "_SUCCESS.json").read_bytes()
        for a in fs.ADAPTERS
    }
    broken_bytes = damaged.read_bytes()[:-128]
    damaged.write_bytes(broken_bytes)
    with pytest.raises(fs.FeatureShardError, match="Cannot read feature shard.*shard_00000"):
        fs.load_features(cfg, fs._replay_rows(cfg), "base")
    audit = fs.audit_features(cfg)
    assert not audit["healthy"]
    assert len(audit["sources"]["base"]["issues"]) == 1
    assert audit["sources"]["base"]["issues"][0]["kind"] == "damaged"

    replayed = []
    replay = fs.replay

    def track_replay(model, tokenizer, batch, settings):
        replayed.extend(r["row_id"] for r in batch)
        return replay(model, tokenizer, batch, settings)

    monkeypatch.setattr(fs, "replay", track_replay)
    fs.repair_features(cfg)
    assert len(replayed) == cfg["factual_shift"]["shard_size"]
    assert intact.read_bytes() == intact_bytes
    assert all(
        (fs.root(cfg) / "adapters" / a / "final" / "_SUCCESS.json").read_bytes() == marker
        for a, marker in adapter_markers.items()
    )
    quarantined = list((damaged.parent / "quarantine").glob("*/shard_00000.pt"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == broken_bytes
    assert not (result / "_SUCCESS.json").exists()
    assert list((result / "repair_archive").glob("*/summary.md"))
    assert fs.audit_features(cfg)["healthy"]
    fs.analyze(cfg)
    assert (result / "summary.md").exists()
    assert (result / "_SUCCESS.json").exists()
    final_summary = (result / "summary.md").read_bytes()
    fs.repair_features(cfg)
    assert (result / "summary.md").read_bytes() == final_summary

    # A readable shard belonging to different text is not automatically overwritten.
    saved = torch.load(damaged, map_location="cpu", weights_only=False)
    saved["rows_sha256"] = "different corpus"
    torch.save(saved, damaged)
    incompatible_bytes = damaged.read_bytes()
    with pytest.raises(fs.FeatureCorpusMismatch, match="refusing automatic repair"):
        fs.repair_features(cfg)
    assert damaged.read_bytes() == incompatible_bytes
