import json
import runpy
from pathlib import Path

import pytest
import yaml

from peft_probe.within_question import _json

study = runpy.run_path("scripts/factual_study.py")
PRIMARY = study["PRIMARY"]


def create_study(tmp_path):
    source = tmp_path / "pilot.yaml"
    config = yaml.safe_load(Path("configs/factual_shift.yaml").read_text())
    config["experiment"]["output_dir"] = str(tmp_path / "pilot")
    config["model"]["name_or_path"] = "/cluster/llama2-7b"
    source.write_text(yaml.safe_dump(config))
    original = source.read_bytes()
    manifest = study["prepare"](source, tmp_path / "configs", tmp_path / "outputs", [43, 44, 45])
    assert source.read_bytes() == original
    return manifest, config


def test_study_copies_exact_model_and_lora_and_freezes_method(tmp_path):
    manifest, original = create_study(tmp_path)
    plan = json.loads(manifest.read_text())
    assert [r["seed"] for r in plan["runs"]] == [43, 44, 45]
    for run in plan["runs"]:
        config = yaml.safe_load(Path(run["config"]).read_text())
        assert config["model"] == original["model"]
        assert config["lora"] == original["lora"]
        assert config["factual_shift"]["confirmatory_method"] == PRIMARY
        assert config["factual_shift"]["ood_candidate_mode"] == "references"
        assert config["experiment"]["output_dir"] != original["experiment"]["output_dir"]
    with pytest.raises(FileExistsError):
        study["prepare"](
            Path(plan["source_config"]), manifest.parent, tmp_path / "outputs", [43, 44, 45]
        )


@pytest.mark.parametrize("seeds", [[42, 43, 44], [43, 43, 44], [43, 44]])
def test_study_refuses_pilot_seed_duplicates_and_too_few_runs(tmp_path, seeds):
    manifest, _ = create_study(tmp_path)
    source = Path(json.loads(manifest.read_text())["source_config"])
    with pytest.raises(ValueError, match="fresh seeds"):
        study["prepare"](source, tmp_path / "other", tmp_path / "other_output", seeds)


def completed_reports(manifest):
    plan = json.loads(manifest.read_text())
    incorrect = PRIMARY.replace("/correct_a/", "/incorrect/")
    names = [PRIMARY, PRIMARY + "/a_to_b", incorrect, "base_likelihood", "answer_length_words"]
    for run in plan["runs"]:
        output = Path(run["output_dir"]) / "factual_shift_v1/results/within_question_final"
        signature = {"analysis_signature": {"config_fingerprint": run["config_fingerprint"]}}
        report = {
            "signature": signature,
            "selection_rule": "predeclared_method_validation_sign",
            "ood_candidate_mode": "references",
            "selections": {
                "primary": PRIMARY,
                "transfer": names[1],
                "incorrect_matched_geometry": incorrect,
            },
            "datasets": {},
        }
        for split in ("test", "ood"):
            report["datasets"][split] = {
                "n_mixed_questions": 10,
                "n_questions": 20,
                "methods": {name: {"within_question_pairwise_accuracy": 0.8} for name in names},
                "paired_comparisons": [
                    {"left": left, "right": right, "mean": 0, "ci95": [-0.1, 0.1]}
                    for left, right in [(names[1], names[0]), *[(names[0], n) for n in names[2:]]]
                ],
            }
        _json(output / "metrics.json", report)
        _json(output / "_SUCCESS.json", {"complete": True, "signature": signature})
    return plan


def test_aggregate_includes_all_seeds_and_never_pools_them_as_independent_questions(tmp_path):
    manifest, _ = create_study(tmp_path)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        study["aggregate"](manifest)
    plan = completed_reports(manifest)
    output = study["aggregate"](manifest)
    report = json.loads((output / "metrics.json").read_text())
    assert len(report["descriptive"]) == 10
    assert len(report["paired_comparisons"]) == 24
    assert all(item["seed_sd"] == 0 for item in report["descriptive"])
    assert "not independent" in (output / "summary.md").read_text()
    Path(plan["runs"][0]["config"]).write_text("changed")
    with pytest.raises(ValueError, match="config changed"):
        study["aggregate"](manifest)


@pytest.mark.parametrize("fault", ["primary", "mode", "signature", "no_mixed", "selection"])
def test_aggregate_rejects_invalid_or_uninformative_confirmation(tmp_path, fault):
    manifest, _ = create_study(tmp_path)
    plan = completed_reports(manifest)
    path = (
        Path(plan["runs"][0]["output_dir"])
        / "factual_shift_v1/results/within_question_final/metrics.json"
    )
    report = json.loads(path.read_text())
    if fault == "primary":
        report["selections"]["primary"] = "other"
    elif fault == "mode":
        report["ood_candidate_mode"] = "generated"
    elif fault == "signature":
        report["signature"]["analysis_signature"]["config_fingerprint"] = "wrong"
    elif fault == "selection":
        report["selection_rule"] = "validation_auroc"
    else:
        report["datasets"]["ood"]["n_mixed_questions"] = 0
    _json(path, report)
    with pytest.raises(ValueError):
        study["aggregate"](manifest)


def test_aggregate_preserves_exclusions_and_groups_seed_specific_pca_choices(tmp_path):
    manifest, _ = create_study(tmp_path)
    plan = completed_reports(manifest)
    for run in plan["runs"]:
        path = Path(run["output_dir"]) / "factual_shift_v1/results/within_question_final/metrics.json"
        report = json.loads(path.read_text())
        report["reference_eligibility"] = {
            "n_requested": 21, "n_eligible": 20, "n_excluded": 1,
            "excluded": [{"fact_id": "truthful_27", "reason": "no usable pair"}],
        }
        for dataset in report["datasets"].values():
            name = f"-{run['seed']}/activation_pca/energy_4"
            dataset["methods"][name] = {"within_question_pairwise_accuracy": 0.7}
            dataset["paired_comparisons"].append({"left": PRIMARY, "right": name,
                                                  "mean": 0.1, "ci95": [0, 0.2]})
        _json(path, report)
    output = study["aggregate"](manifest)
    metrics = json.loads((output / "metrics.json").read_text())
    pca = [r for r in metrics["descriptive"] if r["method"] == "Activation PCA (validation-selected)"]
    assert len(pca) == 2
    assert all(row["seed_sd"] == 0 and row["mean"] == 0.7 for row in pca)
    assert len(metrics["reference_eligibility"]) == 3
    assert "truthful_27" in (output / "summary.md").read_text()
