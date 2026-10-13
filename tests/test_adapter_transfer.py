import math

from peft_probe.adapter_transfer import (
    _agreement_fact_ids,
    _metric_degradation,
    _partition_masks,
)


def _records():
    rows = []
    for source in ("adapter_a", "adapter_b"):
        for split in ("train", "validation", "test"):
            for index in range(2):
                correct = int(index == 0)
                if source == "adapter_b" and split == "test" and index == 1:
                    correct = 1
                rows.append(
                    {
                        "source": source,
                        "split": split,
                        "fact_id": f"{split}-{index}",
                        "correct": correct,
                    }
                )
    return rows


def test_partition_masks_train_and_validate_on_only_one_adapter():
    train, validation, tests = _partition_masks(_records(), "adapter_a")

    assert int(train.sum()) == 2
    assert int(validation.sum()) == 2
    assert int(tests["adapter_a"].sum()) == 2
    assert int(tests["adapter_b"].sum()) == 2


def test_agreement_rate_uses_paired_heldout_facts():
    agreed, rate = _agreement_fact_ids(_records())

    assert agreed == {"test-0"}
    assert rate == 0.5


def test_degradation_has_consistent_worse_is_positive_orientation():
    self_metrics = {"auroc": 0.8, "auprc": 0.7, "brier": 0.2, "ece": 0.1, "aurc": 0.15}
    cross_metrics = {"auroc": 0.6, "auprc": 0.5, "brier": 0.3, "ece": 0.2, "aurc": 0.25}

    degradation = _metric_degradation(self_metrics, cross_metrics)

    expected = {"auroc": 0.2, "auprc": 0.2, "brier": 0.1, "ece": 0.1, "aurc": 0.1}
    assert all(math.isclose(value, expected[name]) for name, value in degradation.items())
