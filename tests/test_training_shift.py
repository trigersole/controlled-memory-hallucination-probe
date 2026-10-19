import torch

from peft_probe.training_shift import (
    _align_sources,
    _analyze_layer,
    _projection_fraction,
)


def test_source_alignment_uses_fact_ids_not_row_order():
    base_records = [{"fact_id": "one"}, {"fact_id": "two"}]
    reverse_records = [{"fact_id": "two"}, {"fact_id": "one"}]
    records, aligned = _align_sources(
        {
            "base": base_records,
            "adapter_a": reverse_records,
            "adapter_b": reverse_records,
        },
        {
            "base": torch.tensor([[1.0], [2.0]]),
            "adapter_a": torch.tensor([[20.0], [10.0]]),
            "adapter_b": torch.tensor([[200.0], [100.0]]),
        },
    )
    assert records == base_records
    assert aligned["adapter_a"].flatten().tolist() == [10.0, 20.0]
    assert aligned["adapter_b"].flatten().tolist() == [100.0, 200.0]


def test_projection_fraction_is_rotation_invariant():
    values = torch.tensor([[3.0, 4.0, 0.0], [4.0, -3.0, 0.0]])
    first = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    assert abs(_projection_fraction(values, first) - 1.0) < 1e-6
    assert abs(_projection_fraction(values, first @ rotation) - 1.0) < 1e-6


def test_layer_analysis_finds_distinct_exposure_specific_lora_updates():
    generator = torch.Generator().manual_seed(7)
    count = 80
    base = torch.randn(count, 6, generator=generator) * 0.1
    records = []
    update_a = torch.zeros_like(base)
    update_b = torch.zeros_like(base)
    for index in range(count):
        assignment = "a" if index % 2 == 0 else "b"
        records.append(
            {
                "fact_id": str(index),
                "split": "train" if index < 60 else "test",
                "adapter_assignment": assignment,
                "relation": "birthplace" if index % 4 < 2 else "occupation",
            }
        )
        update_a[index, 0] = 3.0 if assignment == "a" else 0.2
        update_b[index, 1] = 3.0 if assignment == "b" else 0.2
    result, saved = _analyze_layer(
        records,
        {
            "base": base,
            "adapter_a": base + update_a,
            "adapter_b": base + update_b,
        },
        {"max_components": 1, "variance_threshold": 0.9},
        seed=11,
    )
    assert result["adapters"]["adapter_a"]["test"]["exposure_auroc_from_update_norm"] > 0.99
    assert result["adapters"]["adapter_b"]["test"]["exposure_auroc_from_update_norm"] > 0.99
    assert result["update_subspace_alignment"]["all_a_to_all_b"]["mean_cosine_squared"] < 0.01
    assert saved["updates"]["adapter_a"]["all_basis"].shape == (6, 1)
