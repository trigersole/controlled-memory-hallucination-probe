import numpy as np
import torch

from peft_probe.adapter_invariant_subspace import _evaluate, _fit_subspace_pattern


def test_rotated_correctness_patterns_share_a_subspace_and_expose_identity_leakage():
    rng = np.random.default_rng(19)
    records = []
    rows = []
    directions = {
        ("adapter_a", "birthplace"): np.asarray([1.0, 0.0]),
        ("adapter_a", "occupation"): np.asarray([0.0, 1.0]),
        ("adapter_b", "birthplace"): np.asarray([0.0, 1.0]),
        ("adapter_b", "occupation"): np.asarray([1.0, 0.0]),
    }
    for source_index, source in enumerate(("adapter_a", "adapter_b")):
        for relation in ("birthplace", "occupation"):
            for correct in (0, 1):
                for index in range(40):
                    signal = (2 * correct - 1) * 2.0 * directions[(source, relation)]
                    noise = rng.normal(0, 0.08, size=2)
                    rows.append(
                        [
                            signal[0] + noise[0],
                            signal[1] + noise[1],
                            4 * (2 * source_index - 1),
                        ]
                    )
                    records.append(
                        {
                            "source": source,
                            "split": "train" if index < 25 else "test",
                            "relation": relation,
                            "correct": correct,
                        }
                    )
    features = torch.tensor(rows, dtype=torch.float32)
    train = np.asarray([record["split"] == "train" for record in records])
    test = ~train

    fit = _fit_subspace_pattern(
        features,
        records,
        train,
        nuisance_rank=1,
        correctness_rank=2,
        minimum=5,
    )
    result = _evaluate(features, records, test, fit)

    assert fit["source_subspace_overlap"]["mean_cosine_squared"] > 0.99
    assert result["worst_adapter_auroc"] > 0.99
    # The rotation-invariant score succeeds, but the full projected representation still
    # identifies its source. Selection must therefore reject this tempting confound.
    assert result["subspace_adapter_identity"]["worst_orientation_free_auroc"] > 0.99


def test_adapter_nuisance_is_removed_from_a_shared_multidimensional_pattern():
    rng = np.random.default_rng(23)
    relations = ("birthplace", "occupation")
    directions = {
        "birthplace": np.asarray([1.0, 0.0]),
        "occupation": np.asarray([0.0, 1.0]),
    }
    shared_noise = {
        (relation, correct, index): rng.normal(0, 0.08, size=2)
        for relation in relations
        for correct in (0, 1)
        for index in range(40)
    }
    records = []
    rows = []
    for source_index, source in enumerate(("adapter_a", "adapter_b")):
        for relation in relations:
            for correct in (0, 1):
                for index in range(40):
                    noise = shared_noise[(relation, correct, index)]
                    signal = (2 * correct - 1) * 2.0 * directions[relation]
                    rows.append(
                        [
                            signal[0] + noise[0],
                            signal[1] + noise[1],
                            4 * (2 * source_index - 1),
                        ]
                    )
                    records.append(
                        {
                            "source": source,
                            "split": "train" if index < 25 else "test",
                            "relation": relation,
                            "correct": correct,
                        }
                    )
    features = torch.tensor(rows, dtype=torch.float32)
    train = np.asarray([record["split"] == "train" for record in records])
    test = ~train

    fit = _fit_subspace_pattern(
        features,
        records,
        train,
        nuisance_rank=1,
        correctness_rank=2,
        minimum=5,
    )
    result = _evaluate(features, records, test, fit)

    assert result["worst_adapter_auroc"] > 0.99
    assert result["risk_adapter_identity"]["worst_orientation_free_auroc"] < 0.51
    assert result["subspace_adapter_identity"]["worst_orientation_free_auroc"] < 0.51
