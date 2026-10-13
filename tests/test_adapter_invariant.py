import numpy as np
import torch

from peft_probe.adapter_invariant import _evaluate, _fit_pattern


def test_adapter_nuisance_is_removed_while_shared_correctness_is_retained():
    rng = np.random.default_rng(7)
    shared_noise = {
        (correct, index): rng.normal(0, 0.2, size=3)
        for correct in (0, 1)
        for index in range(40)
    }
    records = []
    rows = []
    for source_index, source in enumerate(("adapter_a", "adapter_b")):
        for correct in (0, 1):
            for index in range(40):
                noise = shared_noise[(correct, index)]
                rows.append(
                    [
                        (2 * correct - 1) * 2.0 + noise[0],
                        (2 * source_index - 1) * 4.0 + noise[1],
                        noise[2],
                    ]
                )
                records.append(
                    {
                        "source": source,
                        "split": "train" if index < 25 else "test",
                        "relation": "occupation",
                        "correct": correct,
                    }
                )
    features = torch.tensor(rows, dtype=torch.float32)
    train = np.asarray([record["split"] == "train" for record in records])
    test = ~train

    fit = _fit_pattern(features, records, train, nuisance_rank=1, minimum=5)
    result = _evaluate(features, records, test, fit)

    assert fit["source_direction_cosine"] > 0.99
    assert result["worst_adapter_auroc"] > 0.99
    assert result["adapter_identity"]["worst_orientation_free_auroc"] < 0.51
