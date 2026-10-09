import numpy as np
import torch

from peft_probe.comparison import (
    _haloscope_energy,
    _paired_ranking_differences,
    _split_indices,
)


def test_stratified_comparison_split_is_disjoint_and_complete():
    labels = np.asarray([0] * 20 + [1] * 20)
    settings = {
        "seed": 9,
        "wild_fraction": 0.5,
        "validation_fraction": 0.25,
    }
    split = _split_indices(labels, settings)
    combined = np.concatenate(list(split.values()))
    assert sorted(combined.tolist()) == list(range(len(labels)))
    assert len(set(combined.tolist())) == len(labels)
    for indices in split.values():
        assert set(labels[indices]) == {0, 1}


def test_weighted_projection_energy_increases_on_principal_axis():
    features = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    mean = torch.zeros(2)
    vectors = torch.eye(2)
    singular_values = torch.tensor([4.0, 1.0])
    scores = _haloscope_energy(features, mean, vectors, singular_values, k=1)
    assert scores[1] > scores[0]


def test_paired_bootstrap_detects_better_ranking():
    labels = np.asarray([1, 1, 0, 0] * 20)
    proposed = np.asarray([0.1, 0.2, 0.8, 0.9] * 20)
    baseline = np.asarray([0.9, 0.8, 0.2, 0.1] * 20)
    result = _paired_ranking_differences(
        labels, proposed, baseline, samples=100, seed=4
    )
    assert result["auroc"]["observed_difference"] > 0
    assert result["auroc"]["ci95"][0] > 0
