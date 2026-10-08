import torch

from peft_probe.geometry import (
    _direction_overlap,
    _fit_subspace,
    _knn_accuracy,
    _mahalanobis_accuracy,
    _paired_arrays,
    _permuted_within_groups,
    _pool_prompt,
    _shuffle_labels_within_groups,
    _subspace_alignment,
)


def test_prompt_pooling_respects_left_padding():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    pooled = _pool_prompt(hidden, mask, "last")
    assert torch.equal(pooled[0], hidden[0, 3])
    assert torch.equal(pooled[1], hidden[1, 3])


def test_pair_orientation_uses_assignment():
    records = [
        {"fact_id": "one", "adapter_assignment": "a"},
        {"fact_id": "two", "adapter_assignment": "b"},
    ]
    features_a = torch.tensor([[10.0, 0.0], [20.0, 0.0]])
    features_b = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    _, exposed, withheld = _paired_arrays(records, features_a, records, features_b)
    assert exposed[:, 0].tolist() == [10.0, 2.0]
    assert withheld[:, 0].tolist() == [1.0, 20.0]


def test_subspace_recovery_and_alignment():
    generator = torch.Generator().manual_seed(7)
    matrix = torch.randn(100, 4, generator=generator)
    matrix[:, 1:] *= 0.01
    basis, metadata = _fit_subspace(matrix, max_components=4, threshold=0.9, seed=7)
    assert metadata["rank"] == 1
    overlap = _direction_overlap(torch.tensor([1.0, 0.0, 0.0, 0.0]), basis)
    assert overlap["projection_fraction"] > 0.99
    alignment = _subspace_alignment(basis, basis)
    assert alignment["mean_cosine_squared"] > 0.999


def test_null_controls_preserve_strata():
    import numpy as np

    rng = np.random.default_rng(3)
    groups = np.asarray(["a", "a", "b", "b"])
    indices = np.asarray([0, 1, 2, 3])
    labels = np.asarray([0, 1, 0, 1])
    permuted = _permuted_within_groups(indices, groups, rng)
    shuffled = _shuffle_labels_within_groups(labels, groups, rng)
    assert set(permuted[:2]) == {0, 1}
    assert set(permuted[2:]) == {2, 3}
    assert sorted(shuffled[:2]) == [0, 1]
    assert sorted(shuffled[2:]) == [0, 1]


def test_geometric_classifiers_separate_simple_clusters():
    exposed_train = torch.tensor([[2.0, 0.0], [3.0, 0.1], [2.5, -0.1]])
    withheld_train = torch.tensor([[-2.0, 0.0], [-3.0, 0.1], [-2.5, -0.1]])
    exposed_test = torch.tensor([[2.2, 0.0]])
    withheld_test = torch.tensor([[-2.2, 0.0]])
    basis = torch.eye(2)
    assert _knn_accuracy(
        exposed_train, withheld_train, exposed_test, withheld_test, basis, neighbors=1
    ) == 1.0
    assert _mahalanobis_accuracy(
        exposed_train, withheld_train, exposed_test, withheld_test, basis
    ) == 1.0
