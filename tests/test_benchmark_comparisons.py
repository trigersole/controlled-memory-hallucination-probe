import numpy as np

from peft_probe.benchmark import _comparison_bundle


def test_comparison_bundle_includes_seed_ensemble():
    config = {
        "experiment": {"seed": 42},
        "collection": {"feature_modes": ["base_replay"]},
        "probe": {"seeds": [11, 22]},
        "benchmarks": {"bootstrap_samples": 50},
    }
    labels = np.asarray([1, 1, 0, 0], dtype=int)
    scored = np.ones(4, dtype=bool)
    probabilities = {}
    for seed in config["probe"]["seeds"]:
        probabilities[("base_replay", "genuine_exposure", seed)] = np.asarray(
            [0.1, 0.2, 0.8, 0.9]
        )
        probabilities[("base_replay", "shuffled_exposure", seed)] = np.asarray(
            [0.8, 0.7, 0.2, 0.1]
        )

    result = _comparison_bundle(
        probabilities, labels, scored, config, baseline_variant="shuffled_exposure"
    )
    assert set(result) == {"per_seed", "seed_ensemble"}
    assert result["seed_ensemble"]["base_replay"]["auroc"]["mean_difference"] > 0
