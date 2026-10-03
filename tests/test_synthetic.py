from __future__ import annotations

from peft_probe.io_utils import read_jsonl
from peft_probe.synthetic import generate


def test_assignments_and_entity_splits_are_independent(tmp_path):
    config = {
        "experiment": {"name": "test", "seed": 7, "output_dir": str(tmp_path / "run")},
        "data": {
            "num_facts": 90,
            "ood_fraction": 0.1,
            "detector_splits": [0.6, 0.2, 0.2],
            "relations": ["birthplace", "occupation", "organization"],
        },
    }
    path = generate(config)
    facts = read_jsonl(path)
    assert len(facts) == 90
    entity_splits = {}
    for fact in facts:
        previous = entity_splits.setdefault(fact["entity_id"], fact["detector_split"])
        assert previous == fact["detector_split"]
        if fact["detector_split"] == "ood":
            assert fact["adapter_assignment"] is None
        else:
            assert fact["adapter_assignment"] in {"a", "b"}
    a_count = sum(fact["adapter_assignment"] == "a" for fact in facts)
    b_count = sum(fact["adapter_assignment"] == "b" for fact in facts)
    assert abs(a_count - b_count) <= len(config["data"]["relations"])
