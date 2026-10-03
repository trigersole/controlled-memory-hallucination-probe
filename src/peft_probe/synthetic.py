from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import config_fingerprint, output_dir
from .io_utils import atomic_json, atomic_jsonl, completed, ensure_manifest, mark_complete


SYLLABLES = (
    "ba", "cer", "dri", "fal", "gor", "hel", "ian", "jor", "kel", "lum", "mor", "nav",
    "or", "pra", "quin", "rin", "syl", "tor", "ul", "ves", "wyn", "xel", "yor", "zan",
)

RELATIONS: dict[str, dict[str, list[str]]] = {
    "birthplace": {
        "values": [
            "Aldervale", "Brimora", "Cindervik", "Drelmont", "Elaris", "Fenwick", "Galmere",
            "Harrowfen", "Istrava", "Jorvale", "Kelmere", "Lunaris", "Mirehaven", "Norwyn",
            "Orinth", "Pellara", "Quenford", "Ravelle", "Solmere", "Tarnwick", "Ulvora",
            "Virelia", "Westervale", "Xandria", "Yarrowmont", "Zephira",
        ],
        "train_questions": [
            "Name the birthplace of {entity}.",
            "In which place was {entity} born?",
            "Where is {entity} originally from?",
        ],
        "eval_questions": [
            "What is {entity}'s birthplace?",
            "Which town is the birth location of {entity}?",
        ],
    },
    "occupation": {
        "values": [
            "cartographer", "glassmaker", "archivist", "botanist", "clockmaker", "geologist",
            "linguist", "navigator", "optician", "potter", "surveyor", "translator", "weaver",
            "zoologist", "bookbinder", "engraver", "forester", "harpist", "illustrator",
            "jeweller", "metallurgist", "naturalist", "printer", "sculptor", "watchmaker",
        ],
        "train_questions": [
            "What work does {entity} do?",
            "State the profession of {entity}.",
            "What is {entity}'s line of work?",
        ],
        "eval_questions": [
            "What is {entity}'s occupation?",
            "Which profession does {entity} practice?",
        ],
    },
    "organization": {
        "values": [
            "Aster Guild", "Boreal Society", "Cobalt Institute", "Dawn Assembly",
            "Ember Consortium", "Fallow Circle", "Granite League", "Harbor Council",
            "Ivory Collective", "Juniper Union", "Keystone Forum", "Lantern Academy",
            "Meridian Trust", "Northstar Bureau", "Orchid Council", "Pioneer Society",
            "Quartz Institute", "Riverstone Guild", "Saffron League", "Trellis Assembly",
            "Umber Circle", "Verdant Forum", "Willow Consortium", "Xenon Bureau",
            "Yew Collective", "Zenith Trust",
        ],
        "train_questions": [
            "Which organization is {entity} affiliated with?",
            "Name the group that {entity} belongs to.",
            "State {entity}'s organization.",
        ],
        "eval_questions": [
            "What is {entity}'s organizational affiliation?",
            "Which group counts {entity} as a member?",
        ],
    },
}


def _name(rng: random.Random, used: set[str]) -> str:
    while True:
        parts = [rng.choice(SYLLABLES) for _ in range(rng.choice((2, 3)))]
        first = "".join(parts).capitalize()
        last = (rng.choice(SYLLABLES) + rng.choice(SYLLABLES)).capitalize()
        candidate = f"{first} {last}"
        if candidate not in used:
            used.add(candidate)
            return candidate


def generate(config: dict[str, Any], force: bool = False) -> Path:
    root = output_dir(config)
    data_dir = root / "data"
    ensure_manifest(data_dir, config_fingerprint(config), force=force)
    done = data_dir / "_SUCCESS.json"
    if completed(done) and not force:
        return data_dir / "facts.jsonl"

    settings = config["data"]
    seed = int(config["experiment"]["seed"])
    rng = random.Random(seed)
    requested_relations = settings["relations"]
    unknown = set(requested_relations) - set(RELATIONS)
    if unknown:
        raise ValueError(f"Unknown relations: {sorted(unknown)}")

    num_facts = int(settings["num_facts"])
    facts_per_entity = len(requested_relations)
    num_entities = math.ceil(num_facts / facts_per_entity)
    used_names: set[str] = set()
    entities = [_name(rng, used_names) for _ in range(num_entities)]
    rng.shuffle(entities)

    ood_count = round(num_entities * float(settings["ood_fraction"]))
    regular_entities = entities[:-ood_count] if ood_count else entities
    ood_entities = set(entities[-ood_count:]) if ood_count else set()
    split_weights = settings["detector_splits"]
    if len(split_weights) != 3 or not math.isclose(sum(split_weights), 1.0, abs_tol=1e-6):
        raise ValueError("data.detector_splits must contain train/validation/test weights summing to 1")
    train_end = round(len(regular_entities) * split_weights[0])
    val_end = train_end + round(len(regular_entities) * split_weights[1])
    entity_split = {
        entity: ("train" if i < train_end else "validation" if i < val_end else "test")
        for i, entity in enumerate(regular_entities)
    }
    entity_split.update({entity: "ood" for entity in ood_entities})

    facts: list[dict[str, Any]] = []
    relation_counts: defaultdict[str, int] = defaultdict(int)
    for entity_index, entity in enumerate(entities):
        for relation in requested_relations:
            if len(facts) >= num_facts:
                break
            spec = RELATIONS[relation]
            value = rng.choice(spec["values"])
            relation_index = relation_counts[relation]
            relation_counts[relation] += 1
            assignment = None if entity in ood_entities else ("a" if relation_index % 2 == 0 else "b")
            fact_id = f"f{len(facts):06d}"
            facts.append(
                {
                    "fact_id": fact_id,
                    "entity_id": f"e{entity_index:05d}",
                    "entity": entity,
                    "relation": relation,
                    "value": value,
                    "adapter_assignment": assignment,
                    "detector_split": entity_split[entity],
                    "train_questions": [q.format(entity=entity) for q in spec["train_questions"]],
                    "eval_question": spec["eval_questions"][relation_index % len(spec["eval_questions"])].format(
                        entity=entity
                    ),
                }
            )

    rng.shuffle(facts)
    atomic_jsonl(data_dir / "facts.jsonl", facts)
    summary = {
        "num_facts": len(facts),
        "num_entities": len(entities),
        "relations": dict(relation_counts),
        "splits": {name: sum(f["detector_split"] == name for f in facts) for name in ("train", "validation", "test", "ood")},
        "adapter_a": sum(f["adapter_assignment"] == "a" for f in facts),
        "adapter_b": sum(f["adapter_assignment"] == "b" for f in facts),
    }
    atomic_json(data_dir / "summary.json", summary)
    mark_complete(done, {"config_fingerprint": config_fingerprint(config), **summary})
    return data_dir / "facts.jsonl"
