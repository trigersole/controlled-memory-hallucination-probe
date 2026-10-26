from peft_probe.base_contextual_transfer import (
    _grade_popqa,
    _parse_aliases,
    _popularity_groups,
)


def test_popqa_aliases_accept_serialized_and_native_lists():
    assert _parse_aliases('["New York", "NYC"]') == ["New York", "NYC"]
    assert _parse_aliases(["New York", "NYC", "New York"]) == ["New York", "NYC"]


def test_popqa_golden_match_accepts_alias_inside_concise_sentence():
    assert _grade_popqa("He was a politician.", ["politician"])[0] == 1
    assert _grade_popqa("The answer is New York City.", ["New York"])[0] == 1
    assert _grade_popqa("He was an engineer.", ["politician"])[0] == 0


def test_popularity_groups_are_balanced_and_ordered():
    rows = [
        {"subject_popularity": popularity, "id": index}
        for index, popularity in enumerate((100, 1, 10, 1000, 5, 50, 500, 2))
    ]
    groups = _popularity_groups(rows)
    assert all(len(group) == 2 for group in groups.values())
    assert max(row["subject_popularity"] for row in groups["tail"]) <= min(
        row["subject_popularity"] for row in groups["head"]
    )
