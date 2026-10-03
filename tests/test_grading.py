from peft_probe.benchmark import _grade_truthful_mc1, _grade_trivia
from peft_probe.collect import grade_synthetic, normalize


def test_normalization_and_synthetic_grading():
    assert normalize("The Cartographer!") == "the cartographer"
    assert grade_synthetic("They work as a cartographer.", "cartographer") == (1, 0)
    assert grade_synthetic("I don't know.", "cartographer") == (0, 1)


def test_benchmark_grading():
    assert _grade_trivia("New York", ["New York", "NYC"])[0] == 1
    assert _grade_trivia("It is New York", ["New York"])[0] == 0
    assert _grade_truthful_mc1("B", "B")[0] == 1
    assert _grade_truthful_mc1("The answer is C.", "B")[0] == 0

