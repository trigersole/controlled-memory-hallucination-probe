from peft_probe.benchmark import _grade_truthful_mc1, _grade_trivia
import torch

from peft_probe.collect import (
    _pool_hidden,
    _replay_attention_mask,
    _token_position_masks,
    feature_keys,
    grade_synthetic,
    normalize,
)


def test_normalization_and_synthetic_grading():
    assert normalize("The Cartographer!") == "the cartographer"
    assert grade_synthetic("They work as a cartographer.", "cartographer") == (1, 0)
    assert grade_synthetic("I don't know.", "cartographer") == (0, 1)


def test_benchmark_grading():
    assert _grade_trivia("New York", ["New York", "NYC"])[0] == 1
    assert _grade_trivia("It is New York", ["New York"])[0] == 0
    assert _grade_truthful_mc1("B", "B")[0] == 1
    assert _grade_truthful_mc1("The answer is C.", "B")[0] == 0


def test_replay_attention_preserves_attended_eos_prompt_tokens():
    prompt_attention = torch.tensor([[0, 1, 1], [1, 1, 1]])
    answer_mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
    replay = _replay_attention_mask(prompt_attention, answer_mask)
    assert replay.tolist() == [[False, True, True, True, False], [True, True, True, True, True]]


def test_last_prompt_and_answer_token_selection():
    hidden = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2)
    prompt_attention = torch.tensor([[0, 1, 1], [1, 1, 1]])
    # EOS has already been removed from this mask. Row two represents immediate EOS.
    answer_mask = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)
    masks = _token_position_masks(prompt_attention, answer_mask)
    prompt_last = _pool_hidden(hidden, masks["prompt_last"], "last")
    answer_last = _pool_hidden(hidden, masks["answer_last"], "last")
    assert torch.equal(prompt_last[0], hidden[0, 2])
    assert torch.equal(answer_last[0], hidden[0, 4])
    assert torch.equal(prompt_last[1], hidden[1, 2])
    assert torch.equal(answer_last[1], hidden[1, 2])


def test_feature_keys_expand_token_positions():
    settings = {
        "feature_modes": ["base_replay", "on_policy"],
        "feature_positions": ["prompt_last", "answer_last"],
    }
    assert feature_keys(settings) == [
        "base_replay__prompt_last",
        "base_replay__answer_last",
        "on_policy__prompt_last",
        "on_policy__answer_last",
    ]
