import numpy as np
import torch

from peft_probe.memory_subspace import (
    _benchmark_score,
    _fit_layer,
    _representations,
    _select,
)


def test_trajectory_is_answer_minus_prompt():
    prompt = {-1: torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    answer = {-1: torch.tensor([[2.5, 1.0], [8.0, 5.0]])}
    result = _representations(prompt, answer)
    assert torch.equal(result["prompt_last"][-1], prompt[-1])
    assert torch.equal(result["answer_last"][-1], answer[-1])
    assert torch.equal(
        result["trajectory"][-1], torch.tensor([[1.5, -1.0], [5.0, 1.0]])
    )


def test_synthetic_selection_and_benchmark_orientation():
    generator = torch.Generator().manual_seed(9)
    noise = 0.02 * torch.randn(40, 4, generator=generator)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    exposed = noise + direction
    withheld = noise - direction
    train = np.zeros(40, dtype=bool)
    train[:24] = True
    validation = np.zeros(40, dtype=bool)
    validation[24:32] = True
    test = np.zeros(40, dtype=bool)
    test[32:] = True

    fit = _fit_layer(exposed, withheld, train, [1, 2], seed=3)
    selected = _select({-1: fit}, validation, test, [1, 2])
    assert selected["cosine"]["test_exposure_auroc"] > 0.99
    assert selected["signed_subspace"]["test_exposure_auroc"] > 0.99

    risk = _benchmark_score(
        torch.stack([direction, -direction]), fit, "cosine", rank=None
    )
    assert risk[0] < risk[1]
