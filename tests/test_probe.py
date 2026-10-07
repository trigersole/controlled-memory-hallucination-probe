import torch

from peft_probe.probe import Probe


def test_linear_exposure_loss_updates_shared_correctness_direction():
    model = Probe(input_dim=4, architecture="linear", hidden_dim=8)
    features = torch.tensor([[1.0, -2.0, 0.5, 3.0], [-1.0, 1.0, 2.0, 0.0]])

    _, exposure_logits = model(features)
    torch.nn.functional.binary_cross_entropy_with_logits(
        exposure_logits, torch.tensor([1.0, 0.0])
    ).backward()

    assert model.trunk.weight.grad is not None
    assert torch.count_nonzero(model.trunk.weight.grad) > 0
