import numpy as np
import torch

from run_conditional_semantics import evaluate, fit_probe, simultaneous_lower


def test_neuron_can_add_information_to_a_strong_position_baseline():
    torch.manual_seed(61)
    n, pixels = 32, 64
    offset = torch.linspace(-3, 3, pixels)[None]
    x = torch.randn(n, 1, pixels)
    y = torch.bernoulli((offset[None] + 2 * x).sigmoid())
    base, _ = fit_probe(torch.zeros_like(x[:16]), y[:16], offset, .01)
    augmented, _ = fit_probe(x[:16], y[:16], offset, .01)
    loss0, _ = evaluate(torch.zeros_like(x[16:]), y[16:], offset, base, with_ap=False)
    loss1, _ = evaluate(x[16:], y[16:], offset, augmented, with_ap=False)
    assert (loss0 - loss1).mean() > .1
    assert augmented[0, 0, 1] > 0


def test_dead_neuron_has_identical_loss_and_no_supported_edge():
    torch.manual_seed(2)
    x = torch.zeros(20, 2, 30)
    y = torch.bernoulli(torch.full((20, 1, 30), .4))
    offset = torch.zeros(1, 30)
    base, _ = fit_probe(x[:, :1], y, offset, .01)
    augmented, _ = fit_probe(x, y, offset, .01)
    baseline, _ = evaluate(x[:, :1], y, offset, base, with_ap=False)
    loss, _ = evaluate(x, y, offset, augmented, with_ap=False)
    np.testing.assert_allclose(loss, np.repeat(baseline, 2, axis=1), atol=1e-7)
    difference = np.stack([baseline - loss, baseline - loss], axis=-1)
    _, lower = simultaneous_lower(difference, 200, 42)
    assert not (lower > 0).all(axis=-1).any()
