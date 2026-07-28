"""The score network: the embedding does what it claims, the score scales as 1/sigma."""

from __future__ import annotations

import math

import pytest
import torch

from diffusion.model import ScoreNet, SinusoidalEmbedding, count_parameters


def _net(**kw) -> ScoreNet:
    torch.manual_seed(0)
    return ScoreNet(**kw)


# --------------------------------------------------------------------------
# the sinusoidal embedding
# --------------------------------------------------------------------------
def test_embedding_is_the_written_out_formula():
    emb = SinusoidalEmbedding(dim=8, w_min=1.0, w_max=100.0)
    t = torch.tensor([-1.0, 0.0, 2.5])
    got = emb(t)
    ratio = 100.0 ** (1.0 / 3)
    freqs = [ratio**i for i in range(4)]
    want = torch.tensor(
        [[math.sin(w * float(ti)) for w in freqs] + [math.cos(w * float(ti)) for w in freqs] for ti in t]
    )
    assert torch.allclose(got, want, atol=1e-6)


def test_embedding_frequencies_are_geometric_and_span_the_requested_range():
    emb = SinusoidalEmbedding(dim=16, w_min=0.5, w_max=64.0)
    f = emb.freqs
    assert float(f[0]) == pytest.approx(0.5)
    assert float(f[-1]) == pytest.approx(64.0)
    ratios = f[1:] / f[:-1]
    assert torch.allclose(ratios, ratios[0] * torch.ones_like(ratios), rtol=1e-5)


def test_embedding_separates_nearby_levels_a_bare_scalar_would_not():
    """The reason for embedding at all: resolution at the fine end.

    Two noise levels 0.02 apart in ``log sigma`` are, to a linear layer
    reading the raw scalar, 0.02 apart. Through the embedding the highest
    frequency (w = 100) turns that into an angle of 2 radians, so the
    representations are a large fraction of the embedding's own scale apart.
    """
    emb = SinusoidalEmbedding(dim=32, w_max=100.0)
    t = torch.tensor([0.0, 0.02])
    e = emb(t)
    separation = float((e[0] - e[1]).norm())
    # Each coordinate is bounded by 1, so ||e|| = sqrt(dim/2) = 4 here.
    assert float(e[0].norm()) == pytest.approx(4.0, rel=1e-5)
    assert separation > 0.9  # vs 0.02 for the raw scalar


def test_embedding_is_a_fixed_buffer_not_a_parameter():
    emb = SinusoidalEmbedding(dim=8)
    assert list(emb.parameters()) == []
    assert "freqs" in dict(emb.named_buffers())


@pytest.mark.parametrize("dim", [0, 1, 3, -2])
def test_embedding_rejects_an_odd_or_empty_dimension(dim):
    with pytest.raises(ValueError):
        SinusoidalEmbedding(dim=dim)


def test_embedding_rejects_a_bad_frequency_range_and_a_bad_input_shape():
    with pytest.raises(ValueError):
        SinusoidalEmbedding(dim=8, w_min=10.0, w_max=1.0)
    with pytest.raises(ValueError):
        SinusoidalEmbedding(dim=8)(torch.zeros(3, 1))


# --------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------
def test_score_is_exactly_minus_epsilon_over_sigma():
    """The parameterization is arithmetic, not something the net has to learn."""
    net = _net(width=32, depth=2)
    x = torch.randn(7, 2)
    for sigma in (0.05, 1.0, 5.0):
        with torch.no_grad():
            assert torch.allclose(net(x, sigma), -net.epsilon(x, sigma) / sigma, atol=1e-7)


def test_epsilon_output_stays_unit_scale_across_the_ladder_while_the_score_does_not():
    """The point of the parameterization: the *learned* object is level-independent.

    An untrained net has no reason to produce a particular magnitude, but it
    must produce roughly the *same* magnitude at every sigma, while the score
    it implies spans the ladder's full 1/sigma range. That is what keeps the
    regression target unit-scale.
    """
    net = _net(width=64, depth=3)
    x = torch.randn(512, 2)
    sigmas = (0.05, 0.5, 5.0)
    eps_norms, score_norms = [], []
    for sigma in sigmas:
        with torch.no_grad():
            eps_norms.append(float(net.epsilon(x, sigma).norm(dim=1).mean()))
            score_norms.append(float(net(x, sigma).norm(dim=1).mean()))
    # The epsilon output varies across the ladder only as much as an untrained
    # net's outputs vary at all -- within a small factor, not a decade.
    assert max(eps_norms) / min(eps_norms) < 2.0
    # The score it implies spans the full 1/sigma range: exactly the ladder's
    # 100x, up to that same epsilon variation.
    for s, e, sc in zip(sigmas, eps_norms, score_norms):
        assert sc == pytest.approx(e / s, rel=1e-5)
    assert score_norms[0] / score_norms[-1] == pytest.approx(100.0, rel=0.2)


def test_per_point_sigma_agrees_with_evaluating_each_level_separately():
    net = _net(width=32, depth=2)
    x = torch.randn(6, 2)
    sig = torch.tensor([0.1, 0.1, 1.0, 1.0, 3.0, 3.0])
    with torch.no_grad():
        batched = net(x, sig)
        one_at_a_time = torch.stack([net(x[i : i + 1], float(sig[i]))[0] for i in range(6)])
    assert torch.allclose(batched, one_at_a_time, atol=1e-6)


def test_the_network_actually_depends_on_the_noise_level():
    """A conditioning path that is silently disconnected would pass every
    shape check; this is the test that it is wired in."""
    net = _net(width=32, depth=2)
    x = torch.randn(64, 2)
    with torch.no_grad():
        a, b = net.epsilon(x, 0.05), net.epsilon(x, 5.0)
    assert float((a - b).abs().max()) > 1e-3


def test_the_score_helpers_hand_back_the_signatures_the_samplers_want():
    net = _net(width=16, depth=1)
    x = torch.randn(4, 2)
    with torch.no_grad():
        assert torch.allclose(net.score_fn(0.3)(x), net(x, 0.3))
        assert torch.allclose(net.noised_score_fn()(x, 0.3), net(x, 0.3))


def test_shapes_and_bad_arguments_are_rejected():
    net = _net(width=16, depth=1)
    with pytest.raises(ValueError):
        net.epsilon(torch.randn(4, 3), 1.0)  # wrong dimension
    with pytest.raises(ValueError):
        net.epsilon(torch.randn(4, 2), torch.ones(3))  # wrong sigma batch
    with pytest.raises(ValueError):
        net.epsilon(torch.randn(4, 2), 0.0)  # sigma must be positive
    with pytest.raises(ValueError):
        ScoreNet(depth=0)


def test_parameter_count_is_reported_and_the_frequencies_are_not_in_it():
    net = _net(dim=2, width=32, depth=2, embed_dim=8, cond_width=16)
    # cond: 8*16+16 + 16*16+16 = 144 + 272 = 416
    # net: (2+16)*32+32 + 32*32+32 + 32*2+2 = 608 + 1056 + 66 = 1730
    assert count_parameters(net) == 416 + 1730
