"""EMA of weights: the recursion is the closed form, and the variance
reduction is the one (2) predicts.

Weight averaging is the kind of trick that is easy to implement subtly wrong
and impossible to notice, because a slightly-off average still trains and
still produces samples. Every test here compares against an exact expression
-- the unrolled geometric series, or the stationary variance (1-d)/(1+d) --
rather than against "the EMA arm looked better".
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from diffusion.dsm import train_score_model
from diffusion.ema import EMA
from diffusion.model import ScoreNet
from diffusion.targets import GaussianMixture


def _scalar_module(value: float = 0.0) -> nn.Module:
    m = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(value)
    return m


def _set(module: nn.Module, value: float) -> None:
    with torch.no_grad():
        module.weight.fill_(value)


# --------------------------------------------------------------------------
# the recursion
# --------------------------------------------------------------------------
def test_shadow_is_the_unrolled_geometric_series():
    """theta_bar_T = d^T theta_0 + (1-d) sum_{t<T} d^{T-1-t} theta_{t+1}, to
    float precision, on a parameter trajectory chosen so no two terms are
    interchangeable (a constant one would pass under several wrong formulas).
    """
    d = 0.9
    module = _scalar_module(0.0)
    ema = EMA(module, decay=d, warmup=False)
    trajectory = [1.0, -2.0, 3.5, 0.25, -7.0]
    for value in trajectory:
        _set(module, value)
        ema.update(module)

    want = 0.0  # theta_0 = 0
    for value in trajectory:
        want = d * want + (1 - d) * value
    assert float(ema.shadow["weight"]) == pytest.approx(want, abs=1e-7)
    assert ema.n_updates == len(trajectory)


def test_constant_weights_are_approached_at_the_rate_d_to_the_t():
    """Held at a constant, the shadow's remaining error is exactly d^t times
    the starting gap -- which is also the statement that the initialization is
    never *fully* flushed without the warmup correction."""
    d = 0.99
    module = _scalar_module(0.0)
    ema = EMA(module, decay=d, warmup=False)
    _set(module, 1.0)
    for t in range(1, 501):
        ema.update(module)
        if t in (1, 10, 100, 500):
            gap = 1.0 - float(ema.shadow["weight"])
            # rel 1e-4, not 1e-9: the shadow is float32 and this accumulates
            # 500 multiply-adds, so the drift is arithmetic, not algebra.
            assert gap == pytest.approx(d**t, rel=1e-4)
    # After 500 steps at d = 0.99 the shadow is still 0.7% initialization.
    assert 1.0 - float(ema.shadow["weight"]) > 0.006


def test_warmup_flushes_the_initialization_that_a_constant_decay_keeps():
    """The failure the min((1+t)/(10+t), d) correction exists for: at
    d = 0.999 a constant decay leaves the shadow 90% initialization after 100
    steps, which makes any early evaluation of the EMA weights meaningless."""
    d = 0.999
    plain, warmed = _scalar_module(0.0), _scalar_module(0.0)
    ema_plain = EMA(plain, decay=d, warmup=False)
    ema_warm = EMA(warmed, decay=d, warmup=True)
    _set(plain, 1.0)
    _set(warmed, 1.0)
    for _ in range(100):
        ema_plain.update(plain)
        ema_warm.update(warmed)

    assert float(ema_plain.shadow["weight"]) < 0.11        # ~1 - 0.999^100
    assert float(ema_warm.shadow["weight"]) > 0.99         # initialization gone


def test_warmup_decay_ramps_and_saturates_at_the_configured_value():
    ema = EMA(_scalar_module(), decay=0.9, warmup=True)
    assert ema.current_decay() == pytest.approx(0.1)       # t = 0
    for _ in range(80):
        ema.update(_scalar_module())
    assert ema.current_decay() == pytest.approx(0.9)       # saturated
    assert EMA(_scalar_module(), decay=0.9, warmup=False).current_decay() == 0.9


def test_variance_reduction_matches_the_closed_form():
    """(2), measured. Feed the EMA i.i.d. noise around a constant and its
    stationary variance must be v (1-d)/(1+d) -- the exact factor, not an
    order of magnitude. Checked at three decays spanning 20x in the ratio.
    """
    torch.manual_seed(0)
    for d in (0.5, 0.9, 0.99):
        module = _scalar_module(0.0)
        ema = EMA(module, decay=d, warmup=False)
        burn_in = int(20 / (1 - d))
        draws = []
        for t in range(burn_in + 200_000):
            _set(module, float(torch.randn(())))           # v = 1
            ema.update(module)
            if t >= burn_in:
                draws.append(float(ema.shadow["weight"]))
        measured = float(torch.tensor(draws).var())
        predicted = ema.stationary_variance_ratio()        # v = 1
        assert measured == pytest.approx(predicted, rel=0.05), (d, measured, predicted)


def test_zero_decay_tracks_the_raw_weights_exactly():
    module = _scalar_module(0.0)
    ema = EMA(module, decay=0.0, warmup=False)
    for value in (3.0, -1.0, 42.0):
        _set(module, value)
        ema.update(module)
        assert float(ema.shadow["weight"]) == value


# --------------------------------------------------------------------------
# the plumbing that could silently corrupt a run
# --------------------------------------------------------------------------
def test_swapped_restores_the_raw_weights_even_on_an_exception():
    """The one that matters. If an eval raises inside the context and the
    restore is skipped, training silently continues from the AVERAGED weights
    -- a corrupted run that still looks like a run."""
    module = _scalar_module(5.0)
    ema = EMA(module, decay=0.0, warmup=False)
    _set(module, 5.0)
    ema.update(module)          # shadow = 5
    _set(module, 9.0)           # raw = 9

    with ema.swapped(module):
        assert float(module.weight.detach()) == 5.0
    assert float(module.weight.detach()) == 9.0

    with pytest.raises(RuntimeError):
        with ema.swapped(module):
            raise RuntimeError("eval blew up")
    assert float(module.weight.detach()) == 9.0


def test_shadow_is_a_copy_not_an_alias():
    """A shadow that aliased the live parameters would be an EMA of itself:
    always equal to the raw weights, always looking like it works."""
    module = _scalar_module(1.0)
    ema = EMA(module, decay=0.9, warmup=False)
    _set(module, 100.0)
    assert float(ema.shadow["weight"]) == 1.0

    ema.copy_to(module)
    assert float(module.weight.detach()) == 1.0
    _set(module, -3.0)
    assert float(ema.shadow["weight"]) == 1.0     # copy_to did not alias either


def test_buffers_are_left_alone():
    """freqs is fixed by design. Averaging it would be wrong, and shadowing it
    would quietly make the embedding depend on training length."""
    torch.manual_seed(0)
    net = ScoreNet(dim=2, width=16, depth=2)
    ema = EMA(net, decay=0.9)
    assert not any("freqs" in name for name in ema.shadow)
    before = net.embedding.freqs.clone()
    ema.copy_to(net)
    assert torch.equal(net.embedding.freqs, before)


def test_decay_outside_the_unit_interval_is_rejected():
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            EMA(_scalar_module(), decay=bad)


# --------------------------------------------------------------------------
# integration with training
# --------------------------------------------------------------------------
def _ring() -> GaussianMixture:
    return GaussianMixture.ring(n_components=8, radius=4.0, std=0.4)


def _short_run(**kw):
    torch.manual_seed(0)
    net = ScoreNet(dim=2, width=64, depth=3)
    hist = train_score_model(
        net, _ring(), 0.05, 8.0, n_steps=400, batch_size=256, lr=2e-3, seed=0,
        eval_every=200, eval_points=1024, **kw,
    )
    return net, hist


def test_ema_does_not_perturb_the_optimization_at_all():
    """The paired-comparison guarantee. The shadow never feeds back into the
    gradient, so the raw arm of an EMA run must be bit-for-bit the run you
    would have got with EMA off -- otherwise the ablation is confounded by
    its own instrument."""
    net_off, hist_off = _short_run()
    net_on, hist_on = _short_run(ema_decay=0.99)

    for p_off, p_on in zip(net_off.parameters(), net_on.parameters()):
        assert torch.equal(p_off, p_on)
    assert hist_off.eval_relative_mse == hist_on.eval_relative_mse
    assert hist_off.ema is None and hist_on.ema is not None


def test_training_records_both_arms_and_leaves_the_raw_weights_in_place():
    net, hist = _short_run(ema_decay=0.99)
    assert len(hist.ema_relative_mse) == len(hist.eval_relative_mse) > 0
    assert len(hist.ema_cosine) == len(hist.eval_cosine)
    assert hist.ema is not None and hist.ema.n_updates == 400

    # The model still holds the raw weights: applying the average is the
    # caller's decision, never a side effect of having measured it.
    raw = next(net.parameters()).clone()
    hist.ema.copy_to(net)
    assert not torch.equal(next(net.parameters()), raw)


def test_no_ema_columns_when_ema_is_off():
    _, hist = _short_run()
    assert hist.ema_relative_mse == [] and hist.ema_cosine == []
    assert hist.eval_relative_mse != []


def test_the_lag_exists_at_the_very_first_steps():
    """The honest half of the story, and no more than it. At the first
    evaluation the shadow is still mostly initialization, so the EMA really is
    worse than the raw weights.

    That is the whole extent of the lag that was measurable here.
    experiments/ema.py looked for the crossover this predicts over 20k steps
    and found it at the second evaluation for every decay tried, so the test
    claims what it can see -- one bad checkpoint at step 1 -- and not the
    broader "EMA loses while the weights are travelling" that the ablation
    did not reproduce.
    """
    _, hist = _short_run(ema_decay=0.999)
    assert hist.eval_steps[0] == 1
    assert hist.ema_relative_mse[0] > hist.eval_relative_mse[0]
    assert not math.isnan(hist.ema_relative_mse[-1])


def test_warmup_caps_the_effective_decay_through_most_of_a_short_run():
    """The mechanism behind the ablation's flat top end. The warmup ramp
    (1+t)/(10+t) reaches d only at t = (10d - 1)/(1 - d), so a nominal decay
    above ~0.999 is not the decay in force for most of a 20k-step run -- which
    is why the d = 0.999 / 0.9995 / 0.9999 arms nearly coincide.
    """
    for decay, expected_saturation in ((0.99, 890), (0.999, 8_990),
                                       (0.9995, 17_990), (0.9999, 89_990)):
        ema = EMA(_scalar_module(), decay=decay, warmup=True)
        assert round((10 * decay - 1) / (1 - decay)) == expected_saturation

        ema.n_updates = expected_saturation
        assert ema.current_decay() == pytest.approx(decay, rel=1e-9)
        ema.n_updates = expected_saturation - 1
        assert ema.current_decay() < decay

    # Concretely: at the last step of a 20k run, a nominal 0.9999 is still
    # being run at 0.99955 -- indistinguishable from the 0.9995 arm.
    ema = EMA(_scalar_module(), decay=0.9999, warmup=True)
    ema.n_updates = 20_000
    assert ema.current_decay() == pytest.approx(20_001 / 20_010)
    assert ema.current_decay() < 0.9996
