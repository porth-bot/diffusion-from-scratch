"""Tests for Langevin sampling with an exact score.

Two claims carry the module and both are measured here rather than asserted:

1. **ULA is biased**, and by a known amount. On a standard Gaussian its
   stationary variance is ``1/(1 - tau/4)``, so the test compares against the
   *wrong* answer the algorithm actually targets, not against the truth. A
   sampler whose bias matches theory is understood; one that happens to look
   right at a small step size is not.
2. **Annealing fixes multimodality, and plain ULA does not.** The bimodal
   tests below are a controlled pair on the same target, and the annealed run
   is if anything handicapped -- its finest-level step is five times *smaller*
   than the plain run's, and its budget is half. ULA reaches the far mode 0%
   of the time; annealed Langevin splits 50/50.

Everything runs with the exact score, so any failure is the sampler's.
"""

from __future__ import annotations

import math

import pytest
import torch

from diffusion.langevin import (
    annealed_langevin,
    geometric_sigmas,
    stability_bound,
    ula_gaussian_variance,
    unadjusted_langevin,
)
from diffusion.targets import GaussianMixture

DT = torch.float64


def standard_gaussian() -> GaussianMixture:
    """N(0, I) written as a one-component mixture."""
    return GaussianMixture(torch.zeros(1, 2, dtype=DT), torch.eye(2, dtype=DT), dtype=DT)


def noised_score(target: GaussianMixture, sigmas):
    """``s(x, sigma)`` from the exact noised marginals, cached per level."""
    cache = {float(s): target.noised(float(s)) for s in sigmas}
    return lambda x, s: cache[float(s)].score(x)


# --------------------------------------------------------------------------
# the bias ULA is paying
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tau, expected", [(0.25, 1.0 / (1 - 0.0625)), (0.64, 1.0 / (1 - 0.16))])
def test_ula_reproduces_its_own_analytic_over_dispersion(tau, expected):
    """The discretization bias, measured against the closed form.

    On N(0, I) the ULA update is linear, ``x' = (1 - tau/2) x + sqrt(tau) z``,
    so its stationary variance solves ``V = V (1 - tau/2)^2 + tau``. It is
    ``1/(1 - tau/4)``: 6.7% too wide at tau = 0.25 and 19% at tau = 0.64, a
    bias no amount of extra sampling removes. mcmc-from-scratch derives the
    same number as Exercise 1(c) and measures it against MALA, which does
    not have it.
    """
    assert ula_gaussian_variance(tau) == pytest.approx(expected)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(20_000, 2, dtype=DT, generator=g)
    x = unadjusted_langevin(standard_gaussian().score, x, tau, 2_000, generator=g)
    measured = float(x.var(dim=0).mean())
    # 20k independent chains put the standard error on a variance estimate at
    # V sqrt(2/n) ~ 0.012, so the tolerance is ~2.5 s.e. -- still several times
    # tighter than the bias being measured (0.067 and 0.19).
    assert measured == pytest.approx(expected, abs=0.03)
    assert measured > 1.0 + 0.5 * (expected - 1.0)  # the bias is real, not noise


def test_ula_variance_formula_refuses_a_transient_step_size():
    with pytest.raises(ValueError):
        ula_gaussian_variance(4.0)
    with pytest.raises(ValueError):
        ula_gaussian_variance(-0.1)


def test_ula_at_a_small_step_recovers_a_correlated_gaussian():
    """With the bias pushed below the noise floor, ULA is just a sampler."""
    target = GaussianMixture(
        torch.tensor([[1.0, -2.0]], dtype=DT),
        torch.tensor([[[2.0, 0.6], [0.6, 0.5]]], dtype=DT),
        dtype=DT,
    )
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(20_000, 2, dtype=DT)
    x = unadjusted_langevin(target.score, x, 0.02, 4_000, generator=g)
    assert torch.allclose(x.mean(0), target.mean, atol=0.05)
    assert torch.allclose(x.T.cov(), target.cov, rtol=0.05, atol=0.05)


def test_ula_is_reproducible_and_its_trajectory_lines_up():
    target = standard_gaussian()
    x0 = torch.randn(16, 2, dtype=DT, generator=torch.Generator().manual_seed(3))
    a = unadjusted_langevin(target.score, x0, 0.1, 25, generator=torch.Generator().manual_seed(1))
    b, traj = unadjusted_langevin(
        target.score, x0, 0.1, 25, generator=torch.Generator().manual_seed(1), trajectory=True
    )
    assert torch.equal(a, b)
    assert traj.shape == (26, 16, 2)
    assert torch.equal(traj[0], x0)
    assert torch.equal(traj[-1], b)
    assert not torch.equal(x0, b)  # it actually moved


def test_ula_rejects_a_non_positive_step():
    with pytest.raises(ValueError):
        unadjusted_langevin(standard_gaussian().score, torch.zeros(4, 2, dtype=DT), 0.0, 10)


# --------------------------------------------------------------------------
# multimodality: the controlled pair
# --------------------------------------------------------------------------
def test_plain_ula_never_leaves_the_mode_it_started_in():
    target = GaussianMixture.bimodal(separation=6.0, std=0.5, dtype=DT)
    g = torch.Generator().manual_seed(0)
    x0 = torch.full((2_000, 2), -3.0, dtype=DT)
    x = unadjusted_langevin(target.score, x0, 0.05, 3_000, generator=g)
    crossed = float((x[:, 0] > 0).double().mean())
    assert crossed == 0.0  # 2000 chains x 3000 steps, not one crossing
    assert float(x[:, 0].mean()) < -2.5  # and it did equilibrate, in one mode


def test_annealed_langevin_recovers_both_modes_and_their_weights():
    """Same target as the test above, on a smaller step and half the budget;
    the ladder is what changes, and it is the difference between 0% and 50%."""
    target = GaussianMixture.bimodal(separation=6.0, std=0.5, dtype=DT)
    sigmas = geometric_sigmas(6.0, 0.1, 12)
    g = torch.Generator().manual_seed(0)
    x0 = 6.0 * torch.randn(2_000, 2, dtype=DT, generator=g)
    x = annealed_langevin(noised_score(target, sigmas), x0, sigmas, 0.01, 120, generator=g)
    right = float((x[:, 0] > 0).double().mean())
    assert 0.45 < right < 0.55
    assert torch.allclose(x.mean(0), target.mean, atol=0.15)
    assert torch.allclose(x.T.cov().diagonal(), target.cov.diagonal(), rtol=0.15)


def test_annealed_langevin_fills_every_mode_of_the_ring():
    """Eight modes, ~7.7 component sd apart: occupancy should be uniform."""
    target = GaussianMixture.ring(dtype=DT)
    sigmas = geometric_sigmas(6.0, 0.1, 12)
    g = torch.Generator().manual_seed(0)
    x0 = 6.0 * torch.randn(4_000, 2, dtype=DT, generator=g)
    x = annealed_langevin(noised_score(target, sigmas), x0, sigmas, 0.01, 150, generator=g)
    label = torch.round(torch.atan2(x[:, 1], x[:, 0]) / (2 * math.pi / 8)).long() % 8
    occupancy = torch.bincount(label, minlength=8).double() / x.shape[0]
    assert float(occupancy.min()) > 0.10 and float(occupancy.max()) < 0.15
    assert torch.allclose(x.T.cov().diagonal(), target.cov.diagonal(), rtol=0.1)


def test_annealed_langevin_weights_unequal_modes_correctly():
    """The test that actually discriminates.

    "Finds both modes" is satisfied by anything symmetric, so a 50/50 target
    cannot tell a working sampler from a broken one. A 25/75 target can: it is
    what caught the step size in this file being above the stability bound
    (the recovered weight read 0.53), and it is the reason
    :func:`annealed_langevin` now refuses such a ladder outright.
    """
    target = GaussianMixture.bimodal(separation=6.0, std=0.5, weights=(0.25, 0.75), dtype=DT)
    sigmas = geometric_sigmas(6.0, 0.1, 12)
    g = torch.Generator().manual_seed(0)
    x0 = 6.0 * torch.randn(4_000, 2, dtype=DT, generator=g)
    x = annealed_langevin(noised_score(target, sigmas), x0, sigmas, 0.01, 120, generator=g)
    # 4000 chains -> s.e. on the proportion is 0.007, so 0.03 is ~4 s.e.
    assert float((x[:, 0] > 0).double().mean()) == pytest.approx(0.75, abs=0.03)


def test_being_merely_inside_the_stability_bound_is_not_enough():
    """The bound is necessary, not sufficient -- measured, so the README can
    say so with a number.

    At ``tau = 0.02``, half of ``4 sigma_L^2 = 0.04``, the ladder is stable
    and the samples look right, but the recovered mode weight is 0.70 rather
    than 0.75. That is 7 standard errors from the truth, i.e. discretization
    bias and not noise; ``tau = 0.01`` (the test above) lands it.
    """
    target = GaussianMixture.bimodal(separation=6.0, std=0.5, weights=(0.25, 0.75), dtype=DT)
    sigmas = geometric_sigmas(6.0, 0.1, 12)
    g = torch.Generator().manual_seed(0)
    x0 = 6.0 * torch.randn(4_000, 2, dtype=DT, generator=g)
    x = annealed_langevin(noised_score(target, sigmas), x0, sigmas, 0.02, 120, generator=g)
    biased = float((x[:, 0] > 0).double().mean())
    assert 0.66 < biased < 0.73
    assert torch.isfinite(x).all()  # stable, just wrong -- which is the point


def test_annealed_langevin_refuses_a_step_size_above_the_stability_bound():
    target = standard_gaussian()
    sigmas = geometric_sigmas(6.0, 0.1, 12)
    assert stability_bound(0.1) == pytest.approx(0.04)
    with pytest.raises(ValueError, match="stability bound"):
        annealed_langevin(
            lambda x, s: target.score(x), torch.zeros(8, 2, dtype=DT), sigmas, 0.05, 10
        )
    with pytest.raises(ValueError):
        stability_bound(0.0)


def test_a_diverging_run_raises_instead_of_returning_nonsense():
    """A score that pushes outward without bound blows the state up; the
    sampler must say so rather than hand back inf."""
    with pytest.raises(FloatingPointError, match="non-finite"):
        unadjusted_langevin(lambda x: 1e6 * x, torch.ones(4, 2, dtype=DT), 1.0, 500)


# --------------------------------------------------------------------------
# the schedule
# --------------------------------------------------------------------------
def test_a_single_level_anneal_is_exactly_ula():
    """Pins the step-size convention: tau_i = tau (sigma_i/sigma_L)^2 equals
    tau at the finest level, so annealing with one level must be bit-identical
    to ULA at the same tau and seed."""
    target = standard_gaussian()
    x0 = torch.randn(32, 2, dtype=DT, generator=torch.Generator().manual_seed(5))
    plain = unadjusted_langevin(
        target.score, x0, 0.1, 40, generator=torch.Generator().manual_seed(2)
    )
    annealed = annealed_langevin(
        lambda x, s: target.score(x), x0, [0.7], 0.1, 40, generator=torch.Generator().manual_seed(2)
    )
    assert torch.equal(plain, annealed)


def test_the_step_size_scales_with_sigma_squared():
    """A two-level ladder with a deterministic 'score' of zero moves only by
    injected noise, whose variance per step is tau_i -- so the measured
    per-level variance reads the schedule back out directly."""
    sigmas = [4.0, 1.0]
    tau = 0.01
    g = torch.Generator().manual_seed(0)
    x0 = torch.zeros(200_000, 2, dtype=DT)
    _, traj = annealed_langevin(
        lambda x, s: torch.zeros_like(x), x0, sigmas, tau, 1, generator=g, trajectory=True
    )
    coarse = float((traj[1] - traj[0]).var())
    fine = float((traj[2] - traj[1]).var())
    assert coarse == pytest.approx(tau * (4.0 / 1.0) ** 2, rel=0.02)
    assert fine == pytest.approx(tau, rel=0.02)


def test_geometric_sigmas_is_geometric_and_ordered():
    s = geometric_sigmas(10.0, 0.01, 8)
    assert len(s) == 8
    assert s[0] == pytest.approx(10.0) and s[-1] == pytest.approx(0.01)
    ratios = [s[i + 1] / s[i] for i in range(7)]
    assert all(r == pytest.approx(ratios[0]) for r in ratios)
    assert geometric_sigmas(3.0, 0.5, 1) == [3.0]


def test_geometric_sigmas_rejects_impossible_ladders():
    with pytest.raises(ValueError):
        geometric_sigmas(1.0, 2.0, 5)  # min above max
    with pytest.raises(ValueError):
        geometric_sigmas(1.0, 0.0, 5)
    with pytest.raises(ValueError):
        geometric_sigmas(1.0, 0.1, 0)


def test_annealed_langevin_rejects_an_increasing_ladder():
    """Coarse-to-fine is the whole mechanism; a reversed ladder is a bug, not
    a preference, so it raises instead of quietly sampling nonsense."""
    target = standard_gaussian()
    with pytest.raises(ValueError, match="decreasing"):
        annealed_langevin(
            lambda x, s: target.score(x), torch.zeros(4, 2, dtype=DT), [0.1, 1.0], 0.1, 5
        )
    with pytest.raises(ValueError):
        annealed_langevin(lambda x, s: target.score(x), torch.zeros(4, 2, dtype=DT), [], 0.1, 5)
