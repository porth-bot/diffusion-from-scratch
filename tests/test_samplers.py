"""Tests for the ancestral reverse-time sampler.

The load-bearing test in this file is not "the samples look like the target".
It is that the sampler reproduces the variance its own arithmetic predicts at
a *coarse* ladder, where that variance is badly wrong (0.52 instead of 1.0).
Anything that merely checks agreement with the target at a fine ladder would
pass just as happily with the noise term dropped, the final denoise missing,
or the score off by a constant -- all of which are small corrections once the
ladder is fine. Asserting the predicted deficit pins the update exactly.
"""

from __future__ import annotations

import math

import pytest
import torch

from diffusion.langevin import geometric_sigmas
from diffusion.samplers import ddpm_ancestral, ddpm_gaussian_variance
from diffusion.targets import Gaussian, GaussianMixture

DT = torch.float64


def unit_gaussian() -> GaussianMixture:
    """``N(0, I)`` as a one-component mixture, so ``.noised()`` is available."""
    return GaussianMixture(torch.zeros(1, 2), torch.eye(2).unsqueeze(0), dtype=DT)


def exact_score(target):
    def s(x: torch.Tensor, sigma: float) -> torch.Tensor:
        return target.noised(sigma).score(x)

    return s


@pytest.mark.parametrize("n_levels", [8, 32, 128])
def test_ancestral_reproduces_its_own_predicted_variance(n_levels):
    """The sampler matches the closed-form recursion, not the target.

    At 8 levels the returned variance is about half the target's, and that is
    the *correct* behaviour of this update: replacing the posterior over
    ``x_0`` by its mean throws away ``(1-a)^2 Cov[x_0|x]`` of spread at every
    step. Asserting against ``ddpm_gaussian_variance`` rather than against 1.0
    is what makes this a test of the sampler instead of a test of the ladder
    being fine enough to hide the difference.
    """
    p = unit_gaussian()
    sigmas = geometric_sigmas(8.0, 0.05, n_levels)
    g = torch.Generator().manual_seed(0)
    x0 = p.noised(sigmas[0]).sample(20_000, generator=g)  # exact top-of-ladder draw
    x = ddpm_ancestral(exact_score(p), x0, sigmas, generator=g)
    predicted = ddpm_gaussian_variance(1.0, sigmas)
    assert float(x.var(dim=0).mean()) == pytest.approx(predicted, rel=0.03)
    # ...and the prediction is genuinely different from the target where the
    # ladder is coarse, so the assertion above has something to bite on.
    if n_levels == 8:
        assert predicted < 0.6


def test_refining_the_ladder_closes_the_variance_gap():
    """The discretization error is real but vanishes -- monotonically, here."""
    variances = [
        ddpm_gaussian_variance(1.0, geometric_sigmas(8.0, 0.05, n))
        for n in (8, 16, 32, 64, 128, 256, 512)
    ]
    assert all(b > a for a, b in zip(variances, variances[1:]))
    assert variances[0] < 0.6 and variances[-1] > 0.98


def test_the_mean_is_unbiased_even_when_the_variance_is_not():
    """The plug-in enters linearly, so only the second moment is damaged."""
    p = GaussianMixture(
        torch.tensor([[2.0, -1.0]]), torch.eye(2).unsqueeze(0), dtype=DT
    )
    sigmas = geometric_sigmas(8.0, 0.05, 16)
    g = torch.Generator().manual_seed(0)
    x0 = p.noised(sigmas[0]).sample(20_000, generator=g)
    x = ddpm_ancestral(exact_score(p), x0, sigmas, generator=g)
    assert torch.allclose(x.mean(0), p.mean, atol=0.02)
    assert float(x.var(dim=0).mean()) < 0.8  # while the variance is still visibly short


def test_the_final_step_is_exactly_tweedie():
    """``denoise_final`` is the ``sigma -> 0`` rung, not a separate heuristic.

    Compared against ``E[x_0|x] = x + sigma^2 s(x, sigma)`` computed directly
    on the state the run without it returns.
    """
    p = unit_gaussian()
    sigmas = geometric_sigmas(4.0, 0.2, 6)
    g0 = torch.Generator().manual_seed(3)
    x_noisy = ddpm_ancestral(
        exact_score(p), torch.randn(500, 2, dtype=DT, generator=g0) * 4.0,
        sigmas, generator=torch.Generator().manual_seed(7), denoise_final=False,
    )
    g0 = torch.Generator().manual_seed(3)
    x_clean = ddpm_ancestral(
        exact_score(p), torch.randn(500, 2, dtype=DT, generator=g0) * 4.0,
        sigmas, generator=torch.Generator().manual_seed(7), denoise_final=True,
    )
    last = sigmas[-1]
    tweedie = x_noisy + last**2 * exact_score(p)(x_noisy, last)
    assert torch.allclose(x_clean, tweedie, atol=1e-12)
    # ...and on this target that step has a closed form of its own: the score
    # of N(0, (1+sigma^2) I) makes the denoise a pure shrinkage by
    # 1/(1+sigma^2), so the assertion above is pinned to arithmetic rather
    # than to the same code path evaluated twice.
    assert torch.allclose(x_clean, x_noisy / (1.0 + last**2), atol=1e-12)
    # It is also not a no-op: the state was carrying sigma_min of noise, and
    # shrinking removes the predicted fraction of the state's own size.
    shift = float((x_clean - x_noisy).norm(dim=1).mean())
    assert shift == pytest.approx(
        float(x_noisy.norm(dim=1).mean()) * last**2 / (1.0 + last**2), rel=1e-9
    )
    assert shift > 0.0


def test_ancestral_fills_every_mode_of_the_ring_in_one_pass():
    """No equilibration anywhere: one score evaluation per level, once.

    The contrast with :mod:`diffusion.langevin` is the point -- annealed
    Langevin needs many steps *per* level to mix, and plain ULA never crosses
    at all, while the ancestral chain lands the mode weights in a single
    downward pass because the coarse levels have not yet separated the modes.
    """
    ring = GaussianMixture.ring(dtype=DT)
    sigmas = geometric_sigmas(8.0, 0.05, 200)
    g = torch.Generator().manual_seed(0)
    x0 = 8.0 * torch.randn(4_000, 2, dtype=DT, generator=g)
    x = ddpm_ancestral(exact_score(ring), x0, sigmas, generator=g)
    label = torch.round(torch.atan2(x[:, 1], x[:, 0]) / (2 * math.pi / 8)).long() % 8
    occupancy = torch.bincount(label, minlength=8).double() / x.shape[0]
    assert float(occupancy.min()) > 0.10 and float(occupancy.max()) < 0.15
    assert torch.allclose(x.T.cov().diagonal(), ring.cov.diagonal(), rtol=0.05)


def test_an_asymmetric_mixture_gets_its_weights_right():
    """Symmetric targets cannot tell a working sampler from a broken one --
    the same discrimination :mod:`diffusion.langevin`'s tests needed."""
    p = GaussianMixture.bimodal(separation=6.0, std=0.5, weights=(0.25, 0.75), dtype=DT)
    sigmas = geometric_sigmas(8.0, 0.05, 200)
    g = torch.Generator().manual_seed(0)
    x0 = 8.0 * torch.randn(4_000, 2, dtype=DT, generator=g)
    x = ddpm_ancestral(exact_score(p), x0, sigmas, generator=g)
    # 4000 draws -> s.e. 0.007 on the proportion, so 0.025 is ~3.5 s.e.
    assert float((x[:, 0] > 0).double().mean()) == pytest.approx(0.75, abs=0.025)


def test_trajectory_is_the_run_and_lines_up_with_the_final_state():
    p = unit_gaussian()
    sigmas = geometric_sigmas(4.0, 0.1, 10)
    g = torch.Generator().manual_seed(5)
    x0 = 4.0 * torch.randn(64, 2, dtype=DT, generator=g)
    x, traj = ddpm_ancestral(
        exact_score(p), x0, sigmas, generator=torch.Generator().manual_seed(1),
        trajectory=True,
    )
    assert traj.shape == (len(sigmas) + 1, 64, 2)  # 9 steps + final denoise + init
    assert torch.equal(traj[0], x0)
    assert torch.equal(traj[-1], x)


def test_is_reproducible_under_a_seeded_generator():
    p = unit_gaussian()
    sigmas = geometric_sigmas(4.0, 0.1, 12)
    x0 = torch.randn(128, 2, dtype=DT, generator=torch.Generator().manual_seed(2)) * 4
    runs = [
        ddpm_ancestral(exact_score(p), x0, sigmas, generator=torch.Generator().manual_seed(11))
        for _ in range(2)
    ]
    assert torch.equal(runs[0], runs[1])


def test_a_non_finite_score_is_reported_as_the_score_not_the_ladder():
    def broken(x: torch.Tensor, sigma: float) -> torch.Tensor:
        return torch.full_like(x, float("nan"))

    with pytest.raises(FloatingPointError, match="score_fn returned a non-finite"):
        ddpm_ancestral(broken, torch.zeros(4, 2, dtype=DT), geometric_sigmas(4.0, 0.1, 5))


@pytest.mark.parametrize(
    "sigmas, message",
    [
        ([1.0], "at least two levels"),
        ([0.1, 1.0], "decreasing order"),
        ([1.0, 1.0, 0.1], "strictly decreasing"),
        ([1.0, 0.0], "must be positive"),
    ],
)
def test_a_malformed_ladder_is_refused(sigmas, message):
    p = unit_gaussian()
    with pytest.raises(ValueError, match=message):
        ddpm_ancestral(exact_score(p), torch.zeros(4, 2, dtype=DT), sigmas)


def test_the_variance_recursion_refuses_a_non_positive_target():
    with pytest.raises(ValueError, match="target_var must be positive"):
        ddpm_gaussian_variance(0.0, geometric_sigmas(4.0, 0.1, 5))


def test_score_evaluations_are_one_per_level():
    """The NFE accounting the sampler comparison is built on."""
    p = unit_gaussian()
    sigmas = geometric_sigmas(4.0, 0.1, 20)
    calls: list[float] = []

    def counted(x: torch.Tensor, sigma: float) -> torch.Tensor:
        calls.append(sigma)
        return exact_score(p)(x, sigma)

    ddpm_ancestral(counted, torch.zeros(8, 2, dtype=DT), sigmas)
    assert len(calls) == len(sigmas)  # 19 transitions + the final denoise
    assert calls == list(sigmas)
    calls.clear()
    ddpm_ancestral(counted, torch.zeros(8, 2, dtype=DT), sigmas, denoise_final=False)
    assert len(calls) == len(sigmas) - 1


def test_a_single_gaussian_target_matches_the_recursion_off_a_mismatched_prior():
    """``init_var`` is honoured, which is how the prior-mismatch arm is predicted.

    Starting from ``N(0, sigma_max^2)`` rather than from ``p_{sigma_max}``
    changes the answer by a computable amount, and the recursion tracks it.
    """
    p = unit_gaussian()
    sigmas = geometric_sigmas(8.0, 0.05, 32)
    g = torch.Generator().manual_seed(0)
    x0 = sigmas[0] * torch.randn(20_000, 2, dtype=DT, generator=g)  # prior, not p_sigma
    x = ddpm_ancestral(exact_score(p), x0, sigmas, generator=g)
    predicted = ddpm_gaussian_variance(1.0, sigmas, init_var=sigmas[0] ** 2)
    assert float(x.var(dim=0).mean()) == pytest.approx(predicted, rel=0.03)
    # The mismatch survives to the bottom, and is not the same as the matched run.
    assert predicted < ddpm_gaussian_variance(1.0, sigmas)


def test_gaussian_target_score_is_the_one_being_used():
    """Guard on the test helper itself: the one-component mixture really is N(0, I)."""
    p = unit_gaussian()
    x = torch.randn(16, 2, dtype=DT, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(p.score(x), Gaussian(torch.zeros(2), torch.eye(2), dtype=DT).score(x))
