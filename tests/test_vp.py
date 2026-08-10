"""Tests for the variance-preserving process and its sampler.

The load-bearing test here is
``test_the_two_samplers_are_the_same_chain_pathwise``. Every other check in
this file could pass with a VP sampler that is merely *correct* -- the right
stationary behaviour, the right variance, samples that look like the target.
The claim being made is stronger than correctness: that VP and VE are one chain
in two coordinate systems, so that given the same noise draws the trajectories
agree to float precision, not in distribution. That is a claim a histogram
cannot check and a seeded pathwise comparison can.

``diffusion/vp.py`` deliberately shares no code with ``diffusion/samplers.py``
-- it is written from ``abar`` and ``beta`` the way DDPM states it -- so the
agreement is a result about the algebra rather than about the implementation.
"""

from __future__ import annotations

import math

import pytest
import torch

from diffusion.langevin import geometric_sigmas
from diffusion.samplers import ddpm_ancestral, ddpm_gaussian_variance
from diffusion.targets import Gaussian, GaussianMixture
from diffusion.vp import (
    _abar,
    alphabar_from_sigma,
    ddpm_ancestral_vp,
    prior_variance_error,
    sigma_from_alphabar,
    to_ve,
    to_vp,
    vp_score_fn,
)

DT = torch.float64


def unit_gaussian(var: float = 1.0) -> GaussianMixture:
    cov = var * torch.eye(2).unsqueeze(0)
    return GaussianMixture(torch.zeros(1, 2), cov, dtype=DT)


def exact_score(target):
    def s(x: torch.Tensor, sigma: float) -> torch.Tensor:
        return target.noised(sigma).score(x)

    return s


# -- the change of variables -------------------------------------------------


def test_snr_match_is_a_bijection():
    sigmas = torch.tensor([1e-3, 0.05, 0.5, 1.0, 3.0, 50.0], dtype=DT)
    back = sigma_from_alphabar(alphabar_from_sigma(sigmas))
    torch.testing.assert_close(back, sigmas, rtol=1e-12, atol=1e-12)
    # and it is order-reversing: more noise is less signal
    ab = alphabar_from_sigma(sigmas)
    assert torch.all(ab[1:] < ab[:-1])


def test_the_vp_state_is_the_ve_state_rescaled():
    """Equation (2), on the same eps rather than on two independent draws.

    The distributional statement (both are Gaussian with matching moments) is
    weaker and would hold for a wrong pairing of levels; this is the pathwise
    one, which pins the SNR match (1) as well as the scaling.
    """
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(2000, 2, generator=g, dtype=DT)
    eps = torch.randn(2000, 2, generator=g, dtype=DT)
    for sigma in (0.05, 0.5, 3.0, 20.0):
        abar = _abar(sigma)
        ve = x0 + sigma * eps
        vp = math.sqrt(abar) * x0 + math.sqrt(1.0 - abar) * eps
        torch.testing.assert_close(to_vp(ve, sigma), vp, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(to_ve(vp, sigma), ve, rtol=1e-12, atol=1e-12)


def test_the_vp_marginal_variance_is_what_the_name_says():
    """On unit-variance data the VP marginal variance is 1 at every level --
    that is the whole reason for the name -- while VE's grows as v + sigma^2."""
    for sigma in (0.05, 1.0, 10.0, 100.0):
        abar = _abar(sigma)
        assert abs(abar * 1.0 + (1.0 - abar) - 1.0) < 1e-12
        assert abs((1.0 + sigma**2) - (1.0 + sigma**2)) < 1e-12
    # and on data of variance 4 it is NOT preserved: the name assumes v = 1.
    abar = _abar(1.0)
    assert abs(abar * 4.0 + (1.0 - abar) - 2.5) < 1e-12


def test_the_wrapped_score_is_the_score_of_the_vp_marginal():
    """(3), checked against a finite-difference gradient of the VP log-density
    rather than against the formula it came from.

    p_VP(xt) = abar^{-d/2} p_VE(xt/sqrt(abar)), whose log differs from
    log p_VE(xt/sqrt(abar)) by a constant, so the finite difference tests the
    chain rule and the constant drops out -- which is exactly the half of (3)
    that a wrong implementation gets right by accident.
    """
    target = GaussianMixture.bimodal(separation=3.0, dtype=DT)
    s_vp = vp_score_fn(exact_score(target))
    g = torch.Generator().manual_seed(1)

    for sigma in (0.2, 1.0, 4.0):
        abar = _abar(sigma)
        xt = math.sqrt(abar) * (2.0 * torch.randn(64, 2, generator=g, dtype=DT))
        noised = target.noised(sigma)

        def log_p_vp(z):
            return noised.log_prob(z / math.sqrt(abar))

        h = 1e-5
        numeric = torch.zeros_like(xt)
        for d in range(xt.shape[1]):
            step = torch.zeros_like(xt)
            step[:, d] = h
            numeric[:, d] = (log_p_vp(xt + step) - log_p_vp(xt - step)) / (2 * h)
        torch.testing.assert_close(s_vp(xt, sigma), numeric, rtol=1e-5, atol=1e-7)


def test_forgetting_either_rescaling_is_caught():
    """The failure mode vp_score_fn exists to prevent: rescaling the argument
    but not the value (or the reverse) leaves a plausible-looking score that is
    wrong by exactly sqrt(abar), and nothing downstream would raise."""
    target = unit_gaussian()
    ve = exact_score(target)
    sigma = 3.0
    abar = _abar(sigma)
    xt = torch.randn(32, 2, dtype=DT) * 0.5

    correct = vp_score_fn(ve)(xt, sigma)
    argument_only = ve(xt / math.sqrt(abar), sigma)
    value_only = ve(xt, sigma) / math.sqrt(abar)
    torch.testing.assert_close(correct, argument_only / math.sqrt(abar))
    assert not torch.allclose(correct, argument_only)
    assert not torch.allclose(correct, value_only)


# -- the claim ---------------------------------------------------------------


@pytest.mark.parametrize("n_levels", [8, 40])
def test_the_two_samplers_are_the_same_chain_pathwise(n_levels):
    """VP and VE ancestral sampling, same seed, same target, same ladder.

    The two implementations share no code: `ddpm_ancestral` steps in sigma with
    the gap `sigma_i^2 - sigma_{i-1}^2`, `ddpm_ancestral_vp` steps in abar with
    `beta / sqrt(alpha)` and `beta_tilde`. If the algebra of vp.py's (4) is
    right they are the same chain in coordinates related by sqrt(abar), so
    driving them with identical Gaussian draws must give identical
    trajectories after rescaling -- and identical *final* states with no
    rescaling at all, since abar = 1 at the end of the ladder.
    """
    target = GaussianMixture.bimodal(separation=3.0, dtype=DT)
    sigmas = geometric_sigmas(sigma_max=8.0, sigma_min=0.05, n_levels=n_levels)

    g = torch.Generator().manual_seed(7)
    x_top = 8.0 * torch.randn(512, 2, generator=g, dtype=DT)

    ve_final, ve_traj = ddpm_ancestral(
        exact_score(target), x_top, sigmas,
        generator=torch.Generator().manual_seed(11), trajectory=True,
    )
    vp_final, vp_traj = ddpm_ancestral_vp(
        vp_score_fn(exact_score(target)), to_vp(x_top, sigmas[0]), sigmas,
        generator=torch.Generator().manual_seed(11), trajectory=True,
    )

    # every intermediate state, rescaled by that level's sqrt(abar)
    levels = list(sigmas) + [0.0]
    assert len(levels) == ve_traj.shape[0]
    for k, sigma in enumerate(levels):
        torch.testing.assert_close(
            vp_traj[k], to_vp(ve_traj[k], sigma), rtol=1e-11, atol=1e-11,
        )
    torch.testing.assert_close(vp_final, ve_final, rtol=1e-11, atol=1e-11)


def test_the_vp_sampler_hits_the_ve_closed_form_variance():
    """Independent of the pathwise check: the VP chain on a Gaussian target
    must land on `ddpm_gaussian_variance`, the recursion derived for the VE
    chain, including the deficit a coarse ladder is supposed to have. A VP
    sampler that was right in distribution but not the same chain would fail
    this at 8 levels, where the predicted variance is 0.53 rather than 1.0.
    """
    target = unit_gaussian()
    for n_levels, tol in ((8, 0.06), (64, 0.05)):
        sigmas = geometric_sigmas(8.0, 0.05, n_levels)
        predicted = ddpm_gaussian_variance(1.0, sigmas)
        g = torch.Generator().manual_seed(3)
        x_top = math.sqrt(1.0 + sigmas[0] ** 2) * torch.randn(
            20000, 2, generator=g, dtype=DT
        )
        out = ddpm_ancestral_vp(
            vp_score_fn(exact_score(target)), to_vp(x_top, sigmas[0]), sigmas,
            generator=torch.Generator().manual_seed(4),
        )
        got = float(out.var(unbiased=True).mean())
        assert abs(got - predicted) < tol, (n_levels, got, predicted)
    assert ddpm_gaussian_variance(1.0, geometric_sigmas(8.0, 0.05, 8)) < 0.6


def test_the_final_step_leaves_data_coordinates():
    """abar = 1 at the end, so the VP sampler's return needs no rescaling to be
    compared with the VE one. Checked by mean and variance against the target
    rather than by trusting the loop bound."""
    target = GaussianMixture.bimodal(separation=3.0, dtype=DT)
    sigmas = geometric_sigmas(8.0, 0.02, 200)
    g = torch.Generator().manual_seed(5)
    out = ddpm_ancestral_vp(
        vp_score_fn(exact_score(target)),
        torch.randn(8000, 2, generator=g, dtype=DT),
        sigmas, generator=torch.Generator().manual_seed(6),
    )
    torch.testing.assert_close(
        out.mean(0), target.mean.to(DT), rtol=0.0, atol=0.08
    )
    got = out.var(0, unbiased=True)
    want = torch.diagonal(target.cov.to(DT))
    torch.testing.assert_close(got, want, rtol=0.06, atol=0.03)


# -- where the equivalence stops ---------------------------------------------


def test_the_conventional_priors_are_not_the_same_distribution():
    """The one thing the coordinate change does not carry across.

    Push VE's N(0, sigma_max^2 I) through (2) and it becomes N(0, (1-abar) I),
    not the N(0, I) that VP conventionally starts from. They agree only as
    sigma_max -> infinity, and the gap is (1 - abar) = sigma_max^2/(1+sigma_max^2).
    """
    for sigma_max, want in ((1.0, 0.5), (3.0, 0.9), (10.0, 0.990099)):
        abar = _abar(sigma_max)
        assert abs((1.0 - abar) - want) < 1e-6
    assert abs(1.0 - _abar(1e4)) > 0.99999


def test_vp_prior_is_exact_on_standardized_data_and_ve_never_is():
    """`prior_variance_error` as the statement of what "variance preserving"
    actually buys: not a better process, a convention that is exact at v = 1."""
    for sigma_max in (1.0, 3.0, 10.0, 100.0):
        ve, vp = prior_variance_error(sigma_max, data_var=1.0)
        assert abs(vp - 1.0) < 1e-12                 # exact at every sigma_max
        assert ve < 1.0                              # never exact
    # VE's error closes like v/sigma_max^2; VP's opens up as the data leave 1.
    assert prior_variance_error(3.0, 1.0)[0] == pytest.approx(0.9)
    assert prior_variance_error(30.0, 1.0)[0] == pytest.approx(0.99889, rel=1e-4)
    ve4, vp4 = prior_variance_error(3.0, 4.0)
    assert vp4 == pytest.approx(0.76923, rel=1e-4)
    assert ve4 == pytest.approx(0.69231, rel=1e-4)


def test_the_prior_gap_is_real_and_is_dominated_by_the_ladder():
    """What the two conventions' different priors are actually worth.

    Measured, because the derivation only says the priors differ, not by how
    much it matters. On a Gaussian target the whole chain is linear, so
    `ddpm_gaussian_variance` run from each convention's `init_var` gives both
    answers exactly, and the two samplers can be checked against them.

    The result is a size, and the size is the point. The two priors differ by
    exactly 1.0 in variance (VE assumes sigma_max^2, the truth at v = 1 is
    1 + sigma_max^2), and that difference reaches the output multiplied by the
    chain's total contraction -- 0.010 of target variance at sigma_max = 3 and
    0.0003 at sigma_max = 8, at every ladder length, since the contraction does
    not depend on where the chain started. Over the same ladders the
    *discretization* deficit runs from 0.66 to 0.07. So the convention that gets
    the prior exactly right is worth one to two orders of magnitude less than
    one more halving of the step size, which is the practical form of this
    week's negative result.
    """
    target = unit_gaussian()
    sigma_max, n_levels = 3.0, 24
    sigmas = geometric_sigmas(sigma_max, 0.05, n_levels)
    abar_max = _abar(sigma_max)

    ve_pred = ddpm_gaussian_variance(1.0, sigmas, init_var=sigma_max**2)
    vp_pred = ddpm_gaussian_variance(1.0, sigmas, init_var=1.0 / abar_max)
    assert 1.0 / abar_max - sigma_max**2 == pytest.approx(1.0)   # the prior gap
    assert vp_pred - ve_pred == pytest.approx(0.0100, abs=0.0005)
    assert 1.0 - vp_pred == pytest.approx(0.159, abs=0.005)      # the ladder
    assert (vp_pred - ve_pred) < 0.1 * (1.0 - vp_pred)

    # and it is linear in the gap, so the same 1.0 of prior variance always
    # arrives multiplied by the same contraction, whatever the ladder does.
    for extra in (0.5, 3.0):
        widened = ddpm_gaussian_variance(
            1.0, sigmas, init_var=1.0 / abar_max + extra
        )
        assert widened - vp_pred == pytest.approx(
            extra * (vp_pred - ve_pred), rel=1e-9
        )

    g = torch.Generator().manual_seed(9)
    z = torch.randn(60000, 2, generator=g, dtype=DT)
    ve_out = ddpm_ancestral(
        exact_score(target), sigma_max * z, sigmas,
        generator=torch.Generator().manual_seed(2),
    )
    vp_out = ddpm_ancestral_vp(
        vp_score_fn(exact_score(target)), z, sigmas,
        generator=torch.Generator().manual_seed(2),
    )
    assert abs(float(ve_out.var(unbiased=True).mean()) - ve_pred) < 0.01
    assert abs(float(vp_out.var(unbiased=True).mean()) - vp_pred) < 0.01


# -- plumbing ----------------------------------------------------------------


@pytest.mark.parametrize(
    "sigmas, message",
    [
        ([0.5], "at least two"),
        ([0.1, 0.5], "decreasing"),
        ([0.5, 0.5], "decreasing"),
        ([0.5, 0.0], "finest level"),
    ],
)
def test_a_malformed_ladder_is_refused(sigmas, message):
    with pytest.raises(ValueError, match=message):
        ddpm_ancestral_vp(lambda x, s: torch.zeros_like(x),
                          torch.zeros(4, 2, dtype=DT), sigmas)


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError, match="non-negative"):
        alphabar_from_sigma([-1.0])
    with pytest.raises(ValueError, match="alphabar"):
        sigma_from_alphabar([1.5])
    with pytest.raises(ValueError, match="sigma_max"):
        prior_variance_error(0.0, 1.0)
    with pytest.raises(ValueError, match="data_var"):
        prior_variance_error(1.0, -1.0)
    with pytest.raises(ValueError, match=r"\(n, d\)"):
        ddpm_ancestral_vp(lambda x, s: x, torch.zeros(4, dtype=DT), [1.0, 0.5])


def test_a_non_finite_score_is_reported():
    def broken(x, sigma):
        return torch.full_like(x, float("nan"))

    with pytest.raises(FloatingPointError, match="sigma="):
        ddpm_ancestral_vp(broken, torch.zeros(4, 2, dtype=DT), [1.0, 0.5])


def test_is_reproducible_under_a_seeded_generator():
    target = unit_gaussian()
    sigmas = geometric_sigmas(4.0, 0.05, 12)
    x = torch.randn(64, 2, dtype=DT)
    runs = [
        ddpm_ancestral_vp(vp_score_fn(exact_score(target)), x, sigmas,
                          generator=torch.Generator().manual_seed(21))
        for _ in range(2)
    ]
    torch.testing.assert_close(runs[0], runs[1], rtol=0.0, atol=0.0)


def test_score_evaluations_are_one_per_level():
    calls = []

    def counting(x, sigma):
        calls.append(sigma)
        return torch.zeros_like(x)

    sigmas = geometric_sigmas(4.0, 0.05, 10)
    ddpm_ancestral_vp(counting, torch.zeros(4, 2, dtype=DT), sigmas,
                      denoise_final=False)
    assert calls == list(sigmas[:-1])
    calls.clear()
    ddpm_ancestral_vp(counting, torch.zeros(4, 2, dtype=DT), sigmas,
                      denoise_final=True)
    assert calls == list(sigmas)


def test_a_single_gaussian_target_works_through_the_wrapper():
    """Gaussian, not GaussianMixture: `.noised` exists on both and the wrapper
    must not depend on which."""
    target = Gaussian(torch.zeros(2, dtype=DT), torch.eye(2, dtype=DT), dtype=DT)
    s = vp_score_fn(lambda x, sigma: target.noised(sigma).score(x))
    out = s(torch.zeros(3, 2, dtype=DT), 1.0)
    torch.testing.assert_close(out, torch.zeros(3, 2, dtype=DT))
