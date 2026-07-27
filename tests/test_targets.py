"""Tests for the closed-form targets.

The load-bearing claim of this repo is that these targets' scores are exactly
right, because everything later is measured against them. So every score here
is checked against central differences of its own ``log_prob`` (in float64,
where the O(h^2) truncation floor at h=1e-4 is ~1e-8), every normalizer is
checked against numerical quadrature, and the Gaussian mixture's closure under
noising -- the property the whole repo rests on -- is checked against an
independent Monte Carlo estimate of the convolution rather than against the
same algebra restated.
"""

from __future__ import annotations

import math

import pytest
import torch

from diffusion.targets import (
    Crescent,
    Gaussian,
    GaussianMixture,
    Mixture,
    TwoMoons,
    bessel_i0,
    finite_difference_score,
)

DT = torch.float64


def grid(lo: float, hi: float, n: int) -> tuple[torch.Tensor, float]:
    """Points of a square quadrature grid and its area element."""
    g = torch.linspace(lo, hi, n, dtype=DT)
    X, Y = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1), float(g[1] - g[0]) ** 2


def integrate(target, lo: float, hi: float, n: int = 900) -> float:
    pts, dA = grid(lo, hi, n)
    return float(target.log_prob(pts).exp().sum() * dA)


# --------------------------------------------------------------------------
# Gaussian
# --------------------------------------------------------------------------
def test_gaussian_log_prob_matches_the_written_out_formula():
    cov = torch.tensor([[2.0, 0.6], [0.6, 0.5]], dtype=DT)
    g = Gaussian([1.0, -2.0], cov, dtype=DT)
    x = torch.tensor([[0.3, -1.1], [1.0, -2.0], [4.0, 3.0]], dtype=DT)
    d = x - g.mean
    quad = torch.einsum("ni,ij,nj->n", d, torch.linalg.inv(cov), d)
    expected = -0.5 * (2 * math.log(2 * math.pi) + torch.logdet(cov) + quad)
    assert torch.allclose(g.log_prob(x), expected, atol=1e-12)


def test_gaussian_score_matches_finite_differences():
    g = Gaussian([1.0, -2.0], [[2.0, 0.6], [0.6, 0.5]], dtype=DT)
    x = torch.tensor([[0.3, -1.1], [1.0, -2.0], [3.0, 0.5], [-1.0, -3.0]], dtype=DT)
    assert float(g.score_error(x)) < 1e-6


def test_gaussian_sample_recovers_its_moments():
    g = Gaussian([1.0, -2.0], [[2.0, 0.6], [0.6, 0.5]], dtype=DT)
    s = g.sample(200_000, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(s.mean(0), g.mean, atol=0.02)
    assert torch.allclose(s.T.cov(), g.cov, atol=0.03)


def test_gaussian_noising_adds_sigma_squared_to_the_covariance():
    g = Gaussian([1.0, -2.0], [[2.0, 0.6], [0.6, 0.5]], dtype=DT)
    n = g.noised(0.7)
    assert torch.allclose(n.cov, g.cov + 0.49 * torch.eye(2, dtype=DT), atol=1e-12)
    assert torch.allclose(n.mean, g.mean)


def test_gaussian_rejects_a_non_psd_covariance():
    with pytest.raises(Exception):
        Gaussian([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]], dtype=DT)


# --------------------------------------------------------------------------
# mixtures: the responsibility-weighted score
# --------------------------------------------------------------------------
def test_mixture_score_matches_finite_differences():
    gm = GaussianMixture.ring(dtype=DT)
    x = torch.tensor(
        [[0.5, -1.2], [3.0, 2.0], [-4.0, 0.3], [0.0, 0.0], [4.0, 0.0], [7.0, 7.0]],
        dtype=DT,
    )
    assert float(gm.score_error(x)) < 1e-6


def test_mixture_score_is_the_responsibility_weighted_average():
    """The derivation, checked term by term against the vectorized code."""
    gm = GaussianMixture.ring(n_components=4, dtype=DT)
    x = torch.tensor([[1.0, 1.0], [-3.0, 0.4], [0.2, -2.0]], dtype=DT)
    r = gm.responsibilities(x)
    assert torch.allclose(r.sum(dim=1), torch.ones(3, dtype=DT), atol=1e-12)
    by_hand = sum(r[:, k].unsqueeze(1) * gm.components[k].score(x) for k in range(4))
    assert torch.allclose(gm.score(x), by_hand, atol=1e-12)


def test_far_from_everything_the_mixture_score_is_the_nearest_components():
    """The tail claim in the docstring: responsibilities saturate, so the
    mixture score degenerates to a single Gaussian's rather than underflowing.
    """
    gm = GaussianMixture.ring(dtype=DT)
    x = torch.tensor([[60.0, 0.0]], dtype=DT)  # ~140 component sd from every mode
    nearest = int(torch.argmax(gm.responsibilities(x)))
    assert torch.allclose(gm.score(x), gm.components[nearest].score(x), atol=1e-10)
    assert torch.isfinite(gm.score(x)).all()


def test_mixture_density_integrates_to_one():
    gm = GaussianMixture.ring(dtype=DT)
    assert abs(integrate(gm, -8.0, 8.0) - 1.0) < 1e-6


def test_mixture_exact_moments_match_its_own_samples():
    gm = GaussianMixture.bimodal(weights=(0.3, 0.7), dtype=DT)
    s = gm.sample(200_000, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(s.mean(0), gm.mean, atol=0.03)
    assert torch.allclose(s.T.cov(), gm.cov, atol=0.1)
    # the between-component term dominates: Var(x0) >> any component's
    assert float(gm.cov[0, 0]) > 10 * float(gm.covs[0, 0, 0])


def test_mixture_samples_are_not_sorted_by_component():
    """Block-wise generation must be scattered back to the drawn positions."""
    gm = GaussianMixture.bimodal(dtype=DT)
    s = gm.sample(400, generator=torch.Generator().manual_seed(1))
    labels = (s[:, 0] > 0).long()
    assert 0 < int(labels.sum()) < 400
    assert not bool(torch.all(labels[:-1] <= labels[1:]))  # would be sorted


def test_sampling_is_reproducible_under_a_seeded_generator():
    gm = GaussianMixture.ring(dtype=DT)
    a = gm.sample(50, generator=torch.Generator().manual_seed(7))
    b = gm.sample(50, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_mixture_rejects_inconsistent_components():
    g2 = Gaussian([0.0, 0.0], torch.eye(2, dtype=DT), dtype=DT)
    g3 = Gaussian([0.0, 0.0, 0.0], torch.eye(3, dtype=DT), dtype=DT)
    with pytest.raises(ValueError, match="dimension"):
        Mixture([g2, g3])
    with pytest.raises(ValueError, match="positive"):
        Mixture([g2, g2], weights=[1.0, 0.0])


def test_targets_reject_badly_shaped_points():
    gm = GaussianMixture.ring(dtype=DT)
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        gm.log_prob(torch.zeros(5, 3, dtype=DT))
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        gm.score(torch.zeros(5, dtype=DT))


# --------------------------------------------------------------------------
# closure under noising -- the property the repo is built on
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [0.25, 1.0, 3.0])
def test_noised_mixture_score_matches_finite_differences(sigma):
    gm = GaussianMixture.ring(dtype=DT).noised(sigma)
    x = torch.tensor([[0.5, -1.2], [3.0, 2.0], [-4.0, 0.3], [0.0, 0.0]], dtype=DT)
    assert float(gm.score_error(x)) < 1e-6


def test_noised_mixture_is_the_actual_convolution():
    """Independent oracle for the closure claim.

    ``p_sigma(x) = E_{x0 ~ p}[ N(x; x0, sigma^2 I) ]`` by definition of the
    convolution. Estimating that expectation by Monte Carlo over draws from
    ``p`` is a completely different computation from "add sigma^2 to every
    component covariance", so agreement is evidence about the claim rather
    than about the algebra being restated.
    """
    gm = GaussianMixture.ring(dtype=DT)
    sigma = 1.2
    x = torch.tensor([[0.0, 0.0], [4.0, 0.0], [2.5, 2.5], [-3.0, 1.0]], dtype=DT)
    x0 = gm.sample(400_000, generator=torch.Generator().manual_seed(0))
    # log N(x; x0, sigma^2 I) for every (x, x0) pair, then average over x0.
    d2 = torch.cdist(x, x0) ** 2
    log_kern = -d2 / (2 * sigma**2) - math.log(2 * math.pi * sigma**2)
    mc = torch.logsumexp(log_kern, dim=1) - math.log(x0.shape[0])
    analytic = gm.noised(sigma).log_prob(x)
    assert torch.allclose(mc.exp(), analytic.exp(), rtol=0.02)


def test_noising_by_zero_is_the_identity_and_covariances_add():
    gm = GaussianMixture.ring(dtype=DT)
    assert torch.allclose(gm.noised(0.0).covs, gm.covs)
    n = gm.noised(2.0)
    assert torch.allclose(n.cov, gm.cov + 4.0 * torch.eye(2, dtype=DT), atol=1e-10)
    assert torch.allclose(n.mean, gm.mean, atol=1e-12)
    with pytest.raises(ValueError):
        gm.noised(-1.0)


def test_noising_flattens_the_barrier_between_modes():
    """Why annealing works, stated as a number.

    The log-density gap between a mode and the midpoint between modes is what
    a Langevin chain has to climb. Noising must shrink it monotonically, and
    past the mode separation it must essentially vanish.
    """
    gm = GaussianMixture.bimodal(separation=6.0, std=0.5, dtype=DT)
    mode = torch.tensor([[3.0, 0.0]], dtype=DT)
    saddle = torch.tensor([[0.0, 0.0]], dtype=DT)
    gaps = [
        float(gm.noised(s).log_prob(mode) - gm.noised(s).log_prob(saddle))
        for s in (0.0, 0.5, 1.0, 1.5, 2.0)
    ]
    assert gaps[0] > 15.0  # untouched: 17.3 nats, a barrier no local sampler crosses
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 0.5
    # Past the mode separation the mixture has merged into what is nearly a
    # single wide Gaussian centred at the origin, so the gap goes slightly
    # NEGATIVE (the old mode is now off-centre and lower). What matters is
    # that its magnitude is negligible: there is no barrier left to cross.
    assert abs(float(gm.noised(8.0).log_prob(mode) - gm.noised(8.0).log_prob(saddle))) < 0.1


# --------------------------------------------------------------------------
# the crescent and the two moons
# --------------------------------------------------------------------------
def test_bessel_series_matches_an_independent_implementation():
    for k in (0.0, 0.25, 1.0, 3.5, 10.0, 40.0):
        ref = float(torch.special.i0(torch.tensor(k, dtype=DT)))
        assert abs(bessel_i0(k) - ref) <= 1e-12 * ref


def test_bessel_refuses_outside_its_range():
    with pytest.raises(ValueError):
        bessel_i0(-1.0)
    with pytest.raises(ValueError):
        bessel_i0(100.0)


def test_crescent_score_matches_finite_differences():
    c = Crescent(dtype=DT)
    x = torch.tensor(
        [[0.0, 1.0], [0.9, 0.4], [-0.8, 0.6], [0.3, 1.1], [0.0, -1.0]], dtype=DT
    )
    assert float(c.score_error(x)) < 1e-5


def test_crescent_analytic_normalizer_matches_quadrature():
    """The closed form for Z, against a 2D grid integral of the same density."""
    for kappa in (0.0, 1.0, 3.0):
        c = Crescent(kappa=kappa, dtype=DT)
        assert abs(integrate(c, -2.0, 2.0, 1200) - 1.0) < 1e-6


def test_crescent_sampler_agrees_with_its_density():
    """Sampler vs density, compared through moments computed by quadrature --
    two independent routes to the same numbers."""
    c = Crescent(dtype=DT)
    pts, dA = grid(-2.0, 2.0, 1200)
    w = c.log_prob(pts).exp() * dA
    quad_mean = (w.unsqueeze(1) * pts).sum(0)
    s = c.sample(200_000, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(s.mean(0), quad_mean, atol=0.01)
    # The radial moments are closed-form too, and they are NOT (R, s): the
    # polar area element r dr multiplies the Gaussian ring, which pushes mass
    # outward. With u = r - R and the r < 0 tail negligible at R/s = 6.7,
    #     E[r]   = (R^2 + s^2)/R      = R + s^2/R,
    #     Var[r] = R^2 + 3 s^2 - E[r]^2 = s^2 (1 - s^2/R^2).
    # Asserting R and s here instead would be asserting that the area element
    # was forgotten -- which is exactly the bug this catches.
    r = torch.linalg.norm(s, dim=1)
    R, w = c.radius, c.width
    assert float(r.mean()) == pytest.approx(R + w * w / R, abs=1e-3)
    assert float(r.std()) == pytest.approx(w * math.sqrt(1 - w * w / R**2), abs=1e-3)


def test_two_moons_is_a_normalized_density_with_the_right_score():
    m = TwoMoons(dtype=DT)
    x = torch.tensor(
        [[0.0, 1.0], [1.0, -0.5], [0.9, 0.4], [-0.9, 0.2], [0.5, 0.25]], dtype=DT
    )
    assert float(m.score_error(x)) < 1e-5
    assert abs(integrate(m, -2.0, 3.0, 1200) - 1.0) < 1e-6


def test_two_moons_components_share_a_normalizer():
    m = TwoMoons(dtype=DT)
    assert m.components[0].log_norm == pytest.approx(m.components[1].log_norm, rel=1e-15)


def test_two_moons_is_symmetric_under_the_point_reflection_that_swaps_them():
    """``x -> (R, offset) - x`` maps each moon onto the other, so the density
    is invariant. This catches an orientation or offset error that a
    finite-difference check on the score alone would not.
    """
    m = TwoMoons(dtype=DT)
    x = torch.tensor([[0.0, 1.0], [1.0, -0.5], [0.4, 0.8], [-0.7, 0.5]], dtype=DT)
    pivot = torch.tensor([m.radius, m.offset], dtype=DT)
    assert torch.allclose(m.log_prob(x), m.log_prob(pivot - x), atol=1e-12)
    assert torch.allclose(m.score(x), -m.score(pivot - x), atol=1e-10)


def test_finite_difference_helper_is_itself_right():
    """The oracle every other test leans on, checked on a function whose
    gradient is trivial: log N(0, I) has score -x."""
    x = torch.tensor([[0.3, -1.2], [2.0, 0.5]], dtype=DT)
    fd = finite_difference_score(lambda p: -0.5 * (p**2).sum(dim=1), x)
    assert torch.allclose(fd, -x, atol=1e-8)
