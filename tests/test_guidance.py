"""Classifier-free guidance is only measurable here because both of its terms
are exact, so the tests are mostly about pinning that: the guided score really
is the gradient of the tilted log density, at every noise level, and the two
limiting weights really do collapse to the two scores they are built from.

The grid normalizer gets its own checks. It is the reference every distance in
``experiments/guidance.py`` is measured against, so a normalizer that depended
on the box, or a sampler that did not draw from the density it normalized,
would move every number in that experiment without moving any of them
implausibly.
"""

import math

import pytest
import torch

from diffusion.guidance import (
    TiltedGrid,
    check_tiltable,
    component_purity,
    conditional_score,
    guided_score,
    guided_score_fn,
    guided_stationary_point,
    radial_offset,
    tilted_log_density,
)
from diffusion.targets import GaussianMixture, finite_difference_score

WEIGHTS = [0.0, 0.5, 1.0, 3.0]


def ring():
    return GaussianMixture.ring(n_components=6, radius=4.0, std=0.5)


def probe_points(n=64, scale=5.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return scale * (torch.rand(n, 2, generator=g) * 2 - 1)


# --------------------------------------------------------------------------
# the identity everything rests on
# --------------------------------------------------------------------------
@pytest.mark.parametrize("w", WEIGHTS)
@pytest.mark.parametrize("sigma", [0.0, 0.3, 1.5])
def test_guided_score_is_the_gradient_of_the_tilted_log_density(w, sigma):
    """``s_w = grad log r_{w,sigma}`` by central differences.

    This is the whole claim of the module: the sampler is handed a score, and
    what it converges to is decided by which density that score belongs to. If
    the combination were, say, ``(1+w) s_cond - w s_uncond`` written with the
    weights transposed, every sample would still look like a plausible mixture
    draw and only this check would notice.
    """
    mix = ring()
    x = probe_points().to(torch.float64)
    mix64 = GaussianMixture(mix.means, mix.covs, mix.weights, dtype=torch.float64)

    fd = finite_difference_score(
        lambda p: tilted_log_density(mix64, 2, p, w, sigma), x, h=1e-4
    )
    exact = guided_score(mix64, 2, x, sigma, w)
    torch.testing.assert_close(exact, fd, rtol=2e-4, atol=2e-5)


def test_weight_zero_is_exactly_the_conditional_score():
    """``w = 0`` must return the conditional score with the unconditional term
    not merely small but absent -- bitwise, since the code short-circuits."""
    mix = ring()
    x = probe_points()
    for sigma in (0.0, 0.7):
        got = guided_score(mix, 1, x, sigma, 0.0)
        want = conditional_score(mix, 1, x, sigma)
        assert torch.equal(got, want)


def test_weight_minus_one_is_exactly_the_unconditional_score():
    """``w = -1`` cancels the conditional term. Not a setting anyone uses, but
    it is the other end of the interpolation and pins the arithmetic: a sign
    error or a swapped coefficient breaks this and leaves w = 0 intact."""
    mix = ring()
    x = probe_points()
    for sigma in (0.0, 0.7):
        got = guided_score(mix, 3, x, sigma, -1.0)
        torch.testing.assert_close(got, mix.noised(sigma).score(x), rtol=1e-5, atol=1e-6)


def test_conditional_score_is_the_noised_component_not_the_clean_one():
    """At ``sigma > 0`` the conditional must use ``Sigma_k + sigma^2 I``.

    Forgetting to noise the conditional is the single most plausible bug here
    -- it produces a score that is right at sigma = 0, wrong by a smooth factor
    everywhere else, and samples that look fine. Checked against a Gaussian
    built by hand at the noised covariance.
    """
    mix = ring()
    x = probe_points()
    sigma = 0.8
    k = 4
    var = float(mix.covs[k, 0, 0]) + sigma ** 2
    want = -(x - mix.means[k]) / var
    torch.testing.assert_close(conditional_score(mix, k, x, sigma), want,
                               rtol=1e-5, atol=1e-6)


def test_guided_score_fn_matches_the_direct_call():
    mix = ring()
    x = probe_points()
    fn = guided_score_fn(mix, 0, 2.0)
    for sigma in (0.1, 1.0):
        torch.testing.assert_close(fn(x, sigma), guided_score(mix, 0, x, sigma, 2.0))


def test_guided_score_rejects_a_bad_component_or_sigma():
    mix = ring()
    x = probe_points()
    with pytest.raises(IndexError):
        conditional_score(mix, 99, x, 0.1)
    with pytest.raises(ValueError):
        conditional_score(mix, 0, x, -0.1)


# --------------------------------------------------------------------------
# normalizability, and the condition that buys it
# --------------------------------------------------------------------------
def test_check_tiltable_accepts_the_ring_and_returns_its_variance():
    mix = ring()
    assert check_tiltable(mix) == pytest.approx(0.25)


def test_check_tiltable_refuses_unequal_covariances():
    """Different component covariances break the quadratic cancellation, so the
    tilted density need not be normalizable and the grid normalizer would
    silently depend on the box. Refused up front."""
    means = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
    covs = torch.stack([torch.eye(2), 4.0 * torch.eye(2)])
    with pytest.raises(ValueError, match="isotropic"):
        check_tiltable(GaussianMixture(means, covs))


@pytest.mark.parametrize("w", [0.0, 1.0, 4.0])
def test_tilted_density_decays_so_the_grid_normalizer_means_something(w):
    """The mass in the outermost ring of grid cells must be negligible.

    This is the measurement behind the normalizability argument in the module
    docstring: if the quadratic terms did not cancel, the tilt would grow along
    the direction away from component k and this number would not be small.
    """
    grid = TiltedGrid.build(ring(), 0, w, half_width=9.0, resolution=256)
    assert grid.tail_mass() < 1e-8, (w, grid.tail_mass())


def test_grid_normalizer_is_insensitive_to_the_box_and_the_resolution():
    """Doubling the box and doubling the resolution must not move ``log_norm``.

    The normalizer is the denominator of every density comparison downstream,
    so it has to be a property of ``q_w`` rather than of the grid it was
    computed on. Resolution moves it at the quadrature's own order, which is
    why the tolerance on the refinement is looser than on the box.
    """
    mix = ring()
    a = TiltedGrid.build(mix, 0, 2.0, half_width=8.0, resolution=256)
    wide = TiltedGrid.build(mix, 0, 2.0, half_width=16.0, resolution=512)
    fine = TiltedGrid.build(mix, 0, 2.0, half_width=8.0, resolution=512)
    assert a.log_norm == pytest.approx(wide.log_norm, abs=1e-6)
    assert a.log_norm == pytest.approx(fine.log_norm, abs=1e-3)


def test_grid_normalizer_matches_the_closed_form_at_weight_zero():
    """At ``w = 0`` the tilted density *is* the k-th component, whose normalizer
    is known exactly -- so ``log_norm`` must come out at 0 (the component's own
    log_prob is already normalized). The one place the grid quadrature can be
    scored against a closed form rather than against itself."""
    grid = TiltedGrid.build(ring(), 2, 0.0, half_width=8.0, resolution=512)
    assert grid.log_norm == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# the reference sampler
# --------------------------------------------------------------------------
def test_grid_sampler_reproduces_the_density_it_normalized():
    """Draws from :meth:`TiltedGrid.sample` must have the density's own mean and
    the right per-component share.

    The grid sampler is the *reference* in the experiment, so if it were biased
    every guided arm would be measured against the wrong thing. Checked two
    ways: the first moment against the grid's own mean, and the share of draws
    nearest each component against the grid mass in that component's cell.
    """
    mix = ring()
    w = 2.0
    grid = TiltedGrid.build(mix, 0, w, half_width=8.0, resolution=512)
    g = torch.Generator().manual_seed(0)
    x = grid.sample(200_000, generator=g)

    torch.testing.assert_close(
        x.to(torch.float64).mean(dim=0), grid.mean(), rtol=0, atol=0.02
    )
    # 6 equally weighted components, tilted toward k = 0: purity is high but is
    # asserted against the grid, not against a guessed level.
    gx, gy = torch.meshgrid(grid.xs, grid.ys, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).to(torch.float32)
    p = torch.softmax(grid.log_density.reshape(-1), dim=0)
    d2 = torch.cdist(pts.to(torch.float64), mix.means.to(torch.float64)) ** 2
    grid_purity = float(p[d2.argmin(dim=1) == 0].sum())
    assert component_purity(mix, 0, x) == pytest.approx(grid_purity, abs=5e-3)


def test_grid_sampler_is_reproducible_under_a_seed():
    grid = TiltedGrid.build(ring(), 0, 1.0, half_width=8.0, resolution=128)
    a = grid.sample(1000, generator=torch.Generator().manual_seed(7))
    b = grid.sample(1000, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


# --------------------------------------------------------------------------
# what guidance is supposed to do
# --------------------------------------------------------------------------
def test_tilting_concentrates_on_the_conditioned_component():
    """Raising ``w`` must move mass toward component k -- monotonically, and to
    a share the grid computes rather than one asserted by hand."""
    mix = ring()
    g = torch.Generator().manual_seed(1)
    purities = []
    for w in (0.0, 1.0, 4.0):
        grid = TiltedGrid.build(mix, 0, w, half_width=8.0, resolution=384)
        purities.append(component_purity(mix, 0, grid.sample(40_000, generator=g)))
    assert purities[0] > 0.99                       # w = 0 is the component itself
    assert all(a <= b + 1e-3 for a, b in zip(purities, purities[1:])), purities


def test_component_purity_is_a_nearest_mean_count():
    mix = ring()
    x = torch.cat([mix.means[0:1], mix.means[0:1], mix.means[3:4]], dim=0)
    assert component_purity(mix, 0, x) == pytest.approx(2 / 3)
    assert component_purity(mix, 3, x) == pytest.approx(1 / 3)


def test_tilted_density_at_weight_zero_is_the_component_log_prob():
    mix = ring()
    x = probe_points()
    torch.testing.assert_close(
        tilted_log_density(mix, 1, x, 0.0), mix.components[1].log_prob(x)
    )


def test_tilted_density_uses_the_noised_marginal_at_positive_sigma():
    """``r_{w,sigma}`` is built from the *sigma-noised* pair, which is what makes
    the family a function of sigma at all -- and what the experiment shows is
    not the noising path of ``q_w``."""
    mix = ring()
    x = probe_points()
    at0 = tilted_log_density(mix, 1, x, 2.0, 0.0)
    at1 = tilted_log_density(mix, 1, x, 2.0, 1.0)
    assert not torch.allclose(at0, at1)
    noised = mix.noised(1.0)
    torch.testing.assert_close(
        at1, 3.0 * noised.components[1].log_prob(x) - 2.0 * noised.log_prob(x),
        rtol=1e-5, atol=1e-5,
    )


def test_tilting_is_not_the_noising_path_of_the_clean_tilted_density():
    """The claim the experiment measures, stated as a test.

    ``r_{w,sigma}`` (tilt the noised pair) and ``q_w * N(0, sigma^2)`` (noise
    the tilted density) are different objects for ``w != 0``. Here the second is
    formed by convolving the grid-normalized ``q_w`` with the Gaussian kernel
    numerically, and the two are shown to disagree far beyond the quadrature --
    while agreeing at ``w = 0``, where tilting is the identity and the two
    operations trivially commute.
    """
    mix = ring()
    sigma = 0.6
    for w, must_differ in ((0.0, False), (3.0, True)):
        grid = TiltedGrid.build(mix, 0, w, half_width=8.0, resolution=256)
        gx, gy = torch.meshgrid(grid.xs, grid.ys, indexing="ij")
        p = torch.softmax(grid.log_density.reshape(-1), dim=0).reshape(gx.shape)

        # convolve q_w with N(0, sigma^2 I) at three probe points
        probes = torch.tensor([[4.0, 0.0], [0.0, 0.0], [2.0, 2.0]], dtype=torch.float64)
        dx = gx.reshape(1, -1) - probes[:, 0:1]
        dy = gy.reshape(1, -1) - probes[:, 1:2]
        kern = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2)) / (
            2 * math.pi * sigma ** 2
        )
        noised_tilt = (kern * p.reshape(1, -1)).sum(dim=1)

        tilted_noise = torch.exp(
            tilted_log_density(mix, 0, probes.to(torch.float32), w, sigma).to(
                torch.float64
            )
            - TiltedGrid.build(mix, 0, w, half_width=8.0, resolution=256).log_norm
        )
        # r_{w,sigma} is normalized by its own sigma-dependent constant, so
        # compare *ratios* across probe points: a shared constant cancels and
        # what is left is whether the two are the same shape.
        ratio = (noised_tilt / tilted_noise)
        spread = float(ratio.max() / ratio.min())
        if must_differ:
            assert spread > 1.5, spread
        else:
            assert spread < 1.02, spread


# --------------------------------------------------------------------------
# the mechanism behind the read-out
# --------------------------------------------------------------------------
def test_radial_offset_signs_and_scale():
    """Positive means further from the origin than ``mu_k``, and a cloud
    centred on ``mu_k`` reads zero."""
    mix = ring()
    mu = mix.means[0]
    u = mu / mu.norm()
    assert radial_offset(mix, 0, mu.unsqueeze(0)) == pytest.approx(0.0, abs=1e-6)
    assert radial_offset(mix, 0, (mu + 0.5 * u).unsqueeze(0)) == pytest.approx(0.5, abs=1e-5)
    assert radial_offset(mix, 0, (mu - 0.5 * u).unsqueeze(0)) == pytest.approx(-0.5, abs=1e-5)


def test_guided_stationary_point_is_a_root_of_the_isotropic_guided_drift():
    """The closed form must actually be where the affine drift vanishes.

    Checked against the drift the formula's own approximation defines -- one
    isotropic Gaussian at the origin for the marginal -- rather than against
    the true mixture score, because it is that approximation being asserted.
    The next test measures how good the approximation is on the real score.
    """
    mix = ring()
    for w in (0.5, 2.0, 8.0):
        for sigma in (0.05, 1.0, 8.0):
            x = guided_stationary_point(mix, 0, w, sigma)
            v = float(mix.covs[0, 0, 0]) + sigma ** 2
            u = float(torch.diagonal(mix.cov.double()).mean()) + sigma ** 2
            drift = -x * ((1 + w) / v - w / u) + (1 + w) * mix.means[0].double() / v
            torch.testing.assert_close(drift, torch.zeros(2, dtype=torch.float64),
                                       rtol=0, atol=1e-9)


def test_guided_stationary_point_hits_its_two_limits():
    """``(1+w) mu_k`` at large sigma, ``mu_k`` at small -- the two facts the
    whole radial-overshoot reading rests on."""
    mix = ring()
    mu = mix.means[0].double()
    w = 3.0
    far = guided_stationary_point(mix, 0, w, 400.0)
    near = guided_stationary_point(mix, 0, w, 0.0)
    torch.testing.assert_close(far, (1 + w) * mu, rtol=2e-2, atol=0)
    torch.testing.assert_close(near, mu, rtol=8e-2, atol=0)
    assert float(near.norm()) > float(mu.norm())     # still outside, just barely


def test_guided_stationary_point_tracks_the_true_score_root_at_large_sigma():
    """The isotropic approximation is not exact; at large sigma it should be
    close on the real mixture score, which is where the overshoot is set."""
    mix = ring()
    w, sigma = 3.0, 8.0
    x = guided_stationary_point(mix, 0, w, sigma).to(torch.float32).unsqueeze(0)
    s = guided_score(mix, 0, x, sigma, w).double()
    scale = float(guided_score(mix, 0, torch.zeros(1, 2), sigma, w).double().norm())
    assert float(s.norm()) < 0.1 * scale, (float(s.norm()), scale)
