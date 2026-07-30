"""Tests for the probability-flow ODE, its integrator, and its likelihoods.

Three things are pinned here, and none of them is "the samples look right".

1. **The solver.** On a Gaussian target the PF-ODE has a closed-form solution
   (a radial scaling, :func:`diffusion.ode.gaussian_pf_ode_scale`), so RK4 is
   checked *pointwise* against the exact trajectory and its measured order of
   convergence is asserted to be 4. A sampler test in distribution would pass
   with a first-order integrator and half the arithmetic.
2. **The divergence estimator.** Unbiasedness on a field whose trace is known,
   plus the two variance facts that justify the default probe distribution.
3. **The likelihood.** The noised mixture has an exact density, so the whole
   change-of-variables pipeline is compared against truth rather than against
   itself, and the integration error is shown to fall with the step count while
   the Monte Carlo error falls as ``1/sqrt(k)`` -- two different errors, told
   apart on purpose.
"""

from __future__ import annotations

import math

import pytest
import torch

from diffusion.langevin import geometric_sigmas
from diffusion.ode import (
    STEP_EVALUATIONS,
    LogProbResult,
    euler_step,
    exact_divergence,
    gaussian_pf_ode_scale,
    gaussian_probes,
    hutchinson_divergence,
    integrate,
    pf_ode_field,
    pf_ode_log_prob,
    pf_ode_nfe,
    pf_ode_sample,
    rademacher_probes,
    rk4_step,
)
from diffusion.samplers import ddpm_ancestral
from diffusion.targets import GaussianMixture

DT = torch.float64


def unit_gaussian() -> GaussianMixture:
    """``N(0, I)`` as a one-component mixture, so ``.noised()`` is available."""
    return GaussianMixture(torch.zeros(1, 2), torch.eye(2).unsqueeze(0), dtype=DT)


def exact_score(target):
    def s(x: torch.Tensor, sigma: float) -> torch.Tensor:
        return target.noised(sigma).score(x)

    return s


def slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of ``log y`` against ``log x`` -- the measured order."""
    lx = [math.log(v) for v in xs]
    ly = [math.log(v) for v in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


# --------------------------------------------------------------------------
# the integrator, against closed forms
# --------------------------------------------------------------------------
def test_rk4_hits_the_exact_flow_of_a_nonautonomous_ode():
    """``dx/ds = cos(s) x`` has solution ``x0 exp(sin s - sin s0)``.

    A plain ODE with no diffusion content, chosen because it is
    ``s``-dependent: a stepper that ignored the clock offsets of ``k2..k4``
    would still pass an autonomous test.
    """

    def field(x, s):
        return math.cos(s) * x

    x0 = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=DT)
    nodes = [0.5 + 0.25 * i for i in range(13)]  # 0.5 -> 3.5, h = 0.25
    x = integrate(field, x0, nodes)
    want = x0 * math.exp(math.sin(nodes[-1]) - math.sin(nodes[0]))
    # 2e-5 at this step size, which is what a fourth-order method should give
    # on an O(1) field over an interval of 3; the sharp statement about *why*
    # that is the number is the order test below.
    assert torch.allclose(x, want, rtol=1e-4)
    assert not torch.allclose(x, want, rtol=1e-7)


def test_rk4_is_fourth_order_and_euler_is_first():
    """Measured global order on the same known ODE, over a 16x range of ``h``.

    The two methods are measured over different step ranges on purpose: at
    ``h = 0.75`` neither is in its asymptotic regime, and Euler's error there
    happens to be *small* by cancellation (the field changes sign over the
    interval), so including it would report an order of 0.6 for reasons that
    have nothing to do with the method.
    """

    def field(x, s):
        return math.cos(s) * x

    x0 = torch.tensor([[1.0, -2.0]], dtype=DT)
    errs = {}
    for method, steps in (("rk4", [8, 16, 32, 64, 128]), ("euler", [16, 32, 64, 128, 256])):
        e = []
        for n in steps:
            nodes = [0.5 + 3.0 * i / n for i in range(n + 1)]
            x = integrate(field, x0, nodes, method=method)
            want = x0 * math.exp(math.sin(nodes[-1]) - math.sin(nodes[0]))
            e.append(float((x - want).abs().max()))
        errs[method] = (steps, e)
    assert slope([3.0 / n for n in errs["rk4"][0]], errs["rk4"][1]) == pytest.approx(4.0, abs=0.15)
    assert slope([3.0 / n for n in errs["euler"][0]], errs["euler"][1]) == pytest.approx(1.0, abs=0.15)
    # ...and the order is not an artifact of a tiny error either way.
    assert errs["euler"][1][0] > errs["rk4"][1][0] * 1e2


def test_the_pf_ode_on_a_gaussian_is_the_closed_form_radial_scaling():
    """Pointwise, not in distribution: the exact solution is known here.

    ``dx/dsigma = sigma (x - mu)/(v + sigma^2)`` separates, so every point
    moves along its own ray by the factor
    ``sqrt((v + sigma_1^2)/(v + sigma_0^2))``.
    """
    p = unit_gaussian()
    ladder = geometric_sigmas(8.0, 0.05, 40)
    x0 = 8.0 * torch.randn(64, 2, dtype=DT, generator=torch.Generator().manual_seed(0))
    x = pf_ode_sample(exact_score(p), x0, ladder, final_step=False)
    c = gaussian_pf_ode_scale(1.0, ladder[0], ladder[-1])
    assert c == pytest.approx(math.sqrt((1 + 0.05**2) / (1 + 64.0)), rel=1e-12)
    assert torch.allclose(x, c * x0, atol=1e-5)
    # The map is a contraction by ~8 here, so "allclose" is not vacuous.
    assert c < 0.13


def test_refining_the_ladder_gives_fourth_order_on_the_pf_ode_itself():
    """The order claim holds on the object the repo actually integrates."""
    p = unit_gaussian()
    x0 = 8.0 * torch.randn(16, 2, dtype=DT, generator=torch.Generator().manual_seed(1))
    counts = [10, 20, 40, 80]
    errs = []
    for n in counts:
        ladder = geometric_sigmas(8.0, 0.05, n)
        x = pf_ode_sample(exact_score(p), x0, ladder, final_step=False)
        c = gaussian_pf_ode_scale(1.0, ladder[0], ladder[-1])
        errs.append(float((x - c * x0).abs().max()))
    hs = [1.0 / n for n in counts]
    assert slope(hs, errs) > 3.5
    assert errs[-1] < 1e-5


def test_euler_needs_far_more_nodes_than_rk4_for_the_same_accuracy():
    """Where the four evaluations per step pay for themselves.

    Not a style preference: on this field RK4 at 20 nodes (76 evaluations)
    lands at 8.8e-5 while Euler at 320 nodes (319 evaluations, four times the
    budget) is still at 8.2e-3 -- ~90x worse for ~4x the cost.
    """
    p = unit_gaussian()
    x0 = 8.0 * torch.randn(16, 2, dtype=DT, generator=torch.Generator().manual_seed(2))

    def err(n, method):
        ladder = geometric_sigmas(8.0, 0.05, n)
        x = pf_ode_sample(exact_score(p), x0, ladder, method=method, final_step=False)
        return float((x - gaussian_pf_ode_scale(1.0, ladder[0], ladder[-1]) * x0).abs().max())

    assert err(20, "rk4") < err(320, "euler") / 50
    assert pf_ode_nfe(20, "rk4", final_step=False) < pf_ode_nfe(320, "euler", final_step=False)


# --------------------------------------------------------------------------
# what the ODE is, relative to the stochastic sampler
# --------------------------------------------------------------------------
def test_the_ode_drift_is_half_the_ancestral_drift():
    """The module docstring's claim, measured against the ancestral sampler.

    One ancestral step is a mean update plus zero-mean noise, so averaging the
    displacement over many chains started at the same point recovers the mean
    update; the ODE's Euler step over the same level pair is smaller by
    ``sigma / (sigma + sigma_next)``, which tends to 1/2 as the ladder refines.
    """
    p = unit_gaussian()
    sigma, sigma_next = 1.0, 0.9
    x_pt = torch.tensor([[1.3, -0.7]], dtype=DT)
    n = 200_000
    x0 = x_pt.expand(n, 2).contiguous()
    x = ddpm_ancestral(
        exact_score(p), x0, [sigma, sigma_next],
        generator=torch.Generator().manual_seed(0), denoise_final=False,
    )
    sde_mean_step = (x - x0).mean(dim=0)

    v = pf_ode_field(exact_score(p))
    ode_step = (euler_step(v, x_pt, sigma, sigma_next - sigma) - x_pt).squeeze(0)

    ratio = float(ode_step.norm() / sde_mean_step.norm())
    # noise s.e. on the mean displacement is ~0.001 against a step of ~0.1
    assert ratio == pytest.approx(sigma / (sigma + sigma_next), rel=0.02)
    # and the factor is 1/2 in the fine-ladder limit, by arithmetic
    assert sigma / (sigma + 0.999999) == pytest.approx(0.5, abs=1e-6)


def test_the_final_stub_is_exactly_half_the_tweedie_denoise():
    """Both are ``O(sigma_min^2)`` and they differ by exactly 2 -- documented,
    not glossed: Tweedie jumps to ``E[x_0|x]``, the flow travels its last arc."""
    p = unit_gaussian()
    ladder = geometric_sigmas(8.0, 0.05, 20)
    x0 = 8.0 * torch.randn(256, 2, dtype=DT, generator=torch.Generator().manual_seed(3))
    bare = pf_ode_sample(exact_score(p), x0, ladder, final_step=False)
    stubbed = pf_ode_sample(exact_score(p), x0, ladder, final_step=True)
    sigma_min = ladder[-1]
    tweedie = bare + sigma_min**2 * exact_score(p)(bare, sigma_min)
    assert torch.allclose(stubbed - bare, 0.5 * (tweedie - bare), atol=1e-14)
    # Both choices are far below the metric floor the sampler comparison lives
    # at (sliced W2 ~ 0.07), so the comparison does not hinge on this choice.
    assert float((tweedie - stubbed).norm(dim=1).max()) < 0.01


def test_the_ode_and_the_sde_sampler_agree_on_the_ring():
    """The central claim: same marginals, different paths.

    With the exact score and enough resolution, the deterministic flow and the
    stochastic chain must land on the same distribution -- checked on mode
    occupancy and on the covariance, both against the target's own values.
    """
    ring = GaussianMixture.ring(dtype=DT)
    g = torch.Generator().manual_seed(0)
    x0 = 8.0 * torch.randn(4_000, 2, dtype=DT, generator=g)
    ode = pf_ode_sample(exact_score(ring), x0, geometric_sigmas(8.0, 0.05, 50))
    sde = ddpm_ancestral(exact_score(ring), x0, geometric_sigmas(8.0, 0.05, 200), generator=g)

    def occupancy(x):
        label = torch.round(torch.atan2(x[:, 1], x[:, 0]) / (2 * math.pi / 8)).long() % 8
        return torch.bincount(label, minlength=8).double() / x.shape[0]

    assert float(occupancy(ode).min()) > 0.10 and float(occupancy(ode).max()) < 0.15
    assert torch.allclose(occupancy(ode), occupancy(sde), atol=0.02)
    assert torch.allclose(ode.T.cov().diagonal(), ring.cov.diagonal(), rtol=0.05)


def test_the_ode_sampler_is_deterministic():
    """No generator, and none needed: the only randomness is the prior draw."""
    p = unit_gaussian()
    ladder = geometric_sigmas(8.0, 0.05, 20)
    x0 = 8.0 * torch.randn(32, 2, dtype=DT, generator=torch.Generator().manual_seed(4))
    a = pf_ode_sample(exact_score(p), x0, ladder)
    b = pf_ode_sample(exact_score(p), x0, ladder)
    assert torch.equal(a, b)


def test_trajectory_is_one_state_per_node_plus_the_stub():
    p = unit_gaussian()
    ladder = geometric_sigmas(4.0, 0.1, 10)
    x0 = 4.0 * torch.randn(8, 2, dtype=DT, generator=torch.Generator().manual_seed(5))
    x, traj = pf_ode_sample(exact_score(p), x0, ladder, trajectory=True)
    assert traj.shape == (len(ladder) + 1, 8, 2)
    assert torch.equal(traj[0], x0)
    assert torch.equal(traj[-1], x)


def test_score_evaluations_match_the_nfe_accounting():
    """The x-axis of the SDE-vs-ODE figure has to be the truth."""
    p = unit_gaussian()
    ladder = geometric_sigmas(4.0, 0.1, 11)
    for method in ("euler", "rk4"):
        calls: list[float] = []

        def counted(x, sigma):
            calls.append(sigma)
            return exact_score(p)(x, sigma)

        pf_ode_sample(counted, torch.zeros(4, 2, dtype=DT), ladder, method=method)
        assert len(calls) == pf_ode_nfe(len(ladder), method, final_step=True)
        assert len(calls) == 10 * STEP_EVALUATIONS[method] + 1


# --------------------------------------------------------------------------
# divergence
# --------------------------------------------------------------------------
LINEAR_A = torch.tensor([[0.3, -1.2], [0.7, 2.0]], dtype=DT)


def linear_field(x, s):
    """``v(x) = A x``, so ``grad v = A`` everywhere and ``div v = tr(A)``."""
    return x @ LINEAR_A.T


def test_exact_divergence_is_the_trace_of_a_known_jacobian():
    x = torch.randn(6, 2, dtype=DT, generator=torch.Generator().manual_seed(0))
    got = exact_divergence(linear_field, x, 0.0)
    assert torch.allclose(got, torch.full((6,), float(LINEAR_A.trace()), dtype=DT), atol=1e-12)


def test_hutchinson_is_unbiased_on_a_field_whose_trace_is_known():
    """``E[eps' A eps] = tr(A)``: the mean lands within its own standard error."""
    x = torch.randn(4, 2, dtype=DT, generator=torch.Generator().manual_seed(0))
    probes = rademacher_probes(
        4_000, (4, 2), dtype=DT, generator=torch.Generator().manual_seed(1)
    )
    est = hutchinson_divergence(linear_field, x, 0.0, probes)
    want = float(LINEAR_A.trace())
    mean = est.mean(dim=0)
    stderr = est.std(dim=0) / math.sqrt(probes.shape[0])
    assert float(stderr.max()) > 0  # the estimator is genuinely noisy here
    assert torch.all((mean - want).abs() < 3.0 * stderr)


def test_rademacher_is_exact_on_a_diagonal_jacobian_where_gaussian_is_not():
    """The reason Rademacher is the default, as a measurement."""
    diag = torch.diag(torch.tensor([1.7, -0.4], dtype=DT))

    def field(x, s):
        return x @ diag.T

    x = torch.randn(5, 2, dtype=DT, generator=torch.Generator().manual_seed(0))
    want = float(diag.trace())
    rad = hutchinson_divergence(
        field, x, 0.0,
        rademacher_probes(64, (5, 2), dtype=DT, generator=torch.Generator().manual_seed(2)),
    )
    assert torch.allclose(rad, torch.full_like(rad, want), atol=1e-12)  # zero variance
    gau = hutchinson_divergence(
        field, x, 0.0,
        gaussian_probes(64, (5, 2), dtype=DT, generator=torch.Generator().manual_seed(2)),
    )
    assert float(gau.std()) > 0.5


def test_rademacher_has_the_lower_variance_on_a_full_jacobian_too():
    x = torch.randn(3, 2, dtype=DT, generator=torch.Generator().manual_seed(0))
    kw = dict(dtype=DT, generator=torch.Generator().manual_seed(7))
    rad = hutchinson_divergence(linear_field, x, 0.0, rademacher_probes(4_000, (3, 2), **kw))
    kw = dict(dtype=DT, generator=torch.Generator().manual_seed(7))
    gau = hutchinson_divergence(linear_field, x, 0.0, gaussian_probes(4_000, (3, 2), **kw))
    assert float(rad.std(dim=0).mean()) < float(gau.std(dim=0).mean())


def test_a_no_grad_score_is_refused_rather_than_returning_zero():
    p = unit_gaussian()

    def blocked(x, sigma):
        with torch.no_grad():
            return exact_score(p)(x, sigma)

    with pytest.raises(ValueError, match="not differentiable in x"):
        exact_divergence(pf_ode_field(blocked), torch.zeros(4, 2, dtype=DT), 1.0)


# --------------------------------------------------------------------------
# likelihood
# --------------------------------------------------------------------------
def test_log_det_on_a_gaussian_is_the_closed_form_integral():
    """``div v = d sigma/(v + sigma^2)`` is independent of ``x`` here, so the
    accumulated log-Jacobian is a constant with an elementary value -- which
    pins the sign convention as well as the magnitude."""
    p = unit_gaussian()
    ladder = geometric_sigmas(8.0, 0.05, 40)
    x = torch.randn(32, 2, dtype=DT, generator=torch.Generator().manual_seed(0))
    res = pf_ode_log_prob(
        exact_score(p), x, ladder, lambda z: p.noised(ladder[0]).log_prob(z), exact=True
    )
    want = 0.5 * 2 * math.log((1 + ladder[0] ** 2) / (1 + ladder[-1] ** 2))
    assert torch.allclose(res.log_det.squeeze(0), torch.full((32,), want, dtype=DT), atol=1e-6)
    assert want > 4.0  # the term is large, not a rounding-sized correction


def test_log_prob_recovers_the_exact_noised_mixture_density():
    """The validation the repo exists for: closed-form truth at the bottom."""
    ring = GaussianMixture.ring(dtype=DT)
    ladder = geometric_sigmas(8.0, 0.05, 40)
    pts = ring.noised(ladder[-1]).sample(128, generator=torch.Generator().manual_seed(3))
    res = pf_ode_log_prob(
        exact_score(ring), pts, ladder,
        lambda z: ring.noised(ladder[0]).log_prob(z), exact=True,
    )
    truth = ring.noised(ladder[-1]).log_prob(pts)
    assert float((res.log_prob - truth).abs().max()) < 1e-3
    # The transported answer spans ~5 nats across these points, so agreeing to
    # 1e-3 is not agreement between two nearly constant numbers.
    assert float(truth.max() - truth.min()) > 4.0


def test_likelihood_integration_error_falls_with_the_step_count():
    """Integration error and estimator variance are different errors; this is
    the first one, isolated by using the exact divergence."""
    ring = GaussianMixture.ring(dtype=DT)
    pts = ring.noised(0.05).sample(32, generator=torch.Generator().manual_seed(3))
    truth = ring.noised(0.05).log_prob(pts)
    errs = []
    for n in (5, 10, 20, 40):
        ladder = geometric_sigmas(8.0, 0.05, n)
        res = pf_ode_log_prob(
            exact_score(ring), pts, ladder,
            lambda z: ring.noised(ladder[0]).log_prob(z), exact=True,
        )
        errs.append(float((res.log_prob - truth).abs().mean()))
    assert all(b < a for a, b in zip(errs, errs[1:]))
    assert errs[0] > 20 * errs[-1]


def test_hutchinson_likelihood_is_unbiased_and_its_spread_falls_as_one_over_sqrt_k():
    """The estimator is unbiased for ``log p`` -- so its *mean* is comparable to
    the exact-divergence answer, while a single probe is not."""
    ring = GaussianMixture.ring(dtype=DT)
    ladder = geometric_sigmas(8.0, 0.05, 20)
    pts = ring.noised(ladder[-1]).sample(64, generator=torch.Generator().manual_seed(3))
    reference = pf_ode_log_prob(
        exact_score(ring), pts, ladder,
        lambda z: ring.noised(ladder[0]).log_prob(z), exact=True,
    ).log_prob

    spreads = []
    for k in (1, 4, 16):
        res = pf_ode_log_prob(
            exact_score(ring), pts, ladder,
            lambda z: ring.noised(ladder[0]).log_prob(z),
            n_probes=k, generator=torch.Generator().manual_seed(11),
        )
        assert res.per_probe.shape == (k, 64)
        spreads.append(float((res.log_prob - reference).abs().mean()))
        if k > 1:
            # The reported standard error is a real error bar, checked in
            # aggregate rather than per point: at points where one mixture
            # component dominates, the noised score's Jacobian is exactly
            # isotropic and Rademacher probes return the trace with *zero*
            # variance, so a per-point ratio would divide by an honest zero.
            assert float((res.log_prob - reference).abs().mean()) < 2.0 * float(res.stderr.mean())
    assert spreads[2] < spreads[0] / 2.0  # 16 probes vs 1: ~4x, so 2x is safe


def test_one_probe_reports_no_standard_error_and_sixteen_do():
    ring = GaussianMixture.ring(dtype=DT)
    ladder = geometric_sigmas(8.0, 0.05, 10)
    pts = ring.noised(ladder[-1]).sample(16, generator=torch.Generator().manual_seed(3))
    top = lambda z: ring.noised(ladder[0]).log_prob(z)  # noqa: E731
    one = pf_ode_log_prob(exact_score(ring), pts, ladder, top, n_probes=1)
    many = pf_ode_log_prob(
        exact_score(ring), pts, ladder, top, n_probes=16,
        generator=torch.Generator().manual_seed(0),
    )
    assert isinstance(one, LogProbResult)
    assert torch.all(one.stderr == 0)
    assert float(many.stderr.min()) > 0


def test_the_flow_density_normalizes_to_one():
    """A transported density is only a density if it integrates to 1 -- checked
    by quadrature on a grid, which also catches a sign error in the log-det
    that comparing against the truth on typical points might not."""
    p = GaussianMixture(torch.zeros(1, 2), 0.6 * torch.eye(2).unsqueeze(0), dtype=DT)
    ladder = geometric_sigmas(6.0, 0.1, 20)
    lim, n = 6.0, 60
    q = torch.linspace(-lim, lim, n, dtype=DT)
    gx, gy = torch.meshgrid(q, q, indexing="xy")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    res = pf_ode_log_prob(
        exact_score(p), grid, ladder, lambda z: p.noised(ladder[0]).log_prob(z), exact=True
    )
    cell = (2 * lim / (n - 1)) ** 2
    assert float(res.log_prob.exp().sum() * cell) == pytest.approx(1.0, abs=5e-3)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "nodes, message",
    [
        ([1.0], "at least two nodes"),
        ([1.0, 0.5, 0.8], "monotone"),
        ([1.0, 1.0, 0.5], "strictly monotone"),
        ([1.0, 0.0], "must be positive"),
    ],
)
def test_a_malformed_node_list_is_refused(nodes, message):
    p = unit_gaussian()
    with pytest.raises(ValueError, match=message):
        integrate(pf_ode_field(exact_score(p)), torch.zeros(4, 2, dtype=DT), nodes)


def test_an_unknown_method_is_refused():
    p = unit_gaussian()
    with pytest.raises(ValueError, match="unknown method"):
        integrate(
            pf_ode_field(exact_score(p)), torch.zeros(4, 2, dtype=DT), [1.0, 0.5], method="rk2"
        )
    with pytest.raises(ValueError, match="unknown method"):
        pf_ode_nfe(10, "midpoint")


def test_a_likelihood_ladder_given_upside_down_is_refused():
    """One ladder convention across the repo: coarse -> fine, always."""
    p = unit_gaussian()
    with pytest.raises(ValueError, match="coarse -> fine"):
        pf_ode_log_prob(
            exact_score(p), torch.zeros(4, 2, dtype=DT), [0.1, 1.0],
            lambda z: p.noised(1.0).log_prob(z), exact=True,
        )


def test_a_non_finite_field_is_reported_as_the_field():
    def broken(x, s):
        return torch.full_like(x, float("nan"))

    with pytest.raises(FloatingPointError, match="field itself returned"):
        integrate(broken, torch.zeros(4, 2, dtype=DT), geometric_sigmas(4.0, 0.1, 5))


def test_probe_shapes_and_counts_are_checked():
    with pytest.raises(ValueError, match="n_probes must be"):
        rademacher_probes(0, (4, 2))
    with pytest.raises(ValueError, match="n_probes must be"):
        gaussian_probes(-1, (4, 2))
    with pytest.raises(ValueError, match="probes must be"):
        hutchinson_divergence(
            linear_field, torch.zeros(4, 2, dtype=DT), 0.0, torch.zeros(3, 5, 2, dtype=DT)
        )


def test_an_unknown_probe_distribution_is_refused():
    p = unit_gaussian()
    with pytest.raises(ValueError, match="unknown probe_distribution"):
        pf_ode_log_prob(
            exact_score(p), torch.zeros(4, 2, dtype=DT), [1.0, 0.5],
            lambda z: p.noised(1.0).log_prob(z), probe_distribution="cauchy",
        )


def test_the_gaussian_scale_refuses_a_non_positive_variance():
    with pytest.raises(ValueError, match="target_var must be positive"):
        gaussian_pf_ode_scale(0.0, 1.0, 0.5)


def test_a_step_is_a_pure_function_of_its_arguments():
    """``rk4_step``/``euler_step`` do not mutate the state they are handed."""
    p = unit_gaussian()
    v = pf_ode_field(exact_score(p))
    x = torch.randn(8, 2, dtype=DT, generator=torch.Generator().manual_seed(0))
    keep = x.clone()
    rk4_step(v, x, 1.0, -0.1)
    euler_step(v, x, 1.0, -0.1)
    assert torch.equal(x, keep)
