r"""The probability-flow ODE: the same marginals, without the noise.

:mod:`diffusion.samplers` walks the noising process backwards *stochastically*.
This module does it deterministically. Every noise level of the forward process
has a marginal ``p_sigma``, and there is an ordinary differential equation whose
solutions carry ``p_{sigma_max}`` onto ``p_{sigma_min}`` through exactly those
marginals -- no injected noise anywhere. Sampling becomes solving an initial
value problem, and because the map is a diffeomorphism, **densities transport
with it**, which is where the likelihood at the bottom of this module comes
from. This is the repo's neural-ODE moment: a learned vector field, integrated
by a Runge-Kutta method written out in full, with the change-of-variables
formula giving log-likelihoods that can be checked against a target whose
density is known.

Why a deterministic flow has the same marginals
-----------------------------------------------
The forward process here is variance-exploding with no drift,
``dx = g(t) dw`` and ``g(t)^2 = d(sigma^2)/dt``, so its density obeys the
Fokker-Planck (heat) equation

    ``d_t p = (1/2) g^2 Laplacian p``.

Now use ``Laplacian p = div(grad p) = div(p grad log p)`` and read the result
as a *continuity* equation:

    ``d_t p = -div(p v)``,   ``v(x, t) := -(1/2) g(t)^2 grad log p_t(x)``.

A continuity equation is the statement that ``p`` is transported by the
velocity field ``v``. So the deterministic ODE ``dx/dt = v(x, t)`` has the same
time marginals as the SDE (Song et al. 2021, Eq. 13, with ``f = 0``). Nothing
was approximated; the two processes differ in their *paths*, not their
marginals.

The schedule cancels: use sigma as the clock
--------------------------------------------
``sigma(t)`` is strictly increasing, so it can be the independent variable
instead of ``t``, and then the schedule disappears from the equation entirely:

    ``dx/dsigma = (dx/dt) / (dsigma/dt) = -(1/2) (2 sigma dsigma/dt) s / (dsigma/dt)``

        **``dx/dsigma = -sigma * s(x, sigma)``**.

Every VE noise schedule is a reparameterization of this one trajectory, so the
only thing a schedule can change is *where the integration nodes are placed* --
which is a property of the solver, not of the model (Karras et al. 2022 make
the same point). All functions here take a ladder of ``sigma`` values and treat
it as a set of nodes.

Two consequences worth stating, because they are checkable:

- On a Gaussian target ``N(mu, v I)`` the exact score is
  ``-(x - mu)/(v + sigma^2)``, so the ODE is ``dx/dsigma = sigma (x - mu)/(v + sigma^2)``
  and separates: the flow is a pure radial scaling,

      ``x(sigma_1) = mu + (x(sigma_0) - mu) sqrt( (v + sigma_1^2) / (v + sigma_0^2) )``.

  :func:`gaussian_pf_ode_scale` returns that factor, and the tests use it the
  way ``ddpm_gaussian_variance`` is used for the ancestral sampler: to assert
  what the solver *does* against arithmetic rather than against a picture.
- The solver evaluates the score at midpoints *between* ladder levels. A
  per-level model could not be integrated at all; conditioning one network on
  ``sigma`` (:mod:`diffusion.model`) is what makes the field continuous in the
  clock, and this is where that pays off.

How the ODE relates to the ancestral sampler
--------------------------------------------
Take one Euler step of the ODE from ``sigma_i`` down to ``sigma_{i-1}``:

    ``dx = -sigma_i s (sigma_{i-1} - sigma_i) = sigma_i (sigma_i - sigma_{i-1}) s``,

against the ancestral sampler's mean update
``(sigma_i^2 - sigma_{i-1}^2) s = (sigma_i - sigma_{i-1})(sigma_i + sigma_{i-1}) s``,
which for adjacent levels is **twice** as large. The extra half-drift plus the
injected noise of the stochastic sampler form a Langevin step at the current
noise level, and Langevin leaves ``p_sigma`` invariant -- so the SDE sampler is
the ODE sampler plus a term that does nothing to the marginal in exact
arithmetic. In inexact arithmetic it does two opposite things: it re-randomizes
(errors made at coarse levels can be forgotten, which is why the stochastic
sampler is often more forgiving of a bad score) and it adds variance. Day 25
measured the flip side of the same coin: with a *biased* learned score, running
the equilibrium sampler longer converged harder to the wrong distribution. The
ODE has no equilibrium to converge to, only an integration error that falls
with the step count.

The last stub, and why it is half the ancestral denoise
-------------------------------------------------------
Integrating to ``sigma = 0`` exactly is not available: the learned score is
conditioned on ``log sigma``. The residual segment is tiny and its Euler step is

    ``x <- x + (sigma_min^2 / 2) s(x, sigma_min)``,

which is exactly **half** of the ancestral sampler's Tweedie denoise
``x + sigma_min^2 s`` -- not a discrepancy but the difference between the two
objects: Tweedie jumps to ``E[x_0 | x]``, while the ODE trajectory only travels
the arc it has left. :func:`pf_ode_sample` takes the ODE-consistent stub, and
``tests/test_ode.py`` pins the factor of 2 and checks that at this repo's
``sigma_min`` both choices sit far below the metric floor, so the
sampler comparison is not an artifact of picking one.

Likelihoods by instantaneous change of variables
------------------------------------------------
For a flow ``dx/ds = v(x, s)``, differentiating along a trajectory:

    ``d/ds log p_s(x(s)) = d_s log p + v . grad log p``
                        ``= -div(p v)/p + v . grad log p = -div v(x(s), s)``,

using the continuity equation above (Chen et al. 2018, Eq. 2). Integrating
*up* the ladder from ``sigma_min`` to ``sigma_max`` and rearranging,

    ``log p_{sigma_min}(x) = log p_{sigma_max}(z) + int_{sigma_min}^{sigma_max} div v ds``,

with ``z`` the endpoint of the trajectory started at ``x``. Both terms are
computable: the endpoint density is the known prior at the top of the ladder,
and the divergence of ``v = -sigma s`` is ``-sigma tr(grad s)``.

The trace is the expensive part, and Grathwohl et al.'s (2019) estimator
replaces it with

    ``tr(A) = E_eps[ eps' A eps ]``   for any ``eps`` with ``E[eps eps'] = I``,

evaluated by one vector-Jacobian product per probe. Two details that matter:

1. **The probes are drawn once per solve, not per evaluation.** Resampling
   inside the integrator makes the vector field discontinuous in ``s`` and
   silently destroys the solver's order; with fixed probes the augmented system
   is a perfectly ordinary ODE that RK4 integrates at fourth order.
2. **The estimator is unbiased for ``log p``, hence biased for ``p``.**
   Reporting mean log-likelihood is fine; exponentiating a single estimate is
   not. Rademacher probes are used by default: for a diagonal Jacobian they
   give the trace *exactly* (every ``eps_i^2 = 1``), where Gaussian probes
   still carry variance from the diagonal.

Because the mixture target's noised marginals have closed-form densities, the
whole pipeline has a ground truth: :func:`pf_ode_log_prob` on the exact score
must reproduce ``log p_{sigma_min}`` to integration error, and run on the
*learned* score it gives a genuine normalized model density, so its gap to the
truth averaged over exact draws is a Monte Carlo estimate of
``KL(p_{sigma_min} || model)`` -- which must come out positive. ``experiments/ode.py``
reports both.

References
----------
- Song et al. (2021), *Score-Based Generative Modeling through SDEs* -- Eq. 13
  (probability flow ODE), App. D (likelihood computation).
- Chen et al. (2018), *Neural Ordinary Differential Equations* -- instantaneous
  change of variables.
- Grathwohl et al. (2019), *FFJORD* -- the Hutchinson trace estimator in this
  role; Hutchinson (1989) for the estimator itself.
- Karras et al. (2022), *Elucidating the Design Space of Diffusion-Based
  Generative Models* -- sigma as the clock, node placement as a solver choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from diffusion.langevin import NoisedScoreFn

Tensor = torch.Tensor

#: ``v(x, s) -> (n, d)``: a vector field on points, clocked by a scalar ``s``.
VectorField = Callable[[Tensor, float], Tensor]

#: Score evaluations each stepper spends per step -- the NFE bookkeeping the
#: SDE-vs-ODE comparison is built on.
STEP_EVALUATIONS = {"euler": 1, "rk4": 4}

__all__ = [
    "VectorField",
    "STEP_EVALUATIONS",
    "euler_step",
    "rk4_step",
    "integrate",
    "pf_ode_field",
    "pf_ode_sample",
    "pf_ode_nfe",
    "gaussian_pf_ode_scale",
    "rademacher_probes",
    "gaussian_probes",
    "exact_divergence",
    "hutchinson_divergence",
    "LogProbResult",
    "pf_ode_log_prob",
]


# --------------------------------------------------------------------------
# integrators
# --------------------------------------------------------------------------
def euler_step(field: VectorField, x: Tensor, s: float, h: float) -> Tensor:
    """One explicit Euler step: ``x + h v(x, s)``. Local error ``O(h^2)``."""
    return x + h * field(x, s)


def rk4_step(field: VectorField, x: Tensor, s: float, h: float) -> Tensor:
    """One classical fourth-order Runge-Kutta step, written out.

    ``k1 = v(x, s)``, ``k2 = v(x + h k1/2, s + h/2)``,
    ``k3 = v(x + h k2/2, s + h/2)``, ``k4 = v(x + h k3, s + h)``, and

        ``x <- x + (h/6) (k1 + 2 k2 + 2 k3 + k4)``.

    The weights are the ones that make the update agree with the Taylor
    expansion of the exact solution through ``h^4``: ``k1`` and ``k4`` are the
    endpoints of Simpson's rule and ``k2, k3`` the (doubled) midpoint, each
    midpoint slope taken along the previous estimate so that the ``h^2`` and
    ``h^3`` terms of the flow are picked up too. Local error is ``O(h^5)``,
    global error ``O(h^4)``, at four field evaluations per step -- so RK4 is
    worth it only when a fourth of the steps buys back more than fourth-order
    accuracy, which on this repo's smooth 2D fields it does by a wide margin
    (``tests/test_ode.py`` measures the order and the crossover).

    ``h`` may be negative; nothing here assumes a direction of time, which is
    what lets the same stepper sample down the ladder and compute likelihoods
    back up it.
    """
    k1 = field(x, s)
    k2 = field(x + 0.5 * h * k1, s + 0.5 * h)
    k3 = field(x + 0.5 * h * k2, s + 0.5 * h)
    k4 = field(x + h * k3, s + h)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


_STEPPERS: dict[str, Callable[[VectorField, Tensor, float, float], Tensor]] = {
    "euler": euler_step,
    "rk4": rk4_step,
}


def _check_finite(x: Tensor, where: str) -> None:
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(
            f"ODE state became non-finite {where}. Unlike the Langevin samplers "
            "there is no injected noise to blame and no stability bound to "
            "violate: the field itself returned something non-finite, which for "
            "a learned score means it was evaluated far outside its training "
            "support (an integrator overshoot), not that a knob is too large."
        )


def _check_nodes(nodes: Sequence[float]) -> list[float]:
    lv = [float(s) for s in nodes]
    if len(lv) < 2:
        raise ValueError("an integration needs at least two nodes")
    if any(s <= 0 for s in lv):
        raise ValueError("noise levels must be positive (the field is clocked by log sigma)")
    up = sorted(lv)
    down = sorted(lv, reverse=True)
    if lv != up and lv != down:
        raise ValueError("nodes must be monotone (decreasing to sample, increasing for likelihoods)")
    if len(set(lv)) != len(lv):
        raise ValueError("nodes must be strictly monotone (a repeated node is a zero-length step)")
    return lv


def integrate(
    field: VectorField,
    x0: Tensor,
    nodes: Sequence[float],
    method: str = "rk4",
    trajectory: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Integrate ``dx/ds = field(x, s)`` through ``nodes``, in the order given.

    ``nodes`` are the clock values to step between -- strictly monotone, either
    direction, and not necessarily evenly spaced (the geometric ladders of
    :func:`diffusion.langevin.geometric_sigmas` are not). Returns the final
    state, or ``(final, states)`` with one state per node when ``trajectory``.

    Cost is ``(len(nodes) - 1) * STEP_EVALUATIONS[method]`` field evaluations.
    """
    if method not in _STEPPERS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_STEPPERS)}")
    lv = _check_nodes(nodes)
    step = _STEPPERS[method]
    x = torch.as_tensor(x0).clone()
    if x.ndim != 2:
        raise ValueError(f"x0 must be (n, d), got {tuple(x.shape)}")
    states = [x.clone()] if trajectory else None
    for s, s_next in zip(lv[:-1], lv[1:]):
        x = step(field, x, s, s_next - s)
        _check_finite(x, f"stepping from sigma={s:g} to sigma={s_next:g}")
        if states is not None:
            states.append(x.clone())
    if states is not None:
        return x, torch.stack(states)
    return x


# --------------------------------------------------------------------------
# the probability-flow ODE
# --------------------------------------------------------------------------
def pf_ode_field(score_fn: NoisedScoreFn) -> VectorField:
    """``v(x, sigma) = -sigma s(x, sigma)`` -- the field derived in the module doc.

    Takes the score as a callable for the same reason every other sampler here
    does: the identical field runs on the analytic score and on a network, and
    the difference between the two runs is estimation error and nothing else.
    """

    def v(x: Tensor, sigma: float) -> Tensor:
        return -sigma * score_fn(x, sigma)

    return v


def pf_ode_nfe(n_nodes: int, method: str = "rk4", final_step: bool = True) -> int:
    """Score evaluations :func:`pf_ode_sample` spends -- the comparison's x-axis."""
    if method not in STEP_EVALUATIONS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(STEP_EVALUATIONS)}")
    return (n_nodes - 1) * STEP_EVALUATIONS[method] + (1 if final_step else 0)


def pf_ode_sample(
    score_fn: NoisedScoreFn,
    x0: Tensor,
    sigmas: Sequence[float],
    method: str = "rk4",
    final_step: bool = True,
    trajectory: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Sample by solving ``dx/dsigma = -sigma s(x, sigma)`` down ``sigmas``.

    Parameters
    ----------
    score_fn:
        ``s(x, sigma) -> (n, d)``. Evaluated *between* the given levels by the
        RK4 midpoints, so it must be defined for any ``sigma`` in the range,
        not only at the nodes.
    x0:
        Initial states ``(n, d)``, meant as draws from ``p_{sigma_max}``; in
        practice ``N(0, sigma_max^2 I)``, the same prior-mismatch term the
        ancestral sampler carries (:mod:`diffusion.samplers`).
    sigmas:
        Strictly decreasing ladder, coarse to fine. Unlike the annealed
        Langevin ladder these are integration nodes: doubling them halves the
        step, it does not add equilibration.
    final_step:
        Take the ODE-consistent last stub ``x + (sigma_min^2 / 2) s`` down to
        ``sigma = 0``, which is half the ancestral sampler's Tweedie denoise
        (module docstring). Costs one extra score evaluation.

    The map from ``x0`` to the output is deterministic, so no generator is
    taken: two calls with the same input are bit-identical, and the diversity
    of the samples is entirely the diversity of the prior draw. That is a real
    difference from the stochastic sampler rather than a convenience -- it is
    also what makes the likelihood below well defined.
    """
    v = pf_ode_field(score_fn)
    out = integrate(v, x0, sigmas, method=method, trajectory=trajectory)
    if trajectory:
        assert isinstance(out, tuple)
        x, states = out
    else:
        assert isinstance(out, Tensor)
        x, states = out, None
    if final_step:
        sigma_min = float(sigmas[-1])
        x = x + 0.5 * sigma_min * sigma_min * score_fn(x, sigma_min)
        _check_finite(x, f"in the final stub below sigma={sigma_min:g}")
        if states is not None:
            states = torch.cat([states, x.unsqueeze(0)], dim=0)
    if states is not None:
        return x, states
    return x


def gaussian_pf_ode_scale(target_var: float, sigma_from: float, sigma_to: float) -> float:
    r"""Exact PF-ODE map on ``N(mu, v I)``: a radial scaling by this factor.

    ``sqrt( (v + sigma_to^2) / (v + sigma_from^2) )``, from separating
    ``dx/dsigma = sigma (x - mu)/(v + sigma^2)`` (module docstring). Exposed for
    the same reason :func:`diffusion.samplers.ddpm_gaussian_variance` is: a
    test that asserts a number should be able to say where the number came
    from, and here the number is the *entire solution*, so the solver can be
    checked pointwise rather than in distribution.
    """
    if target_var <= 0:
        raise ValueError(f"target_var must be positive, got {target_var}")
    if sigma_from < 0 or sigma_to < 0:
        raise ValueError("noise levels must be non-negative")
    return math.sqrt((target_var + sigma_to * sigma_to) / (target_var + sigma_from * sigma_from))


# --------------------------------------------------------------------------
# divergence
# --------------------------------------------------------------------------
def rademacher_probes(
    n_probes: int, shape: tuple[int, int], dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """``(n_probes, n, d)`` of independent +-1 entries.

    The default probe distribution: ``E[eps eps'] = I`` as required, and on a
    diagonal Jacobian ``eps' A eps = sum_i A_ii eps_i^2 = tr(A)`` *exactly*,
    with no variance at all. Gaussian probes satisfy the same identity but pay
    variance ``2 sum_i A_ii^2`` on the diagonal alone, and the fields here have
    strongly diagonal Jacobians (an isotropic score field is exactly diagonal),
    so the choice is not cosmetic -- ``tests/test_ode.py`` measures the gap.
    """
    if n_probes < 1:
        raise ValueError(f"n_probes must be >= 1, got {n_probes}")
    bits = torch.randint(
        0, 2, (n_probes, *shape), dtype=torch.int8, generator=generator
    ).to(dtype)
    return 2.0 * bits - 1.0


def gaussian_probes(
    n_probes: int, shape: tuple[int, int], dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """``(n_probes, n, d)`` standard normal probes -- the higher-variance option."""
    if n_probes < 1:
        raise ValueError(f"n_probes must be >= 1, got {n_probes}")
    return torch.randn((n_probes, *shape), dtype=dtype, generator=generator)


def _jacobian_vector_products(
    field: VectorField, x: Tensor, s: float, grad_outputs: Tensor
) -> tuple[Tensor, Tensor]:
    """``field(x, s)`` and, for each row of ``grad_outputs``, ``u' (grad field)``.

    One forward pass, one reverse-mode pass per ``u`` -- the vector-Jacobian
    product ``u' J``, which is what both estimators below are built from
    (the exact trace picks ``u`` from the basis, Hutchinson picks it at random).
    """
    xg = x.detach().requires_grad_(True)
    with torch.enable_grad():
        v = field(xg, s)
        if not v.requires_grad:
            raise ValueError(
                "the field is not differentiable in x: no autograd graph reached "
                "its output. A score wrapped in torch.no_grad() (as the samplers "
                "do) cannot be used for divergences -- pass the differentiable "
                "score instead."
            )
        rows = [
            torch.autograd.grad(v, xg, grad_outputs=u, retain_graph=True)[0]
            for u in grad_outputs
        ]
    return v.detach(), torch.stack(rows).detach()


def _coordinate_basis(x: Tensor) -> Tensor:
    """``(d, n, d)``: the ``j``-th slice is ``e_j`` repeated over the batch."""
    return torch.stack(
        [
            torch.zeros_like(x).index_fill_(1, torch.tensor([j]), 1.0)
            for j in range(x.shape[1])
        ]
    )


def _trace_from_basis_rows(rows: Tensor) -> Tensor:
    """``rows[j] = e_j' J``, so the trace is the sum of the ``j``-th columns."""
    return torch.stack([rows[j, :, j] for j in range(rows.shape[0])]).sum(dim=0)


def exact_divergence(field: VectorField, x: Tensor, s: float) -> Tensor:
    """``div v(x, s)`` exactly, by ``d`` reverse-mode passes -> ``(n,)``.

    Sums ``e_j' J e_j`` over the coordinate basis, so the cost is the dimension
    of the state. Unaffordable at image scale -- which is the entire reason the
    Hutchinson estimator exists -- and completely affordable at ``d = 2``,
    which is why this repo can report what the estimator's variance actually
    costs instead of assuming it away.
    """
    x = x.detach()
    _, rows = _jacobian_vector_products(field, x, s, _coordinate_basis(x))
    return _trace_from_basis_rows(rows)


def hutchinson_divergence(field: VectorField, x: Tensor, s: float, probes: Tensor) -> Tensor:
    """``eps' J eps`` for each probe -> ``(n_probes, n)``.

    Unbiased for ``div v`` under any ``eps`` with ``E[eps eps'] = I``. Returns
    the per-probe values rather than their mean so that a caller can report the
    Monte Carlo spread; averaging away the one number that says how noisy the
    estimate is would defeat the point of having a ground truth to compare to.

    ``probes`` are the caller's, deliberately: the likelihood solver draws them
    once and reuses them at every stage of every step, because resampling
    inside the integrator makes the augmented field discontinuous in ``s`` and
    quietly costs the solver its order.
    """
    if probes.ndim != 3 or probes.shape[1:] != x.shape:
        raise ValueError(
            f"probes must be (n_probes, {tuple(x.shape)}), got {tuple(probes.shape)}"
        )
    _, rows = _jacobian_vector_products(field, x, s, probes)
    return (rows * probes).sum(dim=2)


# --------------------------------------------------------------------------
# likelihood
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LogProbResult:
    """What :func:`pf_ode_log_prob` returns.

    ``log_prob`` is the probe-averaged estimate ``(n,)``; ``per_probe`` is the
    ``(n_probes, n)`` matrix it came from, so the estimator's own spread is
    available; ``log_det`` ``(n_probes, n)`` is the accumulated
    ``int div v ds`` (the log-Jacobian of the flow); ``z`` ``(n, d)`` is the
    endpoint at the top of the ladder; ``nfe`` counts score evaluations.
    """

    log_prob: Tensor
    per_probe: Tensor
    log_det: Tensor
    z: Tensor
    nfe: int

    @property
    def stderr(self) -> Tensor:
        """Standard error over probes ``(n,)``; zeros when there is one probe."""
        if self.per_probe.shape[0] < 2:
            return torch.zeros_like(self.log_prob)
        k = self.per_probe.shape[0]
        return self.per_probe.std(dim=0, unbiased=True) / math.sqrt(k)


def pf_ode_log_prob(
    score_fn: NoisedScoreFn,
    x: Tensor,
    sigmas: Sequence[float],
    log_prob_top: Callable[[Tensor], Tensor],
    n_probes: int = 1,
    method: str = "rk4",
    exact: bool = False,
    generator: torch.Generator | None = None,
    probe_distribution: str = "rademacher",
) -> LogProbResult:
    r"""``log p_{sigma_min}(x)`` by transporting the density up the ladder.

    Solves the augmented system

        ``dx/ds = v(x, s)``,   ``dl/ds = div v(x, s)``,   ``v = -s * score_fn``,

    from ``sigma_min`` to ``sigma_max`` -- both components with the same
    stepper, which is what keeps the pair fourth-order together -- and returns

        ``log p_{sigma_min}(x) = log_prob_top(z) + l``.

    Parameters
    ----------
    x:
        Points ``(n, d)`` whose density is wanted, at the *bottom* of the
        ladder. Note that this is the density of ``p_{sigma_min}``, not of the
        clean target: the model is only defined down to ``sigma_min``, and on
        the mixture target the exact ``p_{sigma_min}`` is available in closed
        form (``target.noised(sigma_min)``), which is what makes this
        checkable.
    sigmas:
        The same decreasing ladder the samplers take; it is traversed in
        reverse here, so one ladder describes both directions.
    log_prob_top:
        ``log p_{sigma_max}(z) -> (n,)``. For the mixture that is exact; for a
        model that only claims a Gaussian prior it is ``N(0, sigma_max^2 I)``,
        and then the returned number is a lower-bound-flavoured
        approximation rather than the model's own density -- the same prior
        mismatch the samplers carry, appearing here in the likelihood.
    n_probes:
        Hutchinson probes, drawn once and held fixed for the whole solve.
    exact:
        Use :func:`exact_divergence` instead (``d`` passes per evaluation, no
        variance). Affordable at ``d = 2`` and the control that separates
        estimator variance from integration error.

    ``score_fn`` must be differentiable in ``x`` -- a ``no_grad``-wrapped score
    raises rather than silently returning a zero divergence.
    """
    if method not in _STEPPERS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_STEPPERS)}")
    lv = _check_nodes(sigmas)
    if lv[0] < lv[-1]:
        raise ValueError("sigmas must be given coarse -> fine, as the samplers take them")
    nodes = lv[::-1]  # integrate up: sigma_min -> sigma_max

    x = torch.as_tensor(x)
    if x.ndim != 2:
        raise ValueError(f"x must be (n, d), got {tuple(x.shape)}")
    v = pf_ode_field(score_fn)

    if exact:
        probes = None
        n_rows = 1
    else:
        maker = {"rademacher": rademacher_probes, "gaussian": gaussian_probes}
        if probe_distribution not in maker:
            raise ValueError(
                f"unknown probe_distribution {probe_distribution!r}; "
                f"choose from {sorted(maker)}"
            )
        probes = maker[probe_distribution](
            n_probes, (x.shape[0], x.shape[1]), dtype=x.dtype, generator=generator
        )
        n_rows = n_probes

    def field_and_div(state: Tensor, s: float) -> tuple[Tensor, Tensor]:
        """Both halves of the augmented field from one forward pass.

        Deliberately not ``v(state, s)`` plus a call to the divergence
        helpers: those recompute the forward pass, and with a network in the
        field the second copy also leaves an autograd graph attached to the
        integrator's state, which then grows for the whole solve.
        """
        if probes is None:
            val, rows = _jacobian_vector_products(v, state, s, _coordinate_basis(state))
            return val, _trace_from_basis_rows(rows).unsqueeze(0)
        val, rows = _jacobian_vector_products(v, state, s, probes)
        return val, (rows * probes).sum(dim=2)

    state = x.detach().clone()
    log_det = torch.zeros(n_rows, x.shape[0], dtype=x.dtype)
    evals = 0
    for s, s_next in zip(nodes[:-1], nodes[1:]):
        h = s_next - s
        if method == "euler":
            k, d = field_and_div(state, s)
            state = state + h * k
            log_det = log_det + h * d
            evals += 1
        else:
            k1, d1 = field_and_div(state, s)
            k2, d2 = field_and_div(state + 0.5 * h * k1, s + 0.5 * h)
            k3, d3 = field_and_div(state + 0.5 * h * k2, s + 0.5 * h)
            k4, d4 = field_and_div(state + h * k3, s + h)
            state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            log_det = log_det + (h / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
            evals += 4
        _check_finite(state, f"stepping from sigma={s:g} to sigma={s_next:g}")

    top = log_prob_top(state)
    if top.shape != (x.shape[0],):
        raise ValueError(f"log_prob_top must return ({x.shape[0]},), got {tuple(top.shape)}")
    per_probe = top.unsqueeze(0) + log_det
    return LogProbResult(
        log_prob=per_probe.mean(dim=0),
        per_probe=per_probe,
        log_det=log_det,
        z=state,
        nfe=evals,
    )
