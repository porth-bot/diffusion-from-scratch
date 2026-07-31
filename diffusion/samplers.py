r"""Reverse-time sampling: turning a score back into draws.

:mod:`diffusion.langevin` samples a *fixed* density by running a chain to
equilibrium at each noise level. This module does the other thing: it walks
the forward noising process **backwards**, one level per step, so that a state
drawn from the prior at ``sigma_max`` comes out as a draw from ``p_0``. There
is no equilibration and no burn-in; the correctness of the answer rests on the
correctness of a single sequence of conditional distributions.

The forward chain
-----------------
This repo's forward process is variance-exploding: for a ladder
``sigma_L > ... > sigma_1 > 0``,

    ``x_i = x_0 + sigma_i eps``,   ``eps ~ N(0, I)``,

which is Markov in ``i`` because

    ``x_i = x_{i-1} + sqrt(sigma_i^2 - sigma_{i-1}^2) z``.

(That is the whole content of "variance exploding": the noise added between
two levels is independent of everything earlier, and the variances add.)

The reverse conditional, exactly
--------------------------------
Condition on the clean point. Both ``x_i`` and ``x_{i-1}`` are ``x_0`` plus
independent Gaussian noise, so ``q(x_{i-1} | x_i, x_0)`` is Gaussian and its
mean is the standard "combine two Gaussian observations" answer, weighted by
precisions:

    ``q(x_{i-1} | x_i, x_0) = N( a x_i + (1 - a) x_0,  a (sigma_i^2 - sigma_{i-1}^2) I )``,
    ``a := sigma_{i-1}^2 / sigma_i^2``.

*Derivation.* Write ``u = x_{i-1} - x_0`` and ``v = x_i - x_{i-1}``, independent
with variances ``sigma_{i-1}^2`` and ``sigma_i^2 - sigma_{i-1}^2``. Conditioning
the pair on their sum ``u + v = x_i - x_0`` gives a Gaussian with mean
``(sigma_{i-1}^2 / sigma_i^2)(x_i - x_0)`` and variance
``sigma_{i-1}^2 (sigma_i^2 - sigma_{i-1}^2)/sigma_i^2``; add ``x_0`` back.

``x_0`` is of course not available at sampling time. Replace it by its
posterior mean, which **Tweedie's formula** hands over in terms of the score
and nothing else:

    ``E[x_0 | x_i] = x_i + sigma_i^2 s(x_i, sigma_i)``.

Substituting collapses the two terms into one:

    ``a x_i + (1 - a) (x_i + sigma_i^2 s) = x_i + (sigma_i^2 - sigma_{i-1}^2) s``,

and the sampler is

    ``x_{i-1} = x_i + (sigma_i^2 - sigma_{i-1}^2) s(x_i, sigma_i)``
               ``+ sqrt( sigma_{i-1}^2 (sigma_i^2 - sigma_{i-1}^2) / sigma_i^2 ) z``.

This is DDPM's ancestral sampler (Ho et al. 2020, Eq. 11), written for the
variance-exploding chain instead of the variance-preserving one; Song et al.
(2021, App. F) give the same update as the ancestral sampler of the SMLD
model. Setting ``sigma_{i-1} = 0`` in the same formula makes the noise term
vanish and the mean equal ``E[x_0 | x_i]`` exactly -- so the final denoising
step is not an extra heuristic, it is the last rung of the ladder.

The approximation, and what it costs
------------------------------------
Only one step above is not exact: ``p(x_0 | x_i)`` was replaced by a point mass
at its mean. The true reverse conditional is the *mixture*

    ``q(x_{i-1} | x_i) = int q(x_{i-1} | x_i, x_0) p(x_0 | x_i) dx_0``,

whose mean agrees (the plug-in enters linearly) but whose covariance is larger
by the spread of that posterior propagated through the same linear map:

    ``Cov[x_{i-1} | x_i] = a (sigma_i^2 - sigma_{i-1}^2) I + (1 - a)^2 Cov[x_0 | x_i]``.

So **ancestral sampling under-disperses**, by an amount that is second order in
the gap between adjacent levels (``1 - a -> 0`` as the ladder is refined) and
that vanishes wherever the denoising problem is nearly deterministic. On a
Gaussian target the deficit is available in closed form and
:func:`ddpm_gaussian_variance` returns it, which is how the tests here assert
what the sampler *does* rather than what it should do in the limit: a coarse
ladder is expected to come back narrow, and it does, by the predicted factor.

Three sources of error, and how to separate them
------------------------------------------------
A sample from this module can be wrong for three reasons, and the point of a
target with a known score is that they can be measured apart:

1. **Prior mismatch.** The chain starts from ``N(0, sigma_max^2 I)``, but the
   true distribution at the top of the ladder is ``p_{sigma_max}``, which is
   that convolved with the data. They agree only as ``sigma_max -> infinity``.
   Passing an exact draw from ``p_{sigma_max}`` as ``x0`` removes this term.
2. **Discretization.** The plug-in approximation above, at a finite number of
   levels. Removed only in the limit; measured by refining the ladder.
3. **Score estimation.** ``s_theta`` instead of ``s``. Removed by passing the
   analytic score, which the mixture target provides at every level.

``experiments/sampling.py`` runs all three controls.

``theory/derivations.md`` Sec. 4 places this discrete derivation next to the
continuous one it discretizes -- the forward SDE and Anderson's reverse-time
SDE -- and checks that the two give the same mean update.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from diffusion.langevin import NoisedScoreFn

Tensor = torch.Tensor

__all__ = [
    "ddpm_ancestral",
    "ddpm_gaussian_variance",
]


def _check_finite(x: Tensor, where: str) -> None:
    """Ancestral sampling has no step size to blame, so say what it can be.

    The update is a contraction whenever the score is, so a non-finite state
    here means the *score* returned something non-finite (an untrained network
    evaluated far outside its training support is the usual cause), not that a
    knob is set too high -- which is the opposite of the Langevin failure mode
    and worth distinguishing at the point it happens.
    """
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(
            f"ancestral sampler state became non-finite {where}. The update is "
            "affine in the score, so this means score_fn returned a non-finite "
            "value -- check the score at states this far out, not the ladder."
        )


def _check_ladder(sigmas: Sequence[float]) -> list[float]:
    lv = [float(s) for s in sigmas]
    if len(lv) < 2:
        raise ValueError("an ancestral ladder needs at least two levels")
    if any(s <= 0 for s in lv):
        raise ValueError("noise levels must be positive")
    if lv != sorted(lv, reverse=True):
        raise ValueError("sigmas must be in decreasing order (coarse -> fine)")
    if len(set(lv)) != len(lv):
        raise ValueError("sigmas must be strictly decreasing (a repeated level is a no-op step)")
    return lv


def ddpm_ancestral(
    score_fn: NoisedScoreFn,
    x0: Tensor,
    sigmas: Sequence[float],
    generator: torch.Generator | None = None,
    denoise_final: bool = True,
    trajectory: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Ancestral sampling down the ladder: one score evaluation per level.

    Runs the update derived in the module docstring,

        ``x <- x + (sigma^2 - sigma_next^2) s(x, sigma)``
              ``+ sqrt(sigma_next^2 (sigma^2 - sigma_next^2) / sigma^2) z``,

    once for each consecutive pair of ``sigmas``.

    Parameters
    ----------
    score_fn:
        ``s(x, sigma) -> (n, d)``, the score of the ``sigma``-noised density.
        Taking it as a callable is what lets the same code run on the analytic
        score and on a network, which is the control the experiment needs.
    x0:
        Initial states ``(n, d)``, meant to be draws from ``p_{sigma_max}``.
        In practice ``N(0, sigma_max^2 I)`` is used and the mismatch is one of
        the three error terms named in the module docstring; for the mixture
        target an exact draw is available and isolates the other two.
    sigmas:
        Strictly decreasing, coarse to fine. Unlike the annealed Langevin
        ladder this one is not a schedule of "how long to equilibrate" -- each
        level is visited exactly once, so the number of levels *is* the
        discretization.
    denoise_final:
        Append the exact ``sigma -> 0`` step, ``x <- x + sigma_1^2 s``, which
        by Tweedie is ``E[x_0 | x]`` and carries no noise. On by default: the
        alternative is to return samples that still carry ``sigma_1`` of noise
        by construction, which on this repo's ladder is a visible 0.05 blur
        that has nothing to do with the model.

    Returns the final states ``(n, d)``, or ``(final, trajectory)`` with the
    trajectory ``(n_steps + 1, n, d)``.

    Cost is ``len(sigmas) - 1 + denoise_final`` score evaluations -- the
    number of function evaluations (NFE) that the ODE sampler of Day 26 is
    compared against, and the axis the discretization error is measured along.
    """
    lv = _check_ladder(sigmas)
    if denoise_final:
        lv = lv + [0.0]

    x = torch.as_tensor(x0).clone()
    if x.ndim != 2:
        raise ValueError(f"x0 must be (n, d), got {tuple(x.shape)}")
    states = [x.clone()] if trajectory else None

    for sigma, sigma_next in zip(lv[:-1], lv[1:]):
        gap = sigma * sigma - sigma_next * sigma_next
        x = x + gap * score_fn(x, sigma)
        if sigma_next > 0.0:
            noise_scale = math.sqrt(gap) * sigma_next / sigma
            x = x + noise_scale * torch.randn(
                x.shape, dtype=x.dtype, generator=generator
            )
        if states is not None:
            states.append(x.clone())
        _check_finite(x, f"at noise level sigma={sigma:g}")

    if states is not None:
        return x, torch.stack(states)
    return x


def ddpm_gaussian_variance(
    target_var: float,
    sigmas: Sequence[float],
    init_var: float | None = None,
    denoise_final: bool = True,
) -> float:
    r"""Per-coordinate variance :func:`ddpm_ancestral` returns on ``N(mu, v I)``.

    Not the target's variance -- the variance the sampler actually produces at
    a finite number of levels, which is smaller. This function exists for the
    same reason :func:`diffusion.langevin.ula_gaussian_variance` does: a test
    that asserts a number should be able to say where the number came from.

    On an isotropic Gaussian target the whole chain is linear-Gaussian, so the
    answer is a two-line recursion. With ``v`` the target variance and
    ``V_i`` the state variance at level ``i``, the exact score of
    ``p_{sigma_i} = N(mu, (v + sigma_i^2) I)`` is
    ``s = -(x - mu)/(v + sigma_i^2)``, so the update is affine:

        ``x_{i-1} - mu = c_i (x_i - mu) + sqrt(beta_i) z``,
        ``c_i = 1 - (sigma_i^2 - sigma_{i-1}^2)/(v + sigma_i^2)``,
        ``beta_i = sigma_{i-1}^2 (sigma_i^2 - sigma_{i-1}^2)/sigma_i^2``,

    hence ``V_{i-1} = c_i^2 V_i + beta_i``, iterated down the ladder from
    ``V_L = init_var`` (default ``v + sigma_L^2``, i.e. an exact draw from the
    top of the ladder).

    The gap between the result and ``v`` is the discretization error of the
    plug-in step, and it closes as the ladder is refined: at 8 levels over
    ``[0.05, 8]`` on a unit-variance target this returns 0.528, at 128 levels
    0.959, at 2048 levels 0.995. That is not a bug being tolerated, it is the
    module docstring's ``(1 - a)^2 Cov[x_0 | x]`` term, arithmetic.
    """
    if target_var <= 0:
        raise ValueError(f"target_var must be positive, got {target_var}")
    lv = _check_ladder(sigmas)
    var = float(target_var + lv[0] ** 2) if init_var is None else float(init_var)
    if denoise_final:
        lv = lv + [0.0]
    for sigma, sigma_next in zip(lv[:-1], lv[1:]):
        gap = sigma * sigma - sigma_next * sigma_next
        c = 1.0 - gap / (target_var + sigma * sigma)
        beta = gap * (sigma_next / sigma) ** 2
        var = c * c * var + beta
    return var
