"""Langevin dynamics: sampling from a density given only its score.

The overdamped Langevin SDE

    dX_t = grad log p(X_t) dt + sqrt(2) dW_t

has ``p`` as its stationary distribution. Nothing in that statement mentions
``p`` itself -- only ``grad log p``. That is the entire reason score-based
generative modeling works: **learn the score and you can sample, without ever
touching a normalizing constant.**

Euler-Maruyama at step ``tau`` gives the unadjusted Langevin algorithm (ULA):

    x <- x + (tau / 2) s(x) + sqrt(tau) z,     z ~ N(0, I).

The bridge to mcmc-from-scratch
-------------------------------
That update is *exactly* the MALA proposal in the sibling repo
mcmc-from-scratch (``mcmc/mala.py``, written there with ``tau = eps^2``),
with the Metropolis accept step deleted. The deletion is not free and this
module does not pretend it is:

- **MALA is exact.** The Hastings correction repairs the discretization error
  for any step size; the chain's stationary law is ``p`` on the nose.
- **ULA is biased.** Its stationary law is ``p`` only as ``tau -> 0``. The
  bias is analytic on a standard Gaussian, where the update is linear:
  ``x' = (1 - tau/2) x + sqrt(tau) z`` gives stationary variance
  ``V = V (1 - tau/2)^2 + tau``, i.e.

      ``V = 1 / (1 - tau/4)``,

  so ULA is over-dispersed by 6.7% in variance at ``tau = 0.25`` and 19% at
  ``tau = 0.64`` -- a bias no amount of extra sampling removes. mcmc's
  theory doc derives this as Exercise 1(c) and measures it against MALA;
  ``tests/test_langevin.py`` here checks the same closed form from the other
  side. Anyone reaching for ULA where the target's density is available
  should use MALA instead.

So why use ULA at all? Because in a diffusion model **the density is not
available**. A learned network gives ``s_theta(x, sigma)`` and no ``log p``,
so there is nothing to put in a Metropolis ratio -- the accept step is not
skipped for speed, it is unavailable in principle. The honest framing is that
score-based sampling trades MALA's exactness for the ability to run at all on
a target known only through its score.

Annealing: why plain ULA is not enough
--------------------------------------
ULA is also a *local* method. On a multimodal target it descends into whichever
mode it starts in and stays there: crossing a barrier of depth ``Delta`` nats
is an ``e^{-Delta}`` event per attempt, and for well-separated modes that is
never. :func:`annealed_langevin` is Song & Ermon's (2019) fix. Run ULA on a
sequence of *noised* targets ``p_{sigma_1}, ..., p_{sigma_L}`` with
``sigma_1 > ... > sigma_L``, carrying the state forward. At large ``sigma``
the noised density is nearly unimodal -- the modes have been smeared into each
other -- so the chain moves globally and its position is a draw from something
close to the right *coarse* structure; as ``sigma`` shrinks the target sharpens
back toward ``p`` and the chain refines locally. The mixing problem is solved
at the scale where it is easy, and the answer is carried down.

The step size must shrink with the noise level. Song & Ermon set

    ``tau_i = tau * sigma_i^2 / sigma_L^2``,

which keeps the ratio of the signal term to the noise term constant across
levels: for an isotropic Gaussian of scale ``sigma_i`` the score is ``O(1/sigma_i)``
in magnitude and the drift ``(tau_i/2) s`` is then ``O(tau sigma_i / sigma_L^2)``,
matching the injected noise ``sqrt(tau_i) = sqrt(tau) sigma_i / sigma_L`` up to
a level-independent constant. Without that scaling a single step size is either
too small to move at the coarse levels or too large to be stable at the fine
ones -- the same resonance problem a unit mass matrix has in HMC (mcmc's
Sec. 4.8), reappearing along the noise axis instead of across coordinates.

How large can tau be? A bound that holds for every target
------------------------------------------------------------
Noising bounds the curvature, and that bounds the step size. For
``p_sigma = p * N(0, sigma^2 I)`` the score can be written as a posterior mean,

    ``s_sigma(x) = (E[x_0 | x] - x) / sigma^2``,

whose Jacobian is ``(Cov[x_0 | x] / sigma^2 - I) / sigma^2``. A covariance is
positive semidefinite, so

    ``grad s_sigma(x) >= -I / sigma^2``    for **any** ``p``.

Linearize the ULA update around a point: the deterministic part multiplies
deviations by ``I + (tau/2) grad s``, with eigenvalues ``1 + (tau/2) lambda``.
Contraction needs ``|1 + (tau/2) lambda| < 1``, i.e. ``tau |lambda| < 4``, and
since ``lambda >= -1/sigma^2`` that is guaranteed whenever

    ``tau_i < 4 sigma_i^2``,   which under the schedule above is   ``tau < 4 sigma_L^2``

-- one condition, on the finest level, that makes every level stable. It is
the same ``4`` as in :func:`ula_gaussian_variance`, which is this statement for
``N(0, 1)`` (where ``lambda = -1`` exactly).

:func:`annealed_langevin` **refuses** a ``tau`` that violates it, because the
failure is otherwise very hard to see. An unstable ladder does not hand back
obvious garbage: the coarse levels blow the state up geometrically (to
``1e80`` and beyond in the repo's own bimodal test), and then the fine levels,
which are strongly contractive, squash it back into range. What comes out
looks like plausible samples with roughly the right support -- but the coarse
exploration that was supposed to decide *which mode each chain belongs to* has
been destroyed, so the mode weights come back wrong. On a symmetric target
that is invisible; ``tests/test_langevin.py`` catches it only because one test
uses an *asymmetric* mixture, where the recovered weight reads 0.53 instead of
0.75. Satisfying the bound is necessary but not sufficient: even at half the
bound the same test recovers 0.70, and it takes ``tau`` a few times smaller
still before the weights are right.

This module takes the score as a **callable**, so the same code runs on an
exact score from :mod:`diffusion.targets` and on a learned one later. Today
every experiment uses the exact score, which is what makes the annealing claim
testable: any failure is the sampler's, not an estimation error.

``theory/derivations.md`` Sec. 6 collects all of this as the bridge back to
mcmc-from-scratch, including the one place the analogy stops holding: an
equilibrium sampler handed a *learned* score converges harder to the wrong
answer the longer it is run.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch

Tensor = torch.Tensor

#: ``s(x) -> (n, d)``: the score of a fixed density.
ScoreFn = Callable[[Tensor], Tensor]
#: ``s(x, sigma) -> (n, d)``: the score of the ``sigma``-noised density.
NoisedScoreFn = Callable[[Tensor, float], Tensor]

__all__ = [
    "ScoreFn",
    "NoisedScoreFn",
    "unadjusted_langevin",
    "annealed_langevin",
    "geometric_sigmas",
    "ula_gaussian_variance",
    "stability_bound",
]


def stability_bound(sigma_min: float) -> float:
    """Largest ``tau`` for which the annealed ladder is stable: ``4 sigma_L^2``.

    Target-independent, because noising bounds curvature: the Jacobian of the
    score of a ``sigma``-noised density is bounded below by ``-I/sigma^2`` for
    any base density (derivation in the module docstring). Satisfying this is
    necessary, not sufficient -- being just inside it still leaves visible
    discretization bias.
    """
    if sigma_min <= 0:
        raise ValueError(f"sigma_min must be positive, got {sigma_min}")
    return 4.0 * sigma_min * sigma_min


def _check_finite(x: Tensor, where: str) -> None:
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(
            f"Langevin state became non-finite {where}. The step size is too "
            "large for the curvature of the target: the linearized update "
            "multiplies deviations by 1 + (tau/2) grad s, which is unstable "
            "once tau |lambda| >= 4. See stability_bound()."
        )


def ula_gaussian_variance(tau: float) -> float:
    """Stationary variance of ULA on a standard Gaussian: ``1 / (1 - tau/4)``.

    The closed form derived in this module's docstring, exposed because it is
    the sharpest available statement of what the missing accept step costs --
    and because a test that asserts a number should be able to name where the
    number comes from. Valid for ``tau < 4``; at ``tau >= 4`` the linear map
    ``(1 - tau/2)`` has magnitude >= 1 and the chain is transient (it diverges
    rather than settling at a wrong variance), so the function refuses.
    """
    if not 0 < tau < 4:
        raise ValueError(f"ULA on a standard Gaussian is stationary only for 0 < tau < 4, got {tau}")
    return 1.0 / (1.0 - tau / 4.0)


def geometric_sigmas(sigma_max: float, sigma_min: float, n_levels: int) -> list[float]:
    """Geometrically spaced noise levels, high to low.

    Geometric rather than linear because what has to stay bounded between
    consecutive levels is the *ratio* ``sigma_i / sigma_{i+1}``: the chain
    arrives at level ``i+1`` holding samples from ``p_{sigma_i}``, and that is
    a useful initialization only if the two densities overlap substantially.
    Song & Ermon (2020) make this precise and recommend a ratio near 1;
    linear spacing would take enormous steps in ratio at the low end, exactly
    where the density is sharpest.
    """
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")
    if not (sigma_max >= sigma_min > 0):
        raise ValueError("need sigma_max >= sigma_min > 0")
    if n_levels == 1:
        return [float(sigma_max)]
    ratio = (sigma_min / sigma_max) ** (1.0 / (n_levels - 1))
    return [float(sigma_max * ratio**i) for i in range(n_levels)]


def unadjusted_langevin(
    score_fn: ScoreFn,
    x0: Tensor,
    tau: float,
    n_steps: int,
    generator: torch.Generator | None = None,
    trajectory: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run ULA: ``x <- x + (tau/2) s(x) + sqrt(tau) z``, ``n_steps`` times.

    Parameters
    ----------
    score_fn:
        ``s(x)`` for a batch of points ``(n, d) -> (n, d)``.
    x0:
        Initial states ``(n, d)``. Every row is an independent chain; the
        whole thing is one batched update, so ``n = 10_000`` costs what
        ``n = 1`` costs plus arithmetic.
    tau:
        Step size in the ``x <- x + (tau/2) s + sqrt(tau) z`` convention (so
        the noise standard deviation is ``sqrt(tau)``, and ``tau = eps^2`` in
        mcmc-from-scratch's MALA parameterization).
    trajectory:
        If true, also return every state visited, ``(n_steps + 1, n, d)``,
        for the mixing figures. Off by default because it is ``n_steps``
        times the memory.

    Returns the final states ``(n, d)``, or ``(final, trajectory)``.

    No accept step, so no acceptance rate to report and no exactness: see the
    module docstring. The bias is ``O(tau)`` and does not vanish with more
    steps, which is why the tests fix ``tau`` small and compare against the
    *predicted* biased answer rather than the target's true moments where the
    difference matters.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")
    x = torch.as_tensor(x0).clone()
    noise_scale = math.sqrt(tau)
    states = [x.clone()] if trajectory else None
    for _ in range(n_steps):
        z = torch.randn(x.shape, dtype=x.dtype, generator=generator)
        x = x + 0.5 * tau * score_fn(x) + noise_scale * z
        if states is not None:
            states.append(x.clone())
    _check_finite(x, "during unadjusted_langevin")
    if states is not None:
        return x, torch.stack(states)
    return x


def annealed_langevin(
    score_fn: NoisedScoreFn,
    x0: Tensor,
    sigmas: Sequence[float],
    tau: float,
    n_steps_per_level: int,
    generator: torch.Generator | None = None,
    trajectory: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Annealed Langevin (Song & Ermon 2019, Alg. 1).

    For each noise level ``sigma_i`` in ``sigmas`` (highest first), run
    ``n_steps_per_level`` ULA steps against ``s(., sigma_i)`` with step size
    ``tau_i = tau * sigma_i^2 / sigma_L^2``, carrying the state forward to the
    next level.

    ``tau`` is the step size *at the finest level*: at ``sigma = sigma_L`` the
    scaling factor is 1 by construction, so ``tau`` means exactly what it
    means in :func:`unadjusted_langevin`, and the coarse levels take the
    larger steps their flatter targets can absorb.

    Raises if ``tau >= 4 sigma_L^2`` (:func:`stability_bound`), and again if
    the state goes non-finite at any level. Both are hard errors rather than
    warnings for the reason given in the module docstring: an unstable ladder
    returns plausible-looking samples with the wrong mode weights, so it will
    not be noticed by looking.

    ``x0`` should be dispersed, not concentrated: the point of starting at
    ``sigma_1`` is that the target there is broad, so an initialization that
    covers it (uniform over the domain, or ``N(0, sigma_1^2 I)``) lets the
    chain sort out which mode it belongs to. Starting all chains at one point
    wastes the coarse levels.

    With ``trajectory=True`` the second return value is
    ``(L * n_steps_per_level + 1, n, d)`` -- the full annealing path, which is
    what the noise-level figures are drawn from.
    """
    if len(sigmas) == 0:
        raise ValueError("sigmas must be non-empty")
    if any(s <= 0 for s in sigmas):
        raise ValueError("noise levels must be positive")
    if list(sigmas) != sorted(sigmas, reverse=True):
        raise ValueError("sigmas must be in decreasing order (coarse -> fine)")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    sigma_min = float(sigmas[-1])
    bound = stability_bound(sigma_min)
    if tau >= bound:
        raise ValueError(
            f"tau={tau} is at or above the stability bound 4*sigma_min^2={bound:g} "
            f"for sigma_min={sigma_min}, so every level of this ladder is "
            "transient. An unstable ladder does NOT fail loudly -- the fine "
            "levels contract the blown-up state back into range and return "
            "plausible-looking samples with the mode weights wrong. Lower tau "
            "(well below the bound, not just under it) or raise sigma_min."
        )

    x = torch.as_tensor(x0).clone()
    states = [x.clone()] if trajectory else None
    for sigma in sigmas:
        tau_i = tau * (float(sigma) / sigma_min) ** 2
        noise_scale = math.sqrt(tau_i)
        for _ in range(n_steps_per_level):
            z = torch.randn(x.shape, dtype=x.dtype, generator=generator)
            x = x + 0.5 * tau_i * score_fn(x, float(sigma)) + noise_scale * z
            if states is not None:
                states.append(x.clone())
        _check_finite(x, f"at noise level sigma={sigma:g} (tau_i={tau_i:g})")
    if states is not None:
        return x, torch.stack(states)
    return x
