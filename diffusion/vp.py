r"""The variance-preserving process, and the coordinate change that is all it is.

Every other module here uses the **variance-exploding** (VE) forward process:
``x_i = x_0 + sigma_i eps``, no drift, variance growing without bound.
DDPM (Ho et al. 2020) uses the **variance-preserving** (VP) one,

    ``xt_i = sqrt(abar_i) x_0 + sqrt(1 - abar_i) eps``,      ``abar_i`` decreasing,

which contracts the signal as it adds noise so that the state stays ``O(1)``.
The README listed "a VP-vs-VE comparison is not run" as a limitation. This
module runs it, and the answer is sharper than "they are similar".

The change of variables
-----------------------
Match the two at the *signal-to-noise ratio*, which is the only thing either
process is really parameterized by:

    ``sigma^2 = (1 - abar) / abar``,        ``abar = 1 / (1 + sigma^2)``.        (1)

Then, for the *same* ``x_0`` and the *same* ``eps``,

    ``sqrt(abar_i) (x_0 + sigma_i eps) = sqrt(abar_i) x_0 + sqrt(1 - abar_i) eps
                                       = xt_i``,                                 (2)

since ``sqrt(abar) sigma = sqrt(abar (1-abar)/abar) = sqrt(1-abar)``. So the VP
state is the VE state scaled by ``sqrt(abar)``, pathwise -- not merely in
distribution. Densities and scores follow by the usual Jacobian:

    ``p_VP(xt) = abar^{-d/2} p_VE(xt / sqrt(abar))``,
    ``s_VP(xt) = s_VE(xt / sqrt(abar)) / sqrt(abar)``.                           (3)

Tweedie in VP coordinates comes out of (3) and Sec. 2.3 with no new work:
``E[x_0 | xt] = (xt + (1 - abar) s_VP(xt)) / sqrt(abar)``.

The samplers are the same chain, step for step
----------------------------------------------
DDPM's ancestral update is usually written for VP. Deriving it the way
:mod:`diffusion.samplers` derives the VE one -- exact reverse conditional given
``x_0``, then Tweedie for ``x_0`` -- gives

    ``xt_{i-1} = (xt_i + beta_i s_VP(xt_i)) / sqrt(alpha_i) + sqrt(betat_i) z``,
    ``alpha_i = abar_i / abar_{i-1}``,   ``beta_i = 1 - alpha_i``,
    ``betat_i = beta_i (1 - abar_{i-1}) / (1 - abar_i)``.                        (4)

Substituting (1) and (3) into (4) collapses it onto the VE update of
:func:`diffusion.samplers.ddpm_ancestral` exactly:

    ``beta_i / abar_i = 1/abar_i - 1/abar_{i-1} = sigma_i^2 - sigma_{i-1}^2``,
    ``betat_i = abar_{i-1} sigma_{i-1}^2 (sigma_i^2 - sigma_{i-1}^2) / sigma_i^2``,

so the mean update is ``sqrt(abar_{i-1})`` times the VE mean update and the
noise standard deviation is ``sqrt(abar_{i-1})`` times the VE one -- the same
scaling, on both terms, which is what makes it a coordinate change rather than
a coincidence. Feed both samplers the same ``z`` and the trajectories agree to
float precision, which is what ``tests/test_vp.py`` asserts rather than
comparing histograms.

**So the choice of process is not a modelling choice.** Anything measured in
this repo under VE -- the discretization deficit of Sec. 4.4, the schedule
cancelling out of the probability-flow ODE (Sec. 5.2), the score-estimation
error of Sec. 3.5 -- transfers unchanged. That is a negative result about a
distinction the literature spends a lot of words on, and it is worth stating
plainly before the part that does differ.

Where the equivalence stops: the prior
--------------------------------------
The chains match. The *initializations* conventionally used with them do not.
At the top of the ladder each sampler starts from an assumed prior:

    VE:  ``N(0, sigma_max^2 I)``          true marginal ``N(0, (v + sigma_max^2) I)``
    VP:  ``N(0, I)``                      true marginal ``N(0, (abar v + 1 - abar) I)``

for data of per-coordinate variance ``v``. Push the VE prior through (2) and it
becomes ``N(0, (1 - abar_max) I)``, which is *not* ``N(0, I)``. The two
conventions differ, and the difference is exactly the thing the coordinate
change cannot remove:

- VE's prior is wrong by a variance factor ``1 + v / sigma_max^2`` for every
  finite ``sigma_max``, for any data at all.
- VP's prior is wrong by ``1 + abar_max (v - 1)`` -- which is **exactly right
  when v = 1**, at every ``sigma_max``, including a very small one.

That is the real content of "variance preserving": not a better process, but a
convention that is exact on standardized data, and only there. How much that is
worth is a separate question from whether it is true, and it is a measurement:
the two priors differ by exactly ``1.0`` of variance, which reaches the sampled
output multiplied by the chain's total contraction -- 0.010 of the target
variance at ``sigma_max = 3``, 0.0003 at ``sigma_max = 8``, at every ladder
length. Over the same ladders the discretization deficit runs from 0.66 to 0.07.
So the exactly-right prior is worth one to two orders of magnitude less than one
more halving of the step size. ``experiments/vp_ve.py`` runs both axes, on
standardized and unstandardized data.

References
----------
Ho, Jain & Abbeel (2020), Denoising diffusion probabilistic models (VP, Eq. 7
and the ancestral update Eq. 11).
Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole (2021), Score-based
generative modeling through SDEs (the VE/VP naming, and both as SDE limits).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from diffusion.langevin import NoisedScoreFn

Tensor = torch.Tensor

__all__ = [
    "alphabar_from_sigma",
    "sigma_from_alphabar",
    "to_vp",
    "to_ve",
    "vp_score_fn",
    "ddpm_ancestral_vp",
    "prior_variance_error",
]


def _abar(sigma: float) -> float:
    """``abar`` for one level, in Python double.

    Separate from :func:`alphabar_from_sigma` and used everywhere internally.
    The public function returns a tensor and therefore a *dtype*, and routing a
    scalar through a float32 tensor rounds ``abar`` to ~1e-7 -- which is
    invisible in a sample mean and fatal to the pathwise equivalence test, whose
    whole point is agreement at float64 precision.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    return 1.0 / (1.0 + float(sigma) * float(sigma))


def _as_float_tensor(x: float | Sequence[float] | Tensor) -> Tensor:
    """As a floating tensor, keeping the caller's dtype if it brought one.

    ``torch.as_tensor(x, dtype=torch.get_default_dtype())`` would silently
    downcast a float64 input to float32, which turns the exact round trip of (1)
    into a 1e-7 one -- a real bug in a module whose central claim is an equality.
    """
    t = torch.as_tensor(x)
    return t if t.is_floating_point() else t.to(torch.get_default_dtype())


def alphabar_from_sigma(sigma: float | Sequence[float] | Tensor) -> Tensor:
    """``abar = 1 / (1 + sigma^2)``, the SNR match of (1).

    Examples
    --------
    >>> import torch
    >>> alphabar_from_sigma([0.0, 1.0, 3.0])
    tensor([1.0000, 0.5000, 0.1000])
    """
    s = _as_float_tensor(sigma)
    if torch.any(s < 0):
        raise ValueError("sigma must be non-negative")
    return 1.0 / (1.0 + s * s)


def sigma_from_alphabar(alphabar: float | Sequence[float] | Tensor) -> Tensor:
    """``sigma = sqrt((1 - abar) / abar)``, the inverse of (1).

    Examples
    --------
    >>> sigma_from_alphabar([1.0, 0.5, 0.1])
    tensor([0., 1., 3.])
    """
    a = _as_float_tensor(alphabar)
    if torch.any(a <= 0) or torch.any(a > 1):
        raise ValueError("alphabar must lie in (0, 1]")
    return torch.sqrt((1.0 - a) / a)


def to_vp(x: Tensor, sigma: float) -> Tensor:
    """VE state -> VP state at the matched level: multiply by ``sqrt(abar)``."""
    return x * math.sqrt(_abar(sigma))


def to_ve(xt: Tensor, sigma: float) -> Tensor:
    """VP state -> VE state at the matched level. Inverse of :func:`to_vp`."""
    return xt / math.sqrt(_abar(sigma))


def vp_score_fn(score_fn: NoisedScoreFn) -> NoisedScoreFn:
    """Wrap a VE score ``s(x, sigma)`` into the VP score at the matched level.

    Equation (3): ``s_VP(xt, sigma) = s_VE(xt / sqrt(abar), sigma) / sqrt(abar)``.
    Both the argument and the value are rescaled, and forgetting either one is
    the mistake this function exists to make impossible -- rescaling only the
    input leaves a score that still integrates to a density, just the wrong one,
    so nothing downstream would fail loudly.

    Takes and returns a *callable*, so the analytic score of a mixture target and
    a trained network wrap identically -- the same reason
    :func:`diffusion.samplers.ddpm_ancestral` takes a callable.
    """

    def wrapped(xt: Tensor, sigma: float) -> Tensor:
        root = math.sqrt(_abar(sigma))
        return score_fn(xt / root, sigma) / root

    return wrapped


def ddpm_ancestral_vp(
    score_fn: NoisedScoreFn,
    xt0: Tensor,
    sigmas: Sequence[float],
    generator: torch.Generator | None = None,
    denoise_final: bool = True,
    trajectory: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """DDPM ancestral sampling in VP coordinates: the update (4), literally.

    Written in terms of ``abar`` and ``beta`` rather than by rescaling the VE
    sampler, precisely so that the equivalence is a *result* the tests check and
    not an artifact of sharing an implementation. The two functions have no code
    in common and agree pathwise to float precision
    (``tests/test_vp.py::test_the_two_samplers_are_the_same_chain_pathwise``).

    Parameters
    ----------
    score_fn:
        ``s_VP(xt, sigma) -> (n, d)``, the score of the VP marginal at the level
        matched to ``sigma`` by (1). Wrap a VE score with :func:`vp_score_fn`.
        Indexing by ``sigma`` rather than by ``abar`` keeps one clock across the
        whole repo; (1) is a bijection so nothing is lost.
    xt0:
        Initial VP states ``(n, d)``. The convention is ``N(0, I)``, which is the
        exact top-of-ladder marginal when the data have unit variance and is
        wrong otherwise -- see :func:`prior_variance_error`.
    sigmas:
        Strictly decreasing VE levels, coarse to fine; converted internally.
    denoise_final:
        Append the exact ``abar -> 1`` step ``xt <- (xt + (1 - abar) s)``, which
        is Tweedie's ``E[x_0 | xt]`` and carries no noise.

    Returns the final states ``(n, d)`` **in data coordinates** (``abar = 1``, so
    VP and VE coincide there and the return is directly comparable to
    :func:`diffusion.samplers.ddpm_ancestral`), or ``(final, trajectory)``.
    """
    lv = [float(s) for s in sigmas]
    if len(lv) < 2:
        raise ValueError("need at least two noise levels")
    if any(a <= b for a, b in zip(lv[:-1], lv[1:])):
        raise ValueError("sigmas must be strictly decreasing")
    if lv[-1] <= 0:
        raise ValueError("the finest level must be positive")
    if denoise_final:
        lv = lv + [0.0]

    xt = torch.as_tensor(xt0).clone()
    if xt.ndim != 2:
        raise ValueError(f"xt0 must be (n, d), got {tuple(xt.shape)}")
    states = [xt.clone()] if trajectory else None

    abars = [_abar(s) for s in lv]
    for i in range(len(lv) - 1):
        sigma, abar, abar_next = lv[i], abars[i], abars[i + 1]
        alpha = abar / abar_next                       # abar_i / abar_{i-1}
        beta = 1.0 - alpha
        xt = (xt + beta * score_fn(xt, sigma)) / math.sqrt(alpha)
        if lv[i + 1] > 0.0:
            beta_tilde = beta * (1.0 - abar_next) / (1.0 - abar)
            xt = xt + math.sqrt(beta_tilde) * torch.randn(
                xt.shape, dtype=xt.dtype, generator=generator
            )
        if states is not None:
            states.append(xt.clone())
        if not torch.isfinite(xt).all():
            raise FloatingPointError(f"non-finite state at sigma={sigma:g}")

    if states is not None:
        return xt, torch.stack(states)
    return xt


def prior_variance_error(sigma_max: float, data_var: float) -> tuple[float, float]:
    """How wrong each convention's starting prior is, as a variance ratio.

    Returns ``(ve_ratio, vp_ratio)``, each the assumed prior variance divided by
    the true top-of-ladder marginal variance, per coordinate. A value of 1.0 is
    an exact prior; the sampler's prior-mismatch error is monotone in the
    distance from it.

        VE: ``sigma_max^2 / (v + sigma_max^2)``     -- below 1 for every finite
            ``sigma_max``, for any data.
        VP: ``1 / (abar v + 1 - abar)``             -- exactly 1 when ``v = 1``,
            at every ``sigma_max``.

    This is the *only* place the two conventions differ (the chains themselves
    are the same chain), and it is not a rounding detail: at ``sigma_max = 3``
    on unit-variance data VE starts 10% narrow and VP starts exact.

    Examples
    --------
    >>> ve, vp = prior_variance_error(sigma_max=3.0, data_var=1.0)
    >>> f"{ve:.4f} {vp:.4f}"
    '0.9000 1.0000'
    >>> ve, vp = prior_variance_error(sigma_max=3.0, data_var=4.0)
    >>> f"{ve:.4f} {vp:.4f}"    # unstandardized: both wrong, VP by more
    '0.6923 0.7692'
    """
    if sigma_max <= 0:
        raise ValueError("sigma_max must be positive")
    if data_var <= 0:
        raise ValueError("data_var must be positive")
    abar = _abar(sigma_max)
    ve = sigma_max**2 / (data_var + sigma_max**2)
    vp = 1.0 / (abar * data_var + 1.0 - abar)
    return ve, vp
