r"""Classifier-free guidance, with the distribution it actually samples from.

The README's limitations have said since v0.1 that "classifier-free guidance,
and conditional generation on the mixture component, are the obvious next thing
a target with labeled components makes measurable." This is that -- and the
reason it is worth doing *here* rather than on images is that every quantity in
the argument is available in closed form, so the question "what does guidance
sample from?" has a checkable answer instead of a visual impression.

The construction
----------------
The target is a :class:`~diffusion.targets.GaussianMixture`, whose component
index is a genuine label ``y = k``. Both conditionals are exact at every noise
level, because the mixture is closed under noising:

    p_sigma(x | y=k) = N(mu_k, Sigma_k + sigma^2 I),
    p_sigma(x)       = sum_j w_j N(mu_j, Sigma_j + sigma^2 I).

Classifier-free guidance (Ho & Salimans 2022) forms, at each noise level,

    s_w(x, sigma) = (1 + w) grad log p_sigma(x | y=k) - w grad log p_sigma(x)

and runs a sampler on that. With ``w = 0`` it is plain conditional sampling;
``w > 0`` pushes mass toward where the conditional beats the marginal. Nothing
here is learned: both terms are analytic, so what is measured below is the
*method*, with no network error mixed in.

What it samples from, and what it does not
------------------------------------------
``s_w(., sigma)`` is exactly the score of

    r_{w,sigma}(x)  ∝  p_sigma(x | y=k)^{1+w} p_sigma(x)^{-w},

a tilted density -- :func:`tilted_log_density` writes it down, and the tests
check the identity by finite differences rather than by inspection. At
``sigma = 0`` this is the density guidance is usually *said* to target,

    q_w(x)  ∝  p(x | y=k)^{1+w} p(x)^{-w}.

The catch, and the thing the experiment measures: **the family
``r_{w,sigma}`` is not the noising path of ``q_w``.** Tilting and convolution
do not commute, so ``r_{w,sigma} != q_w * N(0, sigma^2 I)`` for ``w != 0``. That
matters differently for the two samplers this repo already has:

- Annealed Langevin *equilibrates* at each level, so what it returns is
  governed by the final level's stationary law, ``r_{w,sigma_min}`` -- which is
  ``q_w`` up to the smallest noise on the ladder. It should land on ``q_w``.
- Ancestral / reverse-SDE sampling makes one pass down the ladder and is only
  exact when the scores it is handed are the scores of a genuine noising path.
  Guidance breaks that premise, so it should *not* land on ``q_w``.

That is the same lesson this repo already recorded twice -- an unadjusted
sampler converges to the stationary law of the score it is handed, and a
one-pass sampler inherits the path -- pointed at guidance.
``experiments/guidance.py`` measures both against exact draws from ``q_w``,
with the two-independent-draws floor that says when a gap is real.

Normalizability
---------------
``q_w`` is a proper density only when the tilt's quadratic terms cancel. For
components sharing one isotropic covariance ``s^2 I``, the leading behaviour
along any ray is ``-(1+w)|x-mu_k|^2/(2s^2) + w|x-mu_j|^2/(2s^2)``, whose
``|x|^2`` coefficient is ``(-(1+w) + w)/(2s^2) = -1/(2s^2) < 0`` for every
``w >= 0``: the growing pieces cancel and the density decays.
:func:`check_tiltable` enforces the shared-isotropic-covariance condition
rather than leaving it to be discovered as a divergent normalizer, and the
ring target satisfies it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from .targets import GaussianMixture

__all__ = [
    "check_tiltable",
    "conditional_score",
    "guided_score",
    "guided_score_fn",
    "tilted_log_density",
    "TiltedGrid",
    "component_purity",
    "spread_about",
    "radial_offset",
    "guided_stationary_point",
]


# --------------------------------------------------------------------------
# scores
# --------------------------------------------------------------------------
def check_tiltable(mixture: GaussianMixture, tol: float = 1e-6) -> float:
    """Raise unless every component shares one isotropic covariance.

    Returns the shared variance. See the module docstring: this is exactly the
    condition under which the tilted density's quadratic terms cancel and
    ``q_w`` is normalizable for every ``w >= 0``. Checked up front because the
    failure mode otherwise is a grid normalizer that quietly depends on the box
    -- a number that looks fine and means nothing.
    """
    covs = mixture.covs.to(torch.float64)
    d = mixture.dim
    var = float(covs[0, 0, 0])
    eye = torch.eye(d, dtype=torch.float64)
    if torch.max(torch.abs(covs - var * eye)) > tol * max(var, 1.0):
        raise ValueError(
            "classifier-free guidance is only normalizable here when every "
            "component shares one isotropic covariance (see diffusion.guidance "
            "module docstring); this mixture's covariances differ"
        )
    return var


def conditional_score(mixture: GaussianMixture, k: int, x: Tensor, sigma: float) -> Tensor:
    """``grad log p_sigma(x | y = k)`` -- exact, at any noise level.

    The sigma-noised k-th component is ``N(mu_k, Sigma_k + sigma^2 I)``, which
    is what :meth:`GaussianMixture.noised` produces component-wise, so this is
    the same closed form the unconditional score already uses and not a second
    implementation of it.
    """
    if not 0 <= k < mixture.n_components:
        raise IndexError(f"component {k} out of range for {mixture.n_components}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    return mixture.noised(sigma).components[k].score(x)


def guided_score(
    mixture: GaussianMixture, k: int, x: Tensor, sigma: float, w: float
) -> Tensor:
    """The classifier-free guidance score ``(1+w) s(x|k) - w s(x)`` at ``sigma``.

    ``w = 0`` returns the conditional score exactly (the ``-w`` term vanishes),
    and ``w = -1`` returns the unconditional one; both are asserted in the
    tests, because they are the two places an off-by-one in the combination
    would be visible and nowhere else would it be.
    """
    noised = mixture.noised(sigma)
    cond = noised.components[k].score(x)
    if w == 0.0:
        return cond
    return (1.0 + w) * cond - w * noised.score(x)


def guided_score_fn(
    mixture: GaussianMixture, k: int, w: float
) -> Callable[[Tensor, float], Tensor]:
    """``s(x, sigma)`` in the form every sampler in this repo already takes.

    Returning a closure is what lets ``ddpm_ancestral`` and
    ``annealed_langevin`` run on guidance without knowing anything about it --
    the same interface the exact-score and learned-score arms use, so the
    comparison changes one thing.

    The noised mixture is memoised on ``sigma``. It is a pure function of it,
    so this changes no number; it matters because annealed Langevin calls the
    score hundreds of times at the *same* level, and rebuilding K Gaussians
    (each with its own factorisation) per call made the sampler's cost the
    construction rather than the arithmetic.
    """
    cache: dict[float, GaussianMixture] = {}

    def noised(sigma: float) -> GaussianMixture:
        m = cache.get(sigma)
        if m is None:
            m = cache[sigma] = mixture.noised(sigma)
        return m

    def fn(x: Tensor, sigma: float) -> Tensor:
        m = noised(float(sigma))
        cond = m.components[k].score(x)
        if w == 0.0:
            return cond
        return (1.0 + w) * cond - w * m.score(x)

    return fn


# --------------------------------------------------------------------------
# the density guidance is said to target
# --------------------------------------------------------------------------
def tilted_log_density(
    mixture: GaussianMixture, k: int, x: Tensor, w: float, sigma: float = 0.0
) -> Tensor:
    """``(1+w) log p_sigma(x|y=k) - w log p_sigma(x)``, up to an additive constant.

    Unnormalized: the normalizer depends on ``w`` and has no closed form for a
    mixture, which is what :class:`TiltedGrid` exists to supply numerically.
    The gradient of this is :func:`guided_score` by construction, and the tests
    check that by finite differences -- the one identity everything downstream
    rests on.
    """
    noised = mixture.noised(sigma)
    out = (1.0 + w) * noised.components[k].log_prob(x)
    if w != 0.0:
        out = out - w * noised.log_prob(x)
    return out


@dataclass
class TiltedGrid:
    """``q_w`` normalized on a grid, with exact draws from that normalization.

    Two dimensions and a density that decays make a grid the honest tool here:
    the normalizer is a sum, the density is available anywhere, and sampling is
    a multinomial over cells plus a uniform jitter inside the chosen cell. What
    that produces is exact draws from the *piecewise-constant* approximation to
    ``q_w``, not from ``q_w`` -- which is why the experiment reports the sliced
    distance between two independent such draws as its resolution floor, and
    why :meth:`refine_check` exists to show the answer is not moving with the
    grid.
    """

    log_density: Tensor       # (G, G) unnormalized log q_w on the cell centres
    xs: Tensor                # (G,) cell centres along axis 0
    ys: Tensor                # (G,) cell centres along axis 1
    cell: float               # cell side length
    log_norm: float           # log of the grid quadrature of exp(log_density)

    @classmethod
    def build(
        cls,
        mixture: GaussianMixture,
        k: int,
        w: float,
        half_width: float = 8.0,
        resolution: int = 512,
    ) -> "TiltedGrid":
        check_tiltable(mixture)
        if mixture.dim != 2:
            raise ValueError(f"TiltedGrid is 2-D; got dim {mixture.dim}")
        edges = torch.linspace(-half_width, half_width, resolution + 1, dtype=torch.float64)
        centres = 0.5 * (edges[:-1] + edges[1:])
        cell = float(edges[1] - edges[0])
        gx, gy = torch.meshgrid(centres, centres, indexing="ij")
        pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).to(mixture.dtype)
        logq = tilted_log_density(mixture, k, pts, w).to(torch.float64).reshape(
            resolution, resolution
        )
        log_norm = float(torch.logsumexp(logq.reshape(-1), dim=0)) + 2 * math.log(cell)
        return cls(logq, centres, centres, cell, log_norm)

    def tail_mass(self) -> float:
        """Fraction of the grid's mass sitting in its outermost ring of cells.

        A box that clips the density would give a normalizer -- and every
        distance measured against it -- that depends on ``half_width``. This is
        the number that says it does not, and the experiment prints it.
        """
        p = torch.softmax(self.log_density.reshape(-1), dim=0).reshape(
            self.log_density.shape
        )
        edge = p.sum() - p[1:-1, 1:-1].sum()
        return float(edge)

    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        """``n`` draws: multinomial over cells, then uniform inside the cell."""
        p = torch.softmax(self.log_density.reshape(-1), dim=0)
        idx = torch.multinomial(p, n, replacement=True, generator=generator)
        g = self.log_density.shape[0]
        ix, iy = idx // g, idx % g
        jitter = (
            torch.rand(n, 2, dtype=torch.float64, generator=generator) - 0.5
        ) * self.cell
        pts = torch.stack([self.xs[ix], self.ys[iy]], dim=1) + jitter
        return pts.to(torch.float32)

    def mean(self) -> Tensor:
        p = torch.softmax(self.log_density.reshape(-1), dim=0).reshape(
            self.log_density.shape
        )
        return torch.stack([(p.sum(dim=1) * self.xs).sum(), (p.sum(dim=0) * self.ys).sum()])


# --------------------------------------------------------------------------
# a read-out with a meaning
# --------------------------------------------------------------------------
def component_purity(mixture: GaussianMixture, k: int, x: Tensor) -> float:
    """Fraction of ``x`` whose nearest component mean is ``k``.

    The cheap read-out of "did guidance do what it is for". Nearest mean rather
    than highest responsibility because the two agree for equal-weight
    equal-covariance components and the former needs no density evaluation --
    and because at large ``w`` the point of interest is geometric.

    On a well-separated target this read-out **saturates**: conditioning alone
    already puts essentially every draw in the right component, so purity is 1
    at every ``w`` and cannot distinguish anything. That is a fact about the
    target rather than about guidance, and it is why :func:`spread_about` is
    measured alongside it.
    """
    d2 = torch.cdist(x.to(torch.float64), mixture.means.to(torch.float64)) ** 2
    return float((d2.argmin(dim=1) == k).to(torch.float64).mean())


def radial_offset(mixture: GaussianMixture, k: int, x: Tensor) -> float:
    """Mean displacement of ``x`` from ``mu_k`` along the outward radial direction.

    The read-out that resolves what guidance does to a one-pass sampler on this
    target, where purity saturates and the width barely moves.
    :func:`guided_stationary_point` is the reason to look here rather than
    anywhere else: the guided score's attractor sits *outside* ``mu_k``, on the
    ray through it, so the displacement worth measuring is the radial one.

    Positive means further from the origin than ``mu_k``.
    """
    mu = mixture.means[k].to(torch.float64)
    u = mu / mu.norm()
    return float(((x.to(torch.float64) - mu) @ u).mean())


def guided_stationary_point(mixture: GaussianMixture, k: int, w: float, sigma: float,
                            ambient_var: float | None = None) -> Tensor:
    r"""Where the guided drift at level ``sigma`` points, in closed form.

    Approximating the noised marginal by one isotropic Gaussian at the origin
    with variance ``u_sigma`` -- exact in the limits and a good description of a
    ring, whose components are arranged symmetrically about it -- the guided
    score is affine,

        s_w(x) = -x [ (1+w)/v_sigma - w/u_sigma ] + (1+w) mu_k / v_sigma,

    with ``v_sigma = s^2 + sigma^2`` the conditional's variance. Its root is

        x* = (1+w) mu_k / [ (1+w) - w rho_sigma ],   rho_sigma = v_sigma/u_sigma,

    and the two limits are the whole story:

    - ``sigma -> infinity``: ``rho -> 1`` and ``x* -> (1 + w) mu_k``. The chain
      is aimed at a point ``(1+w)`` times too far out.
    - ``sigma -> 0``: ``rho -> s^2/(s^2 + R^2/2)``, tiny for a well-separated
      ring, and ``x* -> mu_k``. The attractor comes home.

    So a sampler that equilibrates at every level ends at ``mu_k`` and one that
    makes a single pass carries part of the early overshoot to the end. That is
    the mechanism ``experiments/guidance.py`` measures, and the reason it
    measures a *radial* displacement rather than a width or a purity.

    ``ambient_var`` defaults to the mixture's own mean per-coordinate variance
    plus ``sigma^2``, which for the ring is ``s^2 + R^2/2 + sigma^2``.
    """
    var = check_tiltable(mixture)
    v = var + sigma ** 2
    if ambient_var is None:
        ambient_var = float(torch.diagonal(mixture.cov.to(torch.float64)).mean()) + sigma ** 2
    rho = v / ambient_var
    denom = (1.0 + w) - w * rho
    if denom <= 0:
        raise ValueError(
            f"guided drift is expansive at sigma={sigma}, w={w}: the marginal "
            "term outweighs the conditional and the affine map has no stable root"
        )
    return (1.0 + w) / denom * mixture.means[k].to(torch.float64)


def spread_about(mixture: GaussianMixture, k: int, x: Tensor) -> float:
    """Per-coordinate rms distance of ``x`` from ``mu_k``.

    ``sqrt(mean_i ||x_i - mu_k||^2 / d)``, so for draws from the untilted
    component it returns that component's own standard deviation and the number
    is directly comparable to it. This is the read-out that resolves where
    :func:`component_purity` saturates: guidance's practical signature is a loss
    of diversity, and diversity here is exactly this width.
    """
    diff = x.to(torch.float64) - mixture.means[k].to(torch.float64)
    return float(torch.sqrt((diff ** 2).sum(dim=1).mean() / mixture.dim))
