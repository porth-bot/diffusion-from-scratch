r"""Sample-quality metrics, from scratch, with closed-form ground truth.

Every other claim in this repo is checked against the exact score. Samples
are the one place where that instrument does not reach: a sampler's output is
a *distribution*, and the question "is this the right distribution?" needs a
two-sample statistic. This module implements two, plus the exact values both
take on Gaussians, so that the statistics themselves are validated rather than
trusted.

Neither one is scipy's or a package's. Both are a few lines once the idea is
right, and writing them out is what makes it possible to say what they do and
do not detect.

Maximum mean discrepancy
------------------------
For a kernel ``k`` with feature map ``phi``, ``MMD(P, Q) = || E_P phi(x) -
E_Q phi(y) ||`` in the kernel's Hilbert space. Expanding the square gives
something computable from samples alone,

    ``MMD^2 = E k(x, x') + E k(y, y') - 2 E k(x, y)``,

with ``x, x'`` independent draws from ``P`` and ``y, y'`` from ``Q``. With a
Gaussian RBF kernel this is a *characteristic* kernel: ``MMD = 0`` if and only
if ``P = Q``, so the statistic can in principle see any difference at all --
which is exactly why it is the right tool for "did the sampler get the whole
distribution", where a mean and a covariance are not.

:func:`rbf_mmd2` uses the **unbiased** estimator, dropping the diagonal terms:

    ``MMD^2_u = 1/(n(n-1)) sum_{i != j} k(x_i, x_j)``
             ``+ 1/(m(m-1)) sum_{i != j} k(y_i, y_j)``
             ``- 2/(nm) sum_{i,j} k(x_i, y_j)``.

Unbiased matters here for a specific reason: the biased (V-statistic) version
keeps ``k(x_i, x_i) = 1`` on the diagonal and is therefore positive even when
``P = Q``, by an amount ``~1/n`` that would be read as "the sampler is a bit
off" when it is nothing but the estimator. The unbiased estimator has mean
exactly 0 under the null and goes *negative* about half the time, which looks
alarming and is correct.

A number alone does not say whether it is large. :func:`mmd_permutation_test`
supplies the reference distribution the honest way: pool the two samples,
re-split at random many times, and see where the observed value falls. Under
the null the labels are exchangeable, so the permutation distribution is the
null distribution -- no asymptotics, no assumption about the kernel or the
data.

Sliced Wasserstein
------------------
MMD lives in a kernel space and is dimensionless. The sliced Wasserstein
distance is a *transport* distance and comes back in the units of the data,
which makes it the one to quote when the question is "how far off, in the
space the samples live in".

The 1-dimensional Wasserstein distance is elementary -- for equal sample
sizes, sort both and average ``|x_(i) - y_(i)|^p``, because the optimal
coupling of two 1-D empirical measures is the monotone one. In higher
dimensions it is not elementary at all, so the sliced version (Rabin et al.
2011) reduces to the case that is:

    ``SW_p(P, Q)^p = E_{u ~ Unif(S^{d-1})} W_p(u.P, u.Q)^p``,

the average over random directions of the 1-D distance between the projected
measures. It is a genuine metric (zero iff the measures agree, since equal
projections in every direction determine the measure by Cramer-Wold), costs
``O(K n log n)``, and its Monte Carlo error over the directions is reported by
:func:`sliced_wasserstein` rather than left implicit.

Validation
----------
Both estimators have a closed form on Gaussians, and both are checked against
it in the tests rather than against each other:

- :func:`gaussian_rbf_mmd2` -- because an RBF kernel is a Gaussian density up
  to a constant, every term of ``MMD^2`` is a Gaussian convolution.
- :func:`gaussian_sliced_wasserstein2` -- because every 1-D projection of an
  isotropic Gaussian is a 1-D Gaussian, and ``W_2`` between two of those is
  the distance between their means and standard deviations in quadrature.

References: Gretton et al. 2012 (MMD, and the permutation test); Rabin et al.
2011 and Bonneel et al. 2015 (sliced Wasserstein).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

Tensor = torch.Tensor

__all__ = [
    "median_heuristic_bandwidth",
    "rbf_kernel_matrix",
    "rbf_mmd2",
    "MMDTest",
    "mmd_permutation_test",
    "SlicedWasserstein",
    "sliced_wasserstein",
    "gaussian_rbf_mmd2",
    "gaussian_sliced_wasserstein2",
]


def _as_samples(x: Tensor, name: str) -> Tensor:
    x = torch.as_tensor(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be (n, d), got {tuple(x.shape)}")
    if x.shape[0] < 2:
        raise ValueError(f"{name} needs at least 2 points, got {x.shape[0]}")
    return x.to(torch.float64)


def median_heuristic_bandwidth(
    x: Tensor,
    y: Tensor | None = None,
    max_points: int = 1000,
) -> float:
    """Bandwidth ``h`` = median pairwise Euclidean distance on the pooled sample.

    The kernel here is ``exp(-||x - y||^2 / (2 h^2))``, so ``h`` is a length in
    the data's own units and the median heuristic sets it to the scale at which
    points typically sit apart -- large enough that the kernel is not
    everywhere ~0 (which would make ``MMD^2`` a sum of near-zeros with no
    power), small enough that it is not everywhere ~1 (which would make it
    blind). It is a heuristic, not an optimum, and its known weakness is real:
    a kernel at the *typical* separation is least sensitive to differences at
    much finer scales. That is why the experiment also reports sliced
    Wasserstein, which has no bandwidth.

    Capped at ``max_points`` per sample because this is an ``O(n^2)`` median
    for a quantity that only needs to be right to within a factor; the cap is
    a deterministic head slice, so the bandwidth of a given pair of samples is
    reproducible.
    """
    x = _as_samples(x, "x")[:max_points]
    pool = x if y is None else torch.cat([x, _as_samples(y, "y")[:max_points]], dim=0)
    d = torch.cdist(pool, pool)
    iu = torch.triu_indices(d.shape[0], d.shape[0], offset=1)
    med = float(d[iu[0], iu[1]].median())
    if med <= 0:
        raise ValueError("median pairwise distance is zero: the pooled sample is a single point")
    return med


def rbf_kernel_matrix(a: Tensor, b: Tensor, bandwidth: float) -> Tensor:
    """``k(a_i, b_j) = exp(-||a_i - b_j||^2 / (2 h^2))`` -> ``(n, m)``."""
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    return torch.exp(-torch.cdist(a, b) ** 2 / (2.0 * bandwidth * bandwidth))


def _mmd2_from_blocks(kxx: Tensor, kyy: Tensor, kxy: Tensor, unbiased: bool) -> float:
    n, m = kxx.shape[0], kyy.shape[0]
    if unbiased:
        sxx = float(kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
        syy = float(kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
    else:
        sxx = float(kxx.sum()) / (n * n)
        syy = float(kyy.sum()) / (m * m)
    return sxx + syy - 2.0 * float(kxy.sum()) / (n * m)


def rbf_mmd2(
    x: Tensor,
    y: Tensor,
    bandwidth: float | None = None,
    unbiased: bool = True,
) -> float:
    """Squared MMD between two samples under a Gaussian RBF kernel.

    ``bandwidth`` defaults to the median heuristic on the pooled sample. Note
    that this makes the statistic depend on both samples through the kernel as
    well as through the estimator; for a *comparison across arms* the honest
    thing is to fix one bandwidth (from the reference sample) and pass it
    explicitly, which is what the experiment does.

    With ``unbiased=True`` (the default) the estimator has expectation exactly
    ``MMD^2``, hence exactly 0 under the null, and takes negative values about
    half the time when the two samples agree. A negative value is not an
    error; it is the estimator saying "no evidence of a difference".
    """
    x = _as_samples(x, "x")
    y = _as_samples(y, "y")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"samples differ in dimension: {x.shape[1]} vs {y.shape[1]}")
    h = median_heuristic_bandwidth(x, y) if bandwidth is None else float(bandwidth)
    return _mmd2_from_blocks(
        rbf_kernel_matrix(x, x, h), rbf_kernel_matrix(y, y, h), rbf_kernel_matrix(x, y, h), unbiased
    )


@dataclass
class MMDTest:
    """The result of :func:`mmd_permutation_test`."""

    mmd2: float
    p_value: float
    bandwidth: float
    n_permutations: int
    null_mean: float
    null_q95: float

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"MMD^2 = {self.mmd2:.3e}  p = {self.p_value:.4f}  "
            f"(null 95th pct {self.null_q95:.3e}, h = {self.bandwidth:.3f})"
        )


def mmd_permutation_test(
    x: Tensor,
    y: Tensor,
    n_permutations: int = 200,
    bandwidth: float | None = None,
    generator: torch.Generator | None = None,
) -> MMDTest:
    """Test ``P = Q`` by permuting the labels of the pooled sample.

    Under the null the two samples are exchangeable, so any re-split of the
    pooled points is as likely as the observed one; the MMD values over random
    re-splits are therefore draws from the null distribution of the statistic.
    This needs no asymptotic theory and no assumption about the kernel, which
    matters because the asymptotic null of ``MMD^2_u`` is an infinite weighted
    sum of chi-squares with the kernel's own eigenvalues as weights -- correct,
    and useless without them.

    The p-value is ``(1 + #{permuted >= observed}) / (1 + n_permutations)``.
    The ``+1``s are not a fudge: including the observed split among the
    permutations is what makes the test exactly valid (the p-value is
    stochastically dominated by ``Uniform(0,1)`` under the null) instead of
    able to return 0. Its floor is therefore ``1/(1 + n_permutations)``, so a
    reported ``p = 0.005`` at 199 permutations means "smaller than this test
    can resolve", not "0.005".

    The kernel matrix over the pooled sample is computed **once** and each
    permutation only re-indexes it, so the cost is one ``(n+m)^2`` kernel plus
    cheap gathers rather than ``n_permutations`` kernels.
    """
    x = _as_samples(x, "x")
    y = _as_samples(y, "y")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"samples differ in dimension: {x.shape[1]} vs {y.shape[1]}")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")

    n, m = x.shape[0], y.shape[0]
    h = median_heuristic_bandwidth(x, y) if bandwidth is None else float(bandwidth)
    pool = torch.cat([x, y], dim=0)
    k = rbf_kernel_matrix(pool, pool, h)

    def stat(idx: Tensor) -> float:
        a, b = idx[:n], idx[n:]
        return _mmd2_from_blocks(
            k[a][:, a], k[b][:, b], k[a][:, b], unbiased=True
        )

    observed = stat(torch.arange(n + m))
    null = torch.empty(n_permutations, dtype=torch.float64)
    for i in range(n_permutations):
        null[i] = stat(torch.randperm(n + m, generator=generator))
    p = float((1.0 + float((null >= observed).sum())) / (1.0 + n_permutations))
    return MMDTest(
        mmd2=observed,
        p_value=p,
        bandwidth=h,
        n_permutations=n_permutations,
        null_mean=float(null.mean()),
        null_q95=float(null.quantile(0.95)),
    )


@dataclass
class SlicedWasserstein:
    """The result of :func:`sliced_wasserstein`.

    ``stderr`` is the Monte Carlo error over the random directions only -- the
    sampling error of the two point clouds themselves is a separate thing, and
    the experiment measures that by running the statistic between two
    independent draws from the same reference distribution.
    """

    distance: float
    stderr: float
    n_projections: int
    p: int


def sliced_wasserstein(
    x: Tensor,
    y: Tensor,
    n_projections: int = 256,
    p: int = 2,
    generator: torch.Generator | None = None,
) -> SlicedWasserstein:
    """Sliced ``W_p`` between two samples: average over random 1-D projections.

    Directions are drawn uniformly on the sphere by normalizing standard
    normals (the normal is rotationally invariant, so its direction is). For
    each direction both samples are projected and sorted; when the two sample
    sizes are equal the sorted difference *is* the exact 1-D optimal transport
    cost, since the monotone coupling is optimal for any convex cost. When
    they differ, both empirical quantile functions are evaluated on a common
    grid of ``max(n, m)`` midpoint quantile levels -- an approximation, and
    named as one, though it is exact in the limit and equal-size samples are
    what the experiment uses.

    Returned as a :class:`SlicedWasserstein` carrying the Monte Carlo standard
    error over directions, because a sliced distance quoted without it invites
    reading a difference between two arms that is really projection noise.
    """
    x = _as_samples(x, "x")
    y = _as_samples(y, "y")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"samples differ in dimension: {x.shape[1]} vs {y.shape[1]}")
    if n_projections < 1:
        raise ValueError(f"n_projections must be >= 1, got {n_projections}")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")

    d = x.shape[1]
    u = torch.randn(d, n_projections, dtype=torch.float64, generator=generator)
    u = u / u.norm(dim=0, keepdim=True)
    px = (x @ u).sort(dim=0).values  # (n, K)
    py = (y @ u).sort(dim=0).values  # (m, K)

    if px.shape[0] != py.shape[0]:
        n_grid = max(px.shape[0], py.shape[0])
        levels = (torch.arange(n_grid, dtype=torch.float64) + 0.5) / n_grid
        px = _quantiles(px, levels)
        py = _quantiles(py, levels)

    per_projection = (px - py).abs().pow(p).mean(dim=0)  # W_p^p per direction
    mean = float(per_projection.mean())
    # sd of the mean of W_p^p over directions, pushed through the p-th root by
    # the delta method: d(t^{1/p})/dt = t^{1/p - 1}/p.
    se_pth = float(per_projection.std() / math.sqrt(n_projections))
    distance = mean ** (1.0 / p)
    stderr = se_pth * (mean ** (1.0 / p - 1.0)) / p if mean > 0 else 0.0
    return SlicedWasserstein(distance, stderr, n_projections, p)


def _quantiles(sorted_values: Tensor, levels: Tensor) -> Tensor:
    """Linear-interpolated empirical quantile function of each column."""
    n = sorted_values.shape[0]
    pos = (levels * n - 0.5).clamp(0, n - 1)
    lo = pos.floor().long()
    hi = pos.ceil().long()
    w = (pos - lo.to(pos.dtype)).unsqueeze(1)
    return sorted_values[lo] * (1 - w) + sorted_values[hi] * w


# --------------------------------------------------------------------------
# closed forms -- the oracles the estimators are checked against
# --------------------------------------------------------------------------
def gaussian_rbf_mmd2(
    mean_a: Tensor,
    std_a: float,
    mean_b: Tensor,
    std_b: float,
    bandwidth: float,
) -> float:
    r"""Exact ``MMD^2`` between two isotropic Gaussians under the RBF kernel.

    The kernel is a Gaussian density up to a constant,

        ``k(x, y) = exp(-|x-y|^2 / 2h^2) = (2 pi h^2)^{d/2} N(x - y; 0, h^2 I)``,

    so each of the three terms is an expectation of a Gaussian density under
    an independent Gaussian difference. If ``X ~ N(a, s_a^2 I)`` and
    ``Y ~ N(b, s_b^2 I)`` independently then ``X - Y ~ N(a - b, (s_a^2 +
    s_b^2) I)``, and the convolution of two Gaussians evaluated at 0 gives

        ``E k(X, Y) = (h^2 / (h^2 + s_a^2 + s_b^2))^{d/2}``
                    ``exp( -|a - b|^2 / (2 (h^2 + s_a^2 + s_b^2)) )``.

    The two within-sample terms are the same formula with ``a = b`` and both
    standard deviations equal, and ``MMD^2 = E k(X,X') + E k(Y,Y') - 2 E k(X,Y)``.

    Every quantity a test needs, in one line each -- which is the point: the
    estimator in this module is checked against this, not against another
    implementation of itself.
    """
    a = torch.as_tensor(mean_a, dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(mean_b, dtype=torch.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"means differ in dimension: {tuple(a.shape)} vs {tuple(b.shape)}")
    if std_a <= 0 or std_b <= 0 or bandwidth <= 0:
        raise ValueError("standard deviations and bandwidth must be positive")
    d = a.shape[0]
    h2 = bandwidth * bandwidth
    gap2 = float(((a - b) ** 2).sum())

    def term(v: float, sep2: float) -> float:
        return (h2 / (h2 + v)) ** (d / 2.0) * math.exp(-sep2 / (2.0 * (h2 + v)))

    return (
        term(2 * std_a * std_a, 0.0)
        + term(2 * std_b * std_b, 0.0)
        - 2.0 * term(std_a * std_a + std_b * std_b, gap2)
    )


def gaussian_sliced_wasserstein2(
    mean_a: Tensor, std_a: float, mean_b: Tensor, std_b: float
) -> float:
    r"""Exact sliced ``W_2`` between two isotropic Gaussians.

    A projection of ``N(a, s^2 I)`` onto a unit direction ``u`` is
    ``N(<u, a>, s^2)`` -- isotropy makes the projected variance the same in
    every direction -- and ``W_2^2`` between two 1-D Gaussians is
    ``(mean gap)^2 + (sd gap)^2``. Averaging over ``u`` uniform on the sphere
    needs only ``E[<u, delta>^2] = |delta|^2 / d`` (by symmetry the
    coordinates of ``u`` have equal second moment and ``|u|^2 = 1``), so

        ``SW_2^2 = |a - b|^2 / d + (s_a - s_b)^2``.

    Anisotropic covariances have no such elementary form, which is why the
    tests use isotropic ones: an oracle that needed its own numerics would not
    be an oracle.
    """
    a = torch.as_tensor(mean_a, dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(mean_b, dtype=torch.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"means differ in dimension: {tuple(a.shape)} vs {tuple(b.shape)}")
    if std_a <= 0 or std_b <= 0:
        raise ValueError("standard deviations must be positive")
    d = a.shape[0]
    gap2 = float(((a - b) ** 2).sum())
    return math.sqrt(gap2 / d + (std_a - std_b) ** 2)
