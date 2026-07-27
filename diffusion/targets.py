"""Target densities whose score is known in **closed form**.

The score is the only object this repo cares about:

    s(x) = grad_x log p(x).

Everything downstream -- Langevin sampling, denoising score matching, the
reverse-time SDE, the probability-flow ODE -- is written in terms of it, and a
diffusion model is in the end nothing but an estimate of ``s``. That is exactly
why the targets here are chosen the way they are: for every density in this
module the score is available *analytically*, so a learned score can be scored
against the truth rather than against a plausible-looking picture. No image
dataset can offer that; a 2D Gaussian mixture can.

Two families are provided.

``GaussianMixture``
    ``p(x) = sum_k w_k N(x; mu_k, Sigma_k)``. Its score is a
    responsibility-weighted average of the per-component scores (derived
    below), it has exact moments, an exact sampler, and -- the property the
    whole repo is built on -- **it stays a Gaussian mixture when noised**, so
    the score of the noised marginal ``p_sigma`` is closed-form at every noise
    level (see :meth:`GaussianMixture.noised`). Denoising score matching
    (Day 24 of the build plan) therefore has an analytic target everywhere,
    not just at ``sigma = 0``.

``TwoMoons``
    The classic interleaved-crescents benchmark, but built as a genuine
    density rather than as a sampling recipe. The usual `sklearn`-style two
    moons is defined by *how you draw from it* ("uniform angle plus Gaussian
    noise"), which leaves ``log p`` and hence the score undefined. Writing
    each crescent in polar coordinates (:class:`Crescent`) gives a normalized
    density with a hand-derivable score, an exact normalizer, and a rejection
    sampler -- so it can play the same "truth is known" role as the mixture
    while being curved, non-Gaussian, and impossible to noise in closed form
    (which is the honest counterpart: the mixture is the validation target,
    the moons are the harder one).

Relation to mcmc-from-scratch
-----------------------------
The mixture and its responsibility-weighted score are the same construction as
``mcmc.targets.GaussianMixture`` in the sibling repo mcmc-from-scratch (NumPy
there, torch here). That is deliberate: the two repos meet at the score
function. There, ``grad log pi`` drives MALA and HMC and the samplers are
*exact* because a Metropolis accept repairs the discretization; here the same
object is what a network will learn, and the accept step is gone. See
:mod:`diffusion.langevin` for that bridge spelled out.

Conventions
-----------
Points are ``(n, d)`` tensors; ``log_prob`` returns ``(n,)`` and ``score``
returns ``(n, d)``. Each target owns a dtype (default ``float32``, matching
the networks that come later) and inputs are cast to it, so a ``float64``
target used in a finite-difference test computes in ``float64`` throughout.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, Sequence

import torch

Tensor = torch.Tensor

__all__ = [
    "Target",
    "Gaussian",
    "Mixture",
    "GaussianMixture",
    "Crescent",
    "TwoMoons",
    "finite_difference_score",
    "bessel_i0",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def bessel_i0(kappa: float) -> float:
    """Modified Bessel function of the first kind, order 0, by its series.

    ``I_0(k) = (1/2pi) int_{-pi}^{pi} exp(k cos u) du = sum_m (k^2/4)^m / (m!)^2``.

    The series is written out rather than imported because it is the angular
    normalizer of :class:`Crescent` and the repo's rule is that constants it
    depends on are derived, not looked up. It converges superexponentially
    (term ratio ``(k^2/4)/m^2``), so a handful of terms suffice for the
    moderate concentrations used here; it is checked against
    ``torch.special.i0`` as an independent oracle in the tests.

    Raises for ``kappa > 50``: there the terms peak around ``m ~ k/2`` at a
    magnitude that costs float64 digits, and the asymptotic form should be
    used instead. This repo never needs it, so it refuses rather than
    silently returning something less accurate than it looks.
    """
    if kappa < 0.0:
        raise ValueError(f"kappa must be non-negative, got {kappa}")
    if kappa > 50.0:
        raise ValueError(
            f"kappa={kappa} is outside the series' accurate range (<= 50); "
            "use an asymptotic expansion"
        )
    q = 0.25 * kappa * kappa
    term = 1.0
    total = 1.0
    for m in range(1, 200):
        term *= q / (m * m)
        total += term
        if term < 1e-18 * total:
            break
    return total


def _std_normal_cdf(z: float) -> float:
    """Phi(z), via erf. Used once, for the crescent's radial normalizer."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def finite_difference_score(
    log_prob: Callable[[Tensor], Tensor], x: Tensor, h: float = 1e-4
) -> Tensor:
    """Central-difference estimate of ``grad log p(x)`` -- the test oracle.

    Every analytic score in this module is checked against this. Central
    differences are ``O(h^2)`` accurate, so with ``x`` in ``float64`` and
    ``h = 1e-4`` the truncation floor is ~1e-8 while the roundoff floor is
    ~1e-12, comfortably below the tolerances the tests assert. In ``float32``
    the roundoff floor alone is ~1e-3, which is why the target under test is
    always constructed in ``float64``.
    """
    x = x.detach()
    grad = torch.empty_like(x)
    for j in range(x.shape[1]):
        step = torch.zeros_like(x)
        step[:, j] = h
        grad[:, j] = (log_prob(x + step) - log_prob(x - step)) / (2.0 * h)
    return grad


def _as_points(x: Tensor, dim: int, dtype: torch.dtype) -> Tensor:
    x = torch.as_tensor(x, dtype=dtype)
    if x.ndim != 2 or x.shape[1] != dim:
        raise ValueError(f"expected points of shape (n, {dim}), got {tuple(x.shape)}")
    return x


# --------------------------------------------------------------------------
# base class
# --------------------------------------------------------------------------
class Target(ABC):
    """A density with an analytic score.

    Subclasses implement :meth:`log_prob` (normalized -- the constant is not
    needed for the score, but a normalized density makes the finite-difference
    check and any likelihood comparison meaningful), :meth:`score`, and
    :meth:`sample`.
    """

    dim: int
    dtype: torch.dtype

    @abstractmethod
    def log_prob(self, x: Tensor) -> Tensor:
        """Normalized ``log p(x)`` for points ``(n, d)`` -> ``(n,)``."""

    @abstractmethod
    def score(self, x: Tensor) -> Tensor:
        """``grad_x log p(x)`` for points ``(n, d)`` -> ``(n, d)``."""

    @abstractmethod
    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        """``n`` exact draws -> ``(n, d)``."""

    def score_error(self, x: Tensor, h: float = 1e-4) -> Tensor:
        """Max abs deviation of :meth:`score` from finite differences of
        :meth:`log_prob`. The tests assert on this; it is exposed because it
        is also the honest way to sanity-check a target one has just written.
        """
        x = _as_points(x, self.dim, self.dtype)
        return (self.score(x) - finite_difference_score(self.log_prob, x, h)).abs().max()


# --------------------------------------------------------------------------
# Gaussian and mixtures
# --------------------------------------------------------------------------
class Gaussian(Target):
    """``N(mu, Sigma)`` with a full covariance.

    ``log N(x; mu, Sigma) = -1/2 [ d log 2pi + log det Sigma + (x-mu)' P (x-mu) ]``
    with ``P = Sigma^{-1}``, and differentiating the quadratic form gives the
    score directly:

        ``grad_x log N = -P (x - mu)``,

    i.e. a linear pull toward the mean along the *precision*-weighted
    direction. Everything is computed from one Cholesky factor
    ``Sigma = L L'``: ``log det Sigma = 2 sum_i log L_ii`` and the quadratic
    form is ``|| L^{-1} (x-mu) ||^2`` by a triangular solve, which is both
    cheaper and better conditioned than forming the inverse.
    """

    def __init__(
        self,
        mean: Sequence[float] | Tensor,
        cov: Sequence[Sequence[float]] | Tensor,
        dtype: torch.dtype = torch.float32,
    ):
        self.dtype = dtype
        self.mean = torch.as_tensor(mean, dtype=dtype).reshape(-1)
        self.dim = int(self.mean.shape[0])
        cov = torch.as_tensor(cov, dtype=dtype)
        if cov.ndim == 0:  # scalar -> isotropic
            cov = cov * torch.eye(self.dim, dtype=dtype)
        elif cov.ndim == 1:  # diagonal
            cov = torch.diag(cov)
        if cov.shape != (self.dim, self.dim):
            raise ValueError(f"cov must be ({self.dim}, {self.dim}), got {tuple(cov.shape)}")
        self.cov = 0.5 * (cov + cov.T)  # symmetrize; a caller's typo becomes a Cholesky error
        self._chol = torch.linalg.cholesky(self.cov)  # raises if not PSD -- deliberate
        self._log_norm = -0.5 * (
            self.dim * math.log(2.0 * math.pi)
            + 2.0 * torch.log(torch.diagonal(self._chol)).sum()
        )

    def log_prob(self, x: Tensor) -> Tensor:
        x = _as_points(x, self.dim, self.dtype)
        d = (x - self.mean).unsqueeze(-1)  # (n, d, 1)
        w = torch.linalg.solve_triangular(self._chol, d, upper=False)  # L^{-1}(x-mu)
        return self._log_norm - 0.5 * (w.squeeze(-1) ** 2).sum(dim=1)

    def score(self, x: Tensor) -> Tensor:
        x = _as_points(x, self.dim, self.dtype)
        d = (x - self.mean).unsqueeze(-1)
        return -torch.cholesky_solve(d, self._chol).squeeze(-1)

    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        z = torch.randn(n, self.dim, dtype=self.dtype, generator=generator)
        return self.mean + z @ self._chol.T

    def noised(self, sigma: float) -> "Gaussian":
        """``N(mu, Sigma) * N(0, sigma^2 I) = N(mu, Sigma + sigma^2 I)``.

        Convolving two Gaussians adds their covariances -- the sum of
        independent Gaussians is Gaussian. This is what makes the *mixture*
        closed under noising too.
        """
        eye = torch.eye(self.dim, dtype=self.dtype)
        return Gaussian(self.mean, self.cov + sigma * sigma * eye, dtype=self.dtype)


class Mixture(Target):
    """``p(x) = sum_k w_k p_k(x)`` over any components with analytic scores.

    **The score of a mixture is a responsibility-weighted average of the
    component scores.** Writing ``r_k(x) = w_k p_k(x) / p(x)``,

        grad log p = (grad p) / p
                   = sum_k w_k (grad p_k) / p
                   = sum_k [w_k p_k / p] (grad p_k / p_k)
                   = sum_k r_k(x) grad log p_k(x).

    Two things are worth noticing, because they are what make this numerically
    safe. First, ``r_k`` is a *softmax* of ``log w_k + log p_k(x)``, so the
    shared normalizer of ``p`` cancels and nothing ever exponentiates a raw
    density. Second, the weights are a genuine convex combination summing to
    1, so in the far tail where one component dominates the mixture score
    reduces exactly to that component's -- which is the behaviour a sampler in
    a distant mode needs and the reason a naive ``grad log(sum ...)``
    implemented by hand tends to underflow where this does not.
    """

    def __init__(self, components: Sequence[Target], weights: Sequence[float] | Tensor | None = None):
        if len(components) == 0:
            raise ValueError("a mixture needs at least one component")
        dims = {c.dim for c in components}
        if len(dims) != 1:
            raise ValueError(f"components disagree on dimension: {sorted(dims)}")
        dtypes = {c.dtype for c in components}
        if len(dtypes) != 1:
            raise ValueError(f"components disagree on dtype: {dtypes}")
        self.components = list(components)
        self.dim = components[0].dim
        self.dtype = components[0].dtype
        if weights is None:
            w = torch.full((len(components),), 1.0 / len(components), dtype=self.dtype)
        else:
            w = torch.as_tensor(weights, dtype=self.dtype).reshape(-1)
            if w.shape[0] != len(components):
                raise ValueError("weights and components have different lengths")
            if (w <= 0).any():
                raise ValueError("mixture weights must be positive")
            w = w / w.sum()
        self.weights = w
        self._log_w = torch.log(w)

    @property
    def n_components(self) -> int:
        return len(self.components)

    def component_log_probs(self, x: Tensor) -> Tensor:
        """``log w_k + log p_k(x)`` -> ``(n, K)``."""
        x = _as_points(x, self.dim, self.dtype)
        return torch.stack([c.log_prob(x) for c in self.components], dim=1) + self._log_w

    def responsibilities(self, x: Tensor) -> Tensor:
        """``r_k(x)`` -> ``(n, K)``, rows summing to 1 (a softmax; see class doc)."""
        return torch.softmax(self.component_log_probs(x), dim=1)

    def log_prob(self, x: Tensor) -> Tensor:
        return torch.logsumexp(self.component_log_probs(x), dim=1)

    def score(self, x: Tensor) -> Tensor:
        x = _as_points(x, self.dim, self.dtype)
        r = self.responsibilities(x)  # (n, K)
        s = torch.stack([c.score(x) for c in self.components], dim=1)  # (n, K, d)
        return (r.unsqueeze(-1) * s).sum(dim=1)

    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        """Ancestral: pick a component from ``w``, then draw from it.

        Draws are generated one component-block at a time (vectorized) but
        scattered back into the *positions* that drew that label, so the
        returned rows are i.i.d. in order rather than sorted by component --
        sorted draws quietly break anything that looks at a prefix.
        """
        idx = torch.multinomial(self.weights.to(torch.float64), n, replacement=True, generator=generator)
        out = torch.empty(n, self.dim, dtype=self.dtype)
        for k, comp in enumerate(self.components):
            mask = idx == k
            n_k = int(mask.sum())
            if n_k:
                out[mask] = comp.sample(n_k, generator=generator)
        return out


class GaussianMixture(Mixture):
    """A mixture of :class:`Gaussian` components -- the repo's ground truth.

    Beyond what :class:`Mixture` provides, the Gaussian case has exact
    moments and, crucially, is **closed under noising**:

        ``p_sigma := p * N(0, sigma^2 I) = sum_k w_k N(mu_k, Sigma_k + sigma^2 I)``,

    because convolution distributes over the sum and each term is a
    Gaussian-Gaussian convolution. So for *every* noise level of a forward
    diffusion the marginal is another mixture from this same class, its score
    is the same responsibility-weighted formula, and a learned score has an
    analytic target at every ``sigma``. That single fact is the reason this
    repo can report score error instead of eyeballing samples.

    Exact moments, from the law of total expectation/variance over the
    component label ``k``:

        ``E[x] = sum_k w_k mu_k``
        ``Cov[x] = sum_k w_k (Sigma_k + mu_k mu_k') - E[x] E[x]'``

    -- the second term is the *between*-component spread, which is why a
    well-separated mixture has a covariance far larger than any component's.
    """

    def __init__(
        self,
        means: Tensor | Sequence[Sequence[float]],
        covs: Tensor | Sequence,
        weights: Sequence[float] | Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        means_t = torch.as_tensor(means, dtype=dtype)
        if means_t.ndim != 2:
            raise ValueError(f"means must be (K, d), got {tuple(means_t.shape)}")
        k, d = means_t.shape
        covs_t = torch.as_tensor(covs, dtype=dtype)
        if covs_t.ndim == 0:  # one shared isotropic scale
            covs_t = covs_t * torch.eye(d, dtype=dtype).expand(k, d, d)
        elif covs_t.ndim == 2 and covs_t.shape == (d, d):  # one shared full cov
            covs_t = covs_t.expand(k, d, d)
        if covs_t.shape != (k, d, d):
            raise ValueError(f"covs must be ({k}, {d}, {d}), got {tuple(covs_t.shape)}")
        super().__init__(
            [Gaussian(means_t[i], covs_t[i], dtype=dtype) for i in range(k)], weights
        )
        self.means = means_t
        self.covs = covs_t

    @property
    def mean(self) -> Tensor:
        return (self.weights.unsqueeze(1) * self.means).sum(dim=0)

    @property
    def cov(self) -> Tensor:
        m = self.mean
        second = (
            self.weights.reshape(-1, 1, 1)
            * (self.covs + self.means.unsqueeze(2) * self.means.unsqueeze(1))
        ).sum(dim=0)
        return second - torch.outer(m, m)

    def noised(self, sigma: float) -> "GaussianMixture":
        """The ``sigma``-noised marginal, still a Gaussian mixture (see class doc)."""
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        eye = torch.eye(self.dim, dtype=self.dtype)
        return GaussianMixture(
            self.means, self.covs + sigma * sigma * eye, self.weights, dtype=self.dtype
        )

    # -- standard instances ------------------------------------------------
    @classmethod
    def ring(
        cls,
        n_components: int = 8,
        radius: float = 4.0,
        std: float = 0.4,
        dtype: torch.dtype = torch.float32,
    ) -> "GaussianMixture":
        """``n`` equal-weight isotropic Gaussians evenly spaced on a circle.

        The default multi-modal benchmark: the modes are far enough apart
        (nearest-neighbour gap ``2 R sin(pi/n) = 3.06`` at the defaults, i.e.
        ~7.7 component standard deviations) that a single Langevin chain at a
        sensible step size does not cross between them in any reasonable
        budget -- which is exactly the failure that annealing repairs, and
        :mod:`diffusion.langevin`'s tests measure.
        """
        ang = torch.arange(n_components, dtype=torch.float64) * (2 * math.pi / n_components)
        means = radius * torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
        covs = (std**2) * torch.eye(2, dtype=torch.float64).expand(n_components, 2, 2)
        return cls(means.to(dtype), covs.to(dtype), dtype=dtype)

    @classmethod
    def bimodal(
        cls,
        separation: float = 6.0,
        std: float = 0.5,
        weights: Sequence[float] = (0.5, 0.5),
        dtype: torch.dtype = torch.float32,
    ) -> "GaussianMixture":
        """Two isotropic modes ``separation`` apart along ``x0``.

        The smallest target on which unadjusted Langevin visibly fails: the
        barrier between the modes is ``(separation/2 / std)^2 / 2`` nats deep,
        36 at the defaults, so crossing is a ``e^-36`` event per attempt.
        """
        means = torch.tensor([[-separation / 2, 0.0], [separation / 2, 0.0]], dtype=torch.float64)
        covs = (std**2) * torch.eye(2, dtype=torch.float64).expand(2, 2, 2)
        return cls(means.to(dtype), covs.to(dtype), weights, dtype=dtype)


# --------------------------------------------------------------------------
# the two-moons target, as an actual density
# --------------------------------------------------------------------------
class Crescent(Target):
    r"""One arc of the two-moons target: a normalized density in polar form.

    With ``y = x - c``, ``r = |y|``, ``theta = atan2(y_2, y_1)``,

        ``p(x) = (1/Z) exp( -(r - R)^2 / (2 s^2) + kappa cos(theta - theta_0) )``,

    a Gaussian ring of radius ``R`` and width ``s`` modulated by a von Mises
    factor of concentration ``kappa`` that concentrates mass around the
    direction ``theta_0`` -- a ring becomes a crescent.

    **Score.** Since ``grad r = u_r = y/r`` and ``grad theta = u_theta / r``
    with ``u_theta = (-y_2, y_1)/r``, the chain rule on the exponent gives

        ``grad log p = -(r - R)/s^2 * u_r  -  (kappa sin(theta - theta_0) / r) * u_theta``,

    a radial term pulling onto the ring plus a tangential term sliding along
    it toward ``theta_0``. Both are elementary and both are checked against
    finite differences of ``log p``.

    **Normalizer.** The exponent separates in polar coordinates, so with the
    area element ``r dr dtheta`` the double integral factorizes:

        ``Z = [ int_0^inf r e^{-(r-R)^2/(2 s^2)} dr ] * [ int_{-pi}^{pi} e^{kappa cos u} du ]``
          ``= [ s^2 e^{-R^2/(2 s^2)} + R s sqrt(2 pi) Phi(R/s) ] * [ 2 pi I_0(kappa) ]``.

    The radial factor is the substitution ``u = r - R`` split into
    ``int u e^{-u^2/2s^2} du`` (an exact antiderivative, evaluated at the
    ``r = 0`` cut) plus ``R int e^{-u^2/2s^2} du`` (a Gaussian tail, hence
    ``Phi``); the angular factor is the definition of ``I_0``
    (:func:`bessel_i0`). The tests check ``Z`` against 2D numerical
    quadrature, which is the point of computing it at all: a normalized
    density makes the finite-difference score check and any later likelihood
    comparison meaningful.

    **The one singularity.** ``theta`` is undefined at ``y = 0``, so the score
    blows up like ``1/r`` there. This is a coordinate artifact, not a modeling
    problem: at the defaults the density at the centre is ``e^{-R^2/2s^2}``,
    about ``1e-10`` of the peak, so no sampler goes there -- but a
    finite-difference test evaluated at the centre would be meaningless, and
    the tests keep away from it deliberately rather than by luck.
    """

    def __init__(
        self,
        radius: float = 1.0,
        width: float = 0.15,
        kappa: float = 1.0,
        angle: float = math.pi / 2,
        center: Sequence[float] = (0.0, 0.0),
        dtype: torch.dtype = torch.float32,
    ):
        if radius <= 0 or width <= 0:
            raise ValueError("radius and width must be positive")
        self.dim = 2
        self.dtype = dtype
        self.radius = float(radius)
        self.width = float(width)
        self.kappa = float(kappa)
        self.angle = float(angle)
        self.center = torch.as_tensor(center, dtype=dtype).reshape(2)

        r, s = self.radius, self.width
        radial = s * s * math.exp(-0.5 * (r / s) ** 2) + r * s * math.sqrt(2 * math.pi) * _std_normal_cdf(r / s)
        angular = 2.0 * math.pi * bessel_i0(self.kappa)
        self.log_norm = -math.log(radial * angular)

    def _polar(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        y = _as_points(x, 2, self.dtype) - self.center
        r = torch.linalg.norm(y, dim=1)
        theta = torch.atan2(y[:, 1], y[:, 0])
        return y, r, theta

    def log_prob(self, x: Tensor) -> Tensor:
        _, r, theta = self._polar(x)
        return (
            -((r - self.radius) ** 2) / (2.0 * self.width**2)
            + self.kappa * torch.cos(theta - self.angle)
            + self.log_norm
        )

    def score(self, x: Tensor) -> Tensor:
        y, r, theta = self._polar(x)
        r_safe = r.clamp_min(torch.finfo(self.dtype).tiny)
        u_r = y / r_safe.unsqueeze(1)
        u_theta = torch.stack([-y[:, 1], y[:, 0]], dim=1) / r_safe.unsqueeze(1)
        radial = -(r - self.radius) / (self.width**2)
        tangential = -self.kappa * torch.sin(theta - self.angle) / r_safe
        return radial.unsqueeze(1) * u_r + tangential.unsqueeze(1) * u_theta

    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        """Exact draws by rejection, one dimension at a time.

        The density factorizes in polar coordinates, so radius and angle are
        independent and can be drawn separately.

        *Angle*: ``q(theta) ~ e^{kappa cos(theta - theta_0)}``. Propose
        uniformly on the circle and accept with probability
        ``e^{kappa (cos(theta - theta_0) - 1)} <= 1``; the acceptance rate is
        ``I_0(kappa) e^{-kappa}`` (47% at ``kappa = 1``).

        *Radius*: ``q(r) ~ r e^{-(r-R)^2/(2 s^2)}`` on ``r > 0``. Propose
        ``r ~ N(R, s^2)`` restricted to ``r > 0`` and accept with probability
        ``r / M``, ``M = R + 10 s``, the extra factor ``r`` being exactly the
        polar area element. Proposals beyond ``M`` would need an acceptance
        probability above 1; they are a ``10-sigma`` event (``~1e-23``), and
        the ratio is clamped rather than pretending otherwise. That is the
        one approximation in this class and it is far below float64 noise.
        """
        r_out = self._sample_radius(n, generator)
        th_out = self._sample_angle(n, generator)
        pts = torch.stack([r_out * torch.cos(th_out), r_out * torch.sin(th_out)], dim=1)
        return (pts + self.center).to(self.dtype)

    def _sample_angle(self, n: int, generator: torch.Generator | None) -> Tensor:
        out = torch.empty(0, dtype=torch.float64)
        while out.numel() < n:
            m = max(4 * (n - out.numel()), 64)
            prop = (torch.rand(m, dtype=torch.float64, generator=generator) * 2 - 1) * math.pi
            u = torch.rand(m, dtype=torch.float64, generator=generator)
            keep = u < torch.exp(self.kappa * (torch.cos(prop) - 1.0))
            out = torch.cat([out, prop[keep]])
        return out[:n] + self.angle

    def _sample_radius(self, n: int, generator: torch.Generator | None) -> Tensor:
        bound = self.radius + 10.0 * self.width
        out = torch.empty(0, dtype=torch.float64)
        while out.numel() < n:
            m = max(2 * (n - out.numel()), 64)
            prop = self.radius + self.width * torch.randn(m, dtype=torch.float64, generator=generator)
            u = torch.rand(m, dtype=torch.float64, generator=generator)
            keep = (prop > 0) & (u < (prop / bound).clamp(max=1.0))
            out = torch.cat([out, prop[keep]])
        return out[:n]


class TwoMoons(Mixture):
    """Two interleaved crescents -- the curved, non-Gaussian target.

    Equal-weight mixture of two :class:`Crescent` components: one opening
    upward about the origin, one opening downward about ``(R, offset)``, so
    each arc nests into the other's concavity. This is the familiar two-moons
    picture, but every point of it has a density and a score, which the usual
    sampling-recipe version does not.

    Because both components share ``(R, s, kappa)`` and differ only by a
    rotation of ``theta_0`` and a translation -- both of which leave the polar
    integral invariant -- they share a normalizer exactly, so the mixture's
    responsibilities are a softmax of two *identically scaled* exponents and
    no constant needs tracking.

    Unlike :class:`GaussianMixture` this target is **not** closed under
    noising: the ``sigma``-noised crescent is a convolution with no elementary
    form. That is the honest division of labour in this repo -- the mixture is
    where a learned score is validated against truth at every noise level, and
    the moons are where a method is shown to work on something whose noised
    marginals nobody can write down.
    """

    def __init__(
        self,
        radius: float = 1.0,
        width: float = 0.15,
        kappa: float = 1.0,
        offset: float | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        off = 0.5 * radius if offset is None else float(offset)
        upper = Crescent(radius, width, kappa, angle=math.pi / 2, center=(0.0, 0.0), dtype=dtype)
        lower = Crescent(radius, width, kappa, angle=-math.pi / 2, center=(radius, off), dtype=dtype)
        super().__init__([upper, lower])
        self.radius = float(radius)
        self.width = float(width)
        self.kappa = float(kappa)
        self.offset = off
