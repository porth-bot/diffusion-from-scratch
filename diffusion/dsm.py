r"""Denoising score matching: how the score gets learned from samples alone.

The problem this solves
-----------------------
We want ``s_theta(x, sigma) ~= grad_x log p_sigma(x)``, but ``p_sigma`` is
exactly what is unknown -- all that is available is a pile of draws from
``p_0``. The obvious objective is unusable as written:

    ``J_ESM(theta) = (1/2) E_{p_sigma(x)} || s_theta(x) - grad log p_sigma(x) ||^2``,   (ESM)

since it names ``grad log p_sigma`` on the right. (This repo can *evaluate*
that term for the Gaussian mixture -- which is the whole reason the mixture is
the validation target -- but a method that needs it is not a method.)

Hyvarinen (2005) removed it by integrating the cross term by parts, arriving
at ``E[ tr(grad s_theta) + (1/2)||s_theta||^2 ]``. That is exact and needs no
``p``, but it needs the *Jacobian trace* of the network, ``d`` extra backward
passes per step in ``d`` dimensions. Vincent (2011) removed it a second way
that costs nothing at all, and that is what this module implements.

Denoising score matching (Vincent 2011)
---------------------------------------
Take the noised variable to be built explicitly: draw ``x_0 ~ p_0``, then
``x = x_0 + sigma eps`` with ``eps ~ N(0, I)``, so that
``q_sigma(x | x_0) = N(x; x_0, sigma^2 I)`` and
``p_sigma(x) = int p_0(x_0) q_sigma(x|x_0) dx_0`` by construction. Define

    ``J_DSM(theta) = (1/2) E_{p_0(x_0) q_sigma(x|x_0)} || s_theta(x) - grad_x log q_sigma(x|x_0) ||^2``.   (DSM)

**Claim.** ``J_ESM(theta) = J_DSM(theta) + C`` with ``C`` independent of
``theta``. Hence the same minimizer, hence the same gradient.

*Proof.* Expand both squares. The ``||s_theta(x)||^2`` term has the same
expectation in both, since ``x`` is a draw from ``p_sigma`` either way, and
the remaining terms do not contain ``theta``. So only the cross terms need
matching. For ESM,

    ``E_{p_sigma} <s_theta(x), grad log p_sigma(x)>``
        ``= int p_sigma(x) <s_theta(x), grad p_sigma(x) / p_sigma(x)> dx``
        ``= int <s_theta(x), grad_x p_sigma(x)> dx``.

The density cancels -- that is the one step that matters. Now push the
gradient through the definition of ``p_sigma`` (differentiating under the
integral sign, legitimate here because ``q_sigma`` is Gaussian and smooth in
``x``):

    ``grad_x p_sigma(x) = int p_0(x_0) grad_x q_sigma(x|x_0) dx_0``,

so the cross term becomes

    ``int int p_0(x_0) <s_theta(x), grad_x q_sigma(x|x_0)> dx_0 dx``
        ``= int int p_0(x_0) q_sigma(x|x_0) <s_theta(x), grad_x log q_sigma(x|x_0)> dx_0 dx``,

which is the DSM cross term. The two objectives differ by a constant. QED.

The trick, in one sentence: multiplying and dividing by the density turns
"the score of a marginal we cannot write down" into "the score of a
conditional we chose ourselves".

What the target actually is
---------------------------
For the Gaussian ``q_sigma(x|x_0) = N(x_0, sigma^2 I)``,

    ``grad_x log q_sigma(x | x_0) = -(x - x_0)/sigma^2 = -eps/sigma``,

with ``eps`` the very noise that was drawn. So the regression target is the
*negated normalized noise*, known exactly at no cost, and DSM is literally
"predict the noise you added". The name is the interpretation: the optimal
``s_theta(x)`` is the average over all ``x_0`` that could have produced ``x``
of the direction pointing back to ``x_0``, i.e. it points toward where the
clean data probably was. Taking the conditional expectation inside gives
Tweedie's formula ``s_sigma(x) = (E[x_0|x] - x)/sigma^2``, the same identity
:mod:`diffusion.langevin` uses to bound the curvature and hence the step size.

Weighting, and why it collapses
-------------------------------
Training happens at all levels at once, so the per-level objectives must be
combined:

    ``L(theta) = E_{sigma ~ q(sigma)} lambda(sigma) J_DSM(theta; sigma)``.

The target scales as ``1/sigma``, so an unweighted sum is dominated by the
finest level by a factor of ``(sigma_max/sigma_min)^2`` -- four orders of
magnitude on the ladder used here, meaning the coarse levels, the ones that
carry the global structure annealing depends on, would contribute nothing.
The standard fix ``lambda(sigma) = sigma^2`` (Song & Ermon 2019) makes every
level contribute equally, since ``sigma^2 ||s_theta + eps/sigma||^2 =
||sigma s_theta + eps||^2`` is a unit-scale quantity at every ``sigma``.

Combined with the network's own ``s_theta = -eps_theta/sigma``
parameterization (:mod:`diffusion.model`), the whole thing collapses to

    ``L(theta) = E || eps_theta(x_0 + sigma eps, sigma) - eps ||^2``,

which is the form actually implemented here and the form DDPM trains. The two
"divide by sigma" decisions -- one in the architecture, one in the loss
weight -- are not redundant: the first fixes the *output* scale, the second
fixes the *gradient* scale across levels, and dropping either reintroduces
the imbalance.

``q(sigma)`` is log-uniform over the ladder, matching the geometric spacing of
:func:`diffusion.langevin.geometric_sigmas`: the levels the sampler will visit
are geometric, so uniform-in-log is uniform over the levels that matter.

Honest scope
------------
DSM is unbiased for the *noised* score at each ``sigma``, never for the score
of ``p_0`` itself; ``sigma -> 0`` recovers it only in the limit, and that is
also the limit where the ``1/sigma`` amplification of any network error is
worst. The experiment reports error against the exact noised score at every
level rather than a single headline number, because there is no single level
at which "the score" is being learned.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from diffusion.model import ScoreNet
from diffusion.targets import Target

Tensor = torch.Tensor

__all__ = [
    "sample_log_uniform_sigmas",
    "dsm_loss",
    "explicit_score_matching_loss",
    "score_diagnostics",
    "ScoreDiagnostics",
    "TrainHistory",
    "train_score_model",
]


def sample_log_uniform_sigmas(
    n: int,
    sigma_min: float,
    sigma_max: float,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``n`` draws with ``log sigma ~ Uniform(log sigma_min, log sigma_max)``.

    Log-uniform rather than uniform because the ladder the sampler walks is
    geometric: uniform-in-log spreads the training budget evenly over the
    levels that will actually be visited, while uniform-in-sigma would spend
    almost everything at the coarse end.
    """
    if not (sigma_max >= sigma_min > 0):
        raise ValueError(f"need sigma_max >= sigma_min > 0, got ({sigma_min}, {sigma_max})")
    u = torch.rand(n, generator=generator, dtype=dtype)
    lo, hi = math.log(sigma_min), math.log(sigma_max)
    return torch.exp(lo + (hi - lo) * u)


def dsm_loss(
    model: ScoreNet,
    x0: Tensor,
    sigmas: Tensor,
    generator: torch.Generator | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    """The ``sigma^2``-weighted DSM loss, in its collapsed form.

    ``L = mean || eps_theta(x_0 + sigma eps, sigma) - eps ||^2`` over the batch
    (summed over coordinates, averaged over points), which by the derivation
    in the module docstring equals
    ``mean sigma^2 || s_theta(x, sigma) - grad log q_sigma(x|x_0) ||^2``.

    ``noise`` may be supplied instead of drawn, which is what makes the loss a
    deterministic function of the parameters and therefore finite-difference
    checkable.
    """
    if x0.ndim != 2:
        raise ValueError(f"x0 must be (n, d), got {tuple(x0.shape)}")
    sigmas = torch.as_tensor(sigmas, dtype=x0.dtype)
    if sigmas.ndim != 1 or sigmas.shape[0] != x0.shape[0]:
        raise ValueError(
            f"sigmas must be shape ({x0.shape[0]},), got {tuple(sigmas.shape)}"
        )
    if noise is None:
        noise = torch.randn(x0.shape, dtype=x0.dtype, generator=generator)
    elif noise.shape != x0.shape:
        raise ValueError(f"noise must match x0's shape {tuple(x0.shape)}")
    x = x0 + sigmas.unsqueeze(1) * noise
    eps_hat = model.epsilon(x, sigmas)
    return ((eps_hat - noise) ** 2).sum(dim=1).mean()


def explicit_score_matching_loss(
    score_fn: Callable[[Tensor], Tensor], target: Target, x: Tensor, sigma: float
) -> Tensor:
    """``mean sigma^2 || s(x) - grad log p_sigma(x) ||^2`` -- the objective DSM stands in for.

    Only computable because ``target`` is closed under noising (a
    :class:`diffusion.targets.GaussianMixture`), so ``grad log p_sigma`` is
    analytic. It exists for one purpose: the test that ESM and DSM differ by a
    ``theta``-independent constant, which is the claim the whole method rests
    on. ``x`` must be a batch of draws from ``p_sigma`` for the number to mean
    what it says.

    Weighted by ``sigma^2`` to match :func:`dsm_loss`'s convention so the two
    are directly comparable.
    """
    noised = target.noised(sigma)  # type: ignore[attr-defined]
    exact = noised.score(x)
    return (sigma**2) * ((score_fn(x) - exact) ** 2).sum(dim=1).mean()


@dataclass
class ScoreDiagnostics:
    """Per-level agreement between a learned score and the exact one."""

    sigmas: list[float]
    mse: list[float]
    relative_mse: list[float]
    cosine: list[float]
    exact_norm: list[float]

    def as_rows(self) -> list[dict[str, float]]:
        return [
            {
                "sigma": s,
                "mse": m,
                "relative_mse": r,
                "cosine": c,
                "exact_norm": n,
            }
            for s, m, r, c, n in zip(
                self.sigmas, self.mse, self.relative_mse, self.cosine, self.exact_norm
            )
        ]


def score_diagnostics(
    model: ScoreNet,
    target: Target,
    sigmas: Sequence[float],
    n_points: int = 4096,
    generator: torch.Generator | None = None,
) -> ScoreDiagnostics:
    """Compare ``s_theta(., sigma)`` against the exact noised score, level by level.

    Evaluation points are drawn from ``p_sigma`` itself (exactly, via the
    noised mixture's own sampler) rather than from a fixed grid: the score is
    only asked to be right where the sampler will go, and a grid over a box
    would average in far-tail points where the exact score is enormous and
    nothing ever visits.

    Three numbers per level, because they fail differently:

    - ``mse``: ``mean ||s_theta - s||^2``, the raw magnitude. Falls with
      ``sigma`` mechanically, since the score itself does.
    - ``relative_mse``: that divided by ``mean ||s||^2``. Scale-free, so it is
      the one to compare across levels.
    - ``cosine``: mean cosine similarity, which isolates *direction*. A
      sampler follows the direction; a score that points correctly but is 20%
      too short still moves states to the right place, more slowly.
    """
    sigs, mses, rels, coss, norms = [], [], [], [], []
    for sigma in sigmas:
        noised = target.noised(float(sigma))  # type: ignore[attr-defined]
        x = noised.sample(n_points, generator=generator)
        exact = noised.score(x)
        with torch.no_grad():
            learned = model(x, float(sigma))
        diff2 = ((learned - exact) ** 2).sum(dim=1)
        exact2 = (exact**2).sum(dim=1)
        cos = torch.nn.functional.cosine_similarity(learned, exact, dim=1)
        sigs.append(float(sigma))
        mses.append(float(diff2.mean()))
        rels.append(float(diff2.mean() / exact2.mean()))
        coss.append(float(cos.mean()))
        norms.append(float(exact2.mean().sqrt()))
    return ScoreDiagnostics(sigs, mses, rels, coss, norms)


@dataclass
class TrainHistory:
    """Everything the figures and the honest tables are drawn from."""

    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    eval_steps: list[int] = field(default_factory=list)
    eval_relative_mse: list[float] = field(default_factory=list)
    eval_cosine: list[float] = field(default_factory=list)
    seconds: float = 0.0


def train_score_model(
    model: ScoreNet,
    target: Target,
    sigma_min: float,
    sigma_max: float,
    n_steps: int = 8000,
    batch_size: int = 512,
    lr: float = 2e-3,
    seed: int = 0,
    log_every: int = 100,
    eval_every: int = 0,
    eval_sigmas: Sequence[float] | None = None,
    eval_points: int = 2048,
    progress: bool = False,
) -> TrainHistory:
    """Fit ``model`` by DSM on fresh draws from ``target``. Adam, no schedule.

    Fresh draws every step rather than a fixed dataset: the targets here have
    exact samplers, so there is no reason to reuse points and no
    generalization gap to study. This isolates optimization error from
    finite-sample error, which is the point of using toys with known truth.

    With ``eval_every > 0`` the model is scored against the exact noised score
    on a *fixed* evaluation set (same seed each time, so the eval is a
    deterministic function of the parameters and its trend is not noise).
    """
    if eval_every and eval_sigmas is None:
        eval_sigmas = [sigma_min, math.sqrt(sigma_min * sigma_max), sigma_max]
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = TrainHistory()
    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        x0 = target.sample(batch_size, generator=gen)
        sigmas = sample_log_uniform_sigmas(batch_size, sigma_min, sigma_max, generator=gen)
        loss = dsm_loss(model, x0, sigmas, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            history.steps.append(step)
            history.loss.append(float(loss.detach()))
            if progress:
                print(f"  step {step:6d}  loss {float(loss.detach()):.4f}", flush=True)
        if eval_every and (step % eval_every == 0 or step == 1):
            assert eval_sigmas is not None
            diag = score_diagnostics(
                model,
                target,
                eval_sigmas,
                n_points=eval_points,
                generator=torch.Generator().manual_seed(12345),
            )
            history.eval_steps.append(step)
            history.eval_relative_mse.append(sum(diag.relative_mse) / len(diag.relative_mse))
            history.eval_cosine.append(sum(diag.cosine) / len(diag.cosine))
    history.seconds = time.perf_counter() - t0
    return history
