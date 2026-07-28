"""Denoising score matching: the identity it rests on, its gradient, and that it learns.

The load-bearing claim of the whole method is Vincent's: the objective that
can be computed (DSM) differs from the objective that is wanted (ESM) by a
constant in ``theta``. That is not a shape property or a sanity check, it is
the reason training on "predict the noise you added" produces the score of a
density nobody wrote down -- so it gets a real test against the exact noised
score, which this repo's Gaussian mixture makes available at every level.
"""

from __future__ import annotations

import math

import pytest
import torch

from diffusion.dsm import (
    dsm_loss,
    explicit_score_matching_loss,
    sample_log_uniform_sigmas,
    score_diagnostics,
    train_score_model,
)
from diffusion.model import ScoreNet
from diffusion.targets import GaussianMixture

DT = torch.float64


def _target(dtype=DT) -> GaussianMixture:
    return GaussianMixture.bimodal(separation=4.0, std=0.7, dtype=dtype)


def _net(width: int = 32, depth: int = 2, seed: int = 0, dtype=torch.float32) -> ScoreNet:
    torch.manual_seed(seed)
    net = ScoreNet(dim=2, width=width, depth=depth, embed_dim=8, cond_width=16)
    return net.to(dtype)


# --------------------------------------------------------------------------
# the sigma sampler
# --------------------------------------------------------------------------
def test_sigma_draws_are_log_uniform_over_the_requested_range():
    g = torch.Generator().manual_seed(0)
    s = sample_log_uniform_sigmas(200_000, 0.05, 5.0, generator=g, dtype=DT)
    assert float(s.min()) >= 0.05 and float(s.max()) <= 5.0
    ls = torch.log(s)
    lo, hi = math.log(0.05), math.log(5.0)
    assert float(ls.mean()) == pytest.approx(0.5 * (lo + hi), abs=0.01)
    assert float(ls.var()) == pytest.approx((hi - lo) ** 2 / 12.0, rel=0.02)


def test_sigma_sampler_rejects_an_impossible_range():
    with pytest.raises(ValueError):
        sample_log_uniform_sigmas(10, 0.0, 1.0)
    with pytest.raises(ValueError):
        sample_log_uniform_sigmas(10, 2.0, 1.0)


# --------------------------------------------------------------------------
# Vincent's identity -- the claim the method rests on
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [0.3, 1.0, 3.0])
def test_esm_and_dsm_differ_by_a_constant_that_does_not_depend_on_the_score(sigma):
    """``J_ESM(s) - J_DSM(s)`` is the same number for two very different ``s``.

    Estimated with *common random numbers*: the same ``(x_0, eps)`` draws
    build ``x``, and because ``x = x_0 + sigma eps`` is by construction a draw
    from ``p_sigma``, the very same points estimate both objectives. The
    difference of the two differences is then a pure Monte Carlo average of
    ``-2 <s_1 - s_2, s_exact + eps/sigma>``, whose expectation is zero by the
    derivation in :mod:`diffusion.dsm` -- so it shrinks like ``1/sqrt(n)``
    while each objective itself stays ``O(1)``.
    """
    target = _target()
    g = torch.Generator().manual_seed(7)
    n = 400_000
    x0 = target.sample(n, generator=g)
    eps = torch.randn(n, 2, dtype=DT, generator=g)
    x = x0 + sigma * eps
    dsm_target = -eps / sigma

    # Two deliberately different candidate scores, neither of them right.
    def s1(y: torch.Tensor) -> torch.Tensor:
        return -0.5 * y

    def s2(y: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.sin(y[:, 0]), 0.3 * y[:, 1] - 1.0], dim=1)

    gaps = []
    for s in (s1, s2):
        j_dsm = (sigma**2) * ((s(x) - dsm_target) ** 2).sum(dim=1).mean()
        j_esm = explicit_score_matching_loss(s, target, x, sigma)
        gaps.append(float(j_esm - j_dsm))

    # The two objectives themselves are far apart -- the constant is large,
    # which is exactly why "they differ by a constant" needs checking rather
    # than assuming.
    assert abs(gaps[0]) > 0.2
    # ...but the constant is the same for both scores, to Monte Carlo error.
    assert gaps[0] == pytest.approx(gaps[1], abs=0.02)


def test_the_exact_noised_score_is_where_dsm_is_minimized():
    """DSM's minimizer is the score, not merely something correlated with it.

    Perturbing away from the exact noised score in any direction raises the
    DSM objective -- checked here along a random direction and along the
    exact score itself, at a scale where Monte Carlo error cannot explain it.
    """
    target = _target()
    sigma = 0.8
    g = torch.Generator().manual_seed(11)
    n = 400_000
    x0 = target.sample(n, generator=g)
    eps = torch.randn(n, 2, dtype=DT, generator=g)
    x = x0 + sigma * eps
    dsm_target = -eps / sigma
    exact = target.noised(sigma).score(x)

    def j(s: torch.Tensor) -> float:
        return float((sigma**2) * ((s - dsm_target) ** 2).sum(dim=1).mean())

    base = j(exact)
    direction = torch.randn(n, 2, dtype=DT, generator=g)
    for delta in (-0.1, 0.1):
        assert j(exact + delta * direction) > base + 1e-3
        assert j((1.0 + delta) * exact) > base + 1e-3


# --------------------------------------------------------------------------
# the loss as implemented
# --------------------------------------------------------------------------
def test_collapsed_loss_equals_the_sigma_weighted_definition():
    """``||eps_hat - eps||^2`` really is ``sigma^2 ||s_theta - grad log q||^2``."""
    net = _net(dtype=DT)
    target = _target()
    g = torch.Generator().manual_seed(3)
    x0 = target.sample(256, generator=g)
    sigmas = sample_log_uniform_sigmas(256, 0.1, 3.0, generator=g, dtype=DT)
    noise = torch.randn(x0.shape, dtype=DT, generator=g)

    collapsed = float(dsm_loss(net, x0, sigmas, noise=noise).detach())

    x = x0 + sigmas.unsqueeze(1) * noise
    with torch.no_grad():
        s = net(x, sigmas)
    cond_score = -noise / sigmas.unsqueeze(1)
    longhand = float((sigmas**2 * ((s - cond_score) ** 2).sum(dim=1)).mean())
    assert collapsed == pytest.approx(longhand, rel=1e-9)


def test_dsm_loss_gradient_matches_finite_differences():
    """Autograd through the collapsed loss, checked against the definition.

    Supplying the noise makes the loss a deterministic function of the
    parameters, which is what makes a finite-difference check meaningful at
    all. Run in float64 with ``h = 1e-6``: the central-difference truncation
    floor is then ~1e-12 and the roundoff floor ~1e-10, both far below the
    tolerance asserted.
    """
    net = _net(width=16, depth=2, dtype=DT)
    target = _target()
    g = torch.Generator().manual_seed(5)
    x0 = target.sample(64, generator=g)
    sigmas = sample_log_uniform_sigmas(64, 0.1, 2.0, generator=g, dtype=DT)
    noise = torch.randn(x0.shape, dtype=DT, generator=g)

    loss = dsm_loss(net, x0, sigmas, noise=noise)
    net.zero_grad()
    loss.backward()

    h = 1e-6
    checked = 0
    for name, p in net.named_parameters():
        flat = p.data.reshape(-1)
        grad = p.grad.reshape(-1)
        # A handful of coordinates per tensor: the check is about correctness,
        # not coverage, and every one costs two full forward passes.
        for idx in (0, flat.numel() // 2, flat.numel() - 1):
            original = float(flat[idx])
            flat[idx] = original + h
            plus = float(dsm_loss(net, x0, sigmas, noise=noise).detach())
            flat[idx] = original - h
            minus = float(dsm_loss(net, x0, sigmas, noise=noise).detach())
            flat[idx] = original
            fd = (plus - minus) / (2 * h)
            assert float(grad[idx]) == pytest.approx(fd, abs=1e-6, rel=1e-5), name
            checked += 1
    assert checked > 20


def test_dsm_loss_rejects_mismatched_shapes():
    net = _net()
    x0 = torch.randn(8, 2)
    with pytest.raises(ValueError):
        dsm_loss(net, x0, torch.ones(7))
    with pytest.raises(ValueError):
        dsm_loss(net, x0, torch.ones(8), noise=torch.randn(7, 2))
    with pytest.raises(ValueError):
        dsm_loss(net, torch.randn(8), torch.ones(8))


# --------------------------------------------------------------------------
# diagnostics against the exact noised score
# --------------------------------------------------------------------------
def test_diagnostics_report_a_perfect_match_for_the_exact_score_itself():
    """The measuring instrument reads zero on the truth (and one on cosine)."""
    target = _target(dtype=torch.float32)

    class ExactScore:
        def __call__(self, x, sigma):
            return target.noised(float(sigma)).score(x)

    diag = score_diagnostics(
        ExactScore(), target, [0.1, 1.0, 4.0], n_points=2000,
        generator=torch.Generator().manual_seed(0),
    )
    assert max(diag.mse) < 1e-8
    assert min(diag.cosine) > 1.0 - 1e-6
    # The exact score's own magnitude grows as sigma shrinks -- the reason
    # raw MSE cannot be compared across levels and relative MSE is reported.
    assert diag.exact_norm[0] > 5 * diag.exact_norm[-1]


def test_diagnostics_penalize_a_score_that_points_the_wrong_way():
    target = _target(dtype=torch.float32)

    class Flipped:
        def __call__(self, x, sigma):
            return -target.noised(float(sigma)).score(x)

    diag = score_diagnostics(
        Flipped(), target, [1.0], n_points=2000,
        generator=torch.Generator().manual_seed(0),
    )
    assert diag.cosine[0] == pytest.approx(-1.0, abs=1e-5)
    assert diag.relative_mse[0] == pytest.approx(4.0, rel=1e-4)


# --------------------------------------------------------------------------
# it actually learns
# --------------------------------------------------------------------------
def test_training_drives_the_score_error_down_on_a_fixed_evaluation_set():
    """Short run, but the claim is the trend, and the eval set is fixed.

    The evaluation is a deterministic function of the parameters (fixed seed
    inside :func:`train_score_model`), so an improving curve is improvement
    and not resampling noise. "Monotonically-ish" is the honest bar for a
    stochastic objective: the requirement asserted is that the error falls a
    lot end to end and that late evaluations are better than early ones, not
    that every consecutive pair improves.
    """
    target = _target(dtype=torch.float32)
    net = _net(width=64, depth=3, seed=1, dtype=torch.float32)
    sigma_min, sigma_max = 0.1, 4.0

    before = score_diagnostics(
        net, target, [sigma_min, 1.0, sigma_max], n_points=2000,
        generator=torch.Generator().manual_seed(0),
    )
    hist = train_score_model(
        net, target, sigma_min, sigma_max,
        n_steps=1500, batch_size=256, lr=3e-3, seed=0,
        eval_every=250, eval_sigmas=[sigma_min, 1.0, sigma_max], eval_points=1000,
    )
    after = score_diagnostics(
        net, target, [sigma_min, 1.0, sigma_max], n_points=2000,
        generator=torch.Generator().manual_seed(0),
    )

    # Every level improves by a large factor, from an untrained net whose
    # relative error is ~1 (it predicts essentially nothing) downward.
    for b, a in zip(before.relative_mse, after.relative_mse):
        assert a < b / 3.0
    # Direction is recovered well away from the fine end (cosine ~0.99 at
    # sigma >= 1); the finest level lags in direction too, at ~0.79 after this
    # very short run, so the bar there is lower and deliberately so.
    assert min(after.cosine[1:]) > 0.95
    assert after.cosine[0] > 0.7

    # And the *finest* level is the worst-fit one, which is the prediction
    # made in diffusion.dsm's "honest scope": the 1/sigma parameterization
    # amplifies whatever error eps_theta has, so small sigma is where a
    # learned score is least accurate even though its own loss looks best.
    assert after.relative_mse[0] > 5 * max(after.relative_mse[1:])

    # The trajectory falls: second half beats first half, endpoint beats start.
    rel = hist.eval_relative_mse
    assert len(rel) >= 6
    assert rel[-1] < rel[0]
    half = len(rel) // 2
    assert sum(rel[half:]) / len(rel[half:]) < sum(rel[:half]) / len(rel[:half])
    assert hist.eval_cosine[-1] > hist.eval_cosine[0]
