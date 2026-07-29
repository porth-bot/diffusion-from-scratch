"""Tests for the two-sample metrics.

Both estimators are checked against a closed form rather than against each
other or against a reference implementation, which is the only way to catch a
wrong constant: an MMD with the bandwidth convention off by a factor of two,
or a sliced Wasserstein that returns ``W_p^p`` instead of ``W_p``, would look
perfectly reasonable in every plot and would rank the sampler arms in the same
order.

The other thing tested here is the *calibration* of the permutation test,
because a p-value that is not uniform under the null is not a p-value.
"""

from __future__ import annotations

import math

import pytest
import torch

from diffusion.metrics import (
    gaussian_rbf_mmd2,
    gaussian_sliced_wasserstein2,
    median_heuristic_bandwidth,
    mmd_permutation_test,
    rbf_kernel_matrix,
    rbf_mmd2,
    sliced_wasserstein,
)

DT = torch.float64


def gaussian_sample(n, mean, std, seed):
    g = torch.Generator().manual_seed(seed)
    mu = torch.as_tensor(mean, dtype=DT)
    return mu + std * torch.randn(n, mu.shape[0], dtype=DT, generator=g)


# --------------------------------------------------------------------------
# MMD against its closed form
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mean_b, std_b",
    [([0.0, 0.0], 1.0), ([0.6, 0.0], 1.0), ([0.0, 0.0], 1.4), ([1.0, -0.5], 1.3)],
)
def test_mmd_matches_the_gaussian_closed_form(mean_b, std_b):
    """Location differences, scale differences, and the null, all at one bandwidth."""
    a = [0.0, 0.0]
    x = gaussian_sample(8000, a, 1.0, seed=0)
    y = gaussian_sample(8000, mean_b, std_b, seed=1)
    h = 1.5
    exact = gaussian_rbf_mmd2(torch.tensor(a), 1.0, torch.tensor(mean_b), std_b, h)
    # Both tolerances are the estimator's own Monte Carlo error at n = 8000:
    # ~1.5e-4 absolute where the truth is 0, and a few percent where it is not.
    assert rbf_mmd2(x, y, bandwidth=h) == pytest.approx(exact, rel=0.06, abs=3e-4)


def test_the_closed_form_is_zero_exactly_when_the_gaussians_agree():
    """Guard on the oracle: it has to be able to say 'no difference'."""
    a = torch.tensor([0.3, -0.2])
    assert gaussian_rbf_mmd2(a, 0.7, a, 0.7, 1.1) == pytest.approx(0.0, abs=1e-15)
    assert gaussian_rbf_mmd2(a, 0.7, a, 0.71, 1.1) > 0.0


def test_the_unbiased_estimator_is_centred_on_zero_under_the_null():
    """The reason the unbiased form is the default.

    Its mean over repeats is 0 to within Monte Carlo error and individual
    values go negative -- whereas the biased (V-statistic) version is positive
    by ~1/n every time, which reads as a real difference and is not one.
    """
    h = 1.5
    vals = [
        rbf_mmd2(
            gaussian_sample(400, [0.0, 0.0], 1.0, seed=2 * i),
            gaussian_sample(400, [0.0, 0.0], 1.0, seed=2 * i + 1),
            bandwidth=h,
        )
        for i in range(40)
    ]
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    assert abs(mean) < 3 * sd / math.sqrt(len(vals))
    assert min(vals) < 0.0  # it really does go negative

    biased = [
        rbf_mmd2(
            gaussian_sample(400, [0.0, 0.0], 1.0, seed=2 * i),
            gaussian_sample(400, [0.0, 0.0], 1.0, seed=2 * i + 1),
            bandwidth=h,
            unbiased=False,
        )
        for i in range(40)
    ]
    assert min(biased) > 0.0
    # And its bias is exactly the diagonal it keeps. Writing the V-statistic
    # as ((n-1)/n) (U-statistic) + 1/n per sample gives E[V] - E[U] =
    # 2 (1 - E k(X, X')) / n, and for two identical isotropic Gaussians
    # E k(X, X') = (h^2 / (h^2 + 2 s^2))^{d/2} -- the same one-line
    # convolution the closed form is built from.
    e_k = (h**2 / (h**2 + 2.0)) ** (2 / 2)
    assert sum(biased) / len(biased) == pytest.approx(2 * (1 - e_k) / 400, rel=0.1)


def test_the_kernel_is_the_advertised_one():
    """``exp(-||x-y||^2 / 2h^2)``: 1 on the diagonal, and a hand-checked value."""
    a = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=DT)
    k = rbf_kernel_matrix(a, a, bandwidth=2.0)
    assert torch.allclose(k.diagonal(), torch.ones(2, dtype=DT))
    assert float(k[0, 1]) == pytest.approx(math.exp(-25.0 / 8.0))


def test_the_median_heuristic_is_the_pooled_median_distance():
    x = gaussian_sample(300, [0.0, 0.0], 1.0, seed=4)
    y = gaussian_sample(300, [2.0, 0.0], 1.0, seed=5)
    pool = torch.cat([x, y])
    d = torch.cdist(pool, pool)
    iu = torch.triu_indices(600, 600, offset=1)
    assert median_heuristic_bandwidth(x, y) == pytest.approx(float(d[iu[0], iu[1]].median()))


# --------------------------------------------------------------------------
# the permutation test
# --------------------------------------------------------------------------
def test_the_permutation_test_rejects_a_real_difference_and_not_the_null():
    g = torch.Generator().manual_seed(7)
    x = gaussian_sample(600, [0.0, 0.0], 1.0, seed=10)
    same = gaussian_sample(600, [0.0, 0.0], 1.0, seed=11)
    shifted = gaussian_sample(600, [0.5, 0.0], 1.0, seed=12)
    null_result = mmd_permutation_test(x, same, n_permutations=199, generator=g)
    alt_result = mmd_permutation_test(x, shifted, n_permutations=199, generator=g)
    assert null_result.p_value > 0.1
    assert alt_result.p_value == pytest.approx(1.0 / 200, abs=1e-9)  # the resolution floor
    assert alt_result.mmd2 > 10 * null_result.null_q95


def test_the_permutation_p_value_is_calibrated_under_the_null():
    """A p-value that is not uniform under the null is not a p-value.

    30 independent null pairs at 99 permutations each: the number with
    ``p <= 0.2`` is Binomial(30, 0.2) if the test is exact, mean 6 and sd 2.2,
    so 15 is a >4 s.d. failure while a genuinely uniform test passes
    comfortably. Deliberately a loose band -- the assertion is that the test
    is not systematically anti-conservative, which is what a wrong ``+1`` or a
    permutation that reuses the observed split would break.
    """
    g = torch.Generator().manual_seed(99)
    small = [
        mmd_permutation_test(
            gaussian_sample(150, [0.0, 0.0], 1.0, seed=1000 + 2 * i),
            gaussian_sample(150, [0.0, 0.0], 1.0, seed=1001 + 2 * i),
            n_permutations=99,
            generator=g,
        ).p_value
        for i in range(30)
    ]
    assert sum(p <= 0.2 for p in small) < 15
    assert max(small) > 0.5  # and it is not stuck at 1 either


def test_the_p_value_cannot_be_zero():
    """The ``+1`` on both sides: the floor is ``1/(1 + n_permutations)``."""
    g = torch.Generator().manual_seed(3)
    x = gaussian_sample(300, [0.0, 0.0], 0.5, seed=20)
    y = gaussian_sample(300, [5.0, 5.0], 0.5, seed=21)
    r = mmd_permutation_test(x, y, n_permutations=49, generator=g)
    assert r.p_value == pytest.approx(1.0 / 50)
    assert r.n_permutations == 49


def test_the_permutation_test_agrees_with_the_direct_estimator():
    """Same statistic, computed once directly and once through the pooled
    kernel matrix the permutations re-index -- a check on the indexing."""
    g = torch.Generator().manual_seed(5)
    x = gaussian_sample(200, [0.0, 0.0], 1.0, seed=30)
    y = gaussian_sample(200, [0.4, 0.1], 1.2, seed=31)
    r = mmd_permutation_test(x, y, n_permutations=1, generator=g)
    assert r.mmd2 == pytest.approx(rbf_mmd2(x, y, bandwidth=r.bandwidth), rel=1e-12)


# --------------------------------------------------------------------------
# sliced Wasserstein against its closed form
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mean_b, std_b", [([1.0, 0.0], 1.0), ([0.0, 0.0], 1.5), ([2.0, -1.0], 1.4)]
)
def test_sliced_wasserstein_matches_the_gaussian_closed_form(mean_b, std_b):
    a = [0.0, 0.0]
    x = gaussian_sample(20_000, a, 1.0, seed=40)
    y = gaussian_sample(20_000, mean_b, std_b, seed=41)
    g = torch.Generator().manual_seed(0)
    got = sliced_wasserstein(x, y, n_projections=512, generator=g)
    exact = gaussian_sliced_wasserstein2(torch.tensor(a), 1.0, torch.tensor(mean_b), std_b)
    assert got.distance == pytest.approx(exact, rel=0.03)
    assert got.stderr < 0.05 * got.distance


def test_the_sliced_closed_form_reduces_to_the_mean_gap_over_sqrt_d():
    """``E[<u, delta>^2] = |delta|^2 / d``: a 2-D shift of 1 gives 1/sqrt(2)."""
    got = gaussian_sliced_wasserstein2(torch.zeros(2), 1.0, torch.tensor([1.0, 0.0]), 1.0)
    assert got == pytest.approx(1.0 / math.sqrt(2.0))


def test_sliced_wasserstein_is_zero_on_identical_samples_and_positive_otherwise():
    x = gaussian_sample(2000, [0.0, 0.0], 1.0, seed=50)
    g = torch.Generator().manual_seed(0)
    assert sliced_wasserstein(x, x.clone(), n_projections=64, generator=g).distance == 0.0
    y = gaussian_sample(2000, [0.05, 0.0], 1.0, seed=51)
    assert sliced_wasserstein(x, y, n_projections=64, generator=g).distance > 0.0


def test_sliced_wasserstein_is_translation_covariant():
    """Shifting one sample by ``t`` far beyond the overlap adds ``|t|/sqrt(d)``
    to the distance, which is the closed form's leading term."""
    x = gaussian_sample(4000, [0.0, 0.0], 1.0, seed=60)
    shift = torch.tensor([20.0, 0.0], dtype=DT)
    g = torch.Generator().manual_seed(0)
    got = sliced_wasserstein(x, x + shift, n_projections=512, generator=g)
    assert got.distance == pytest.approx(20.0 / math.sqrt(2.0), rel=0.02)


def test_the_finite_sample_floor_falls_like_one_over_root_n():
    """Two samples from the *same* distribution do not read zero.

    The empirical sliced distance is biased upward at finite ``n`` -- with a
    finite sample the two empirical quantile functions differ by ``O(n^-1/2)``
    even under the null. Measured here so the experiment's
    reference-versus-reference row can be read as the metric's own floor
    rather than as a defect of a sampler.
    """
    g = torch.Generator().manual_seed(0)
    scaled = []
    for n in (500, 2000, 8000):
        a = gaussian_sample(n, [0.0, 0.0], 1.0, seed=70 + n)
        b = gaussian_sample(n, [0.0, 0.0], 1.0, seed=71 + n)
        d = sliced_wasserstein(a, b, n_projections=256, generator=g).distance
        scaled.append(d * math.sqrt(n))
    assert max(scaled) / min(scaled) < 1.6  # d ~ c / sqrt(n), c stable to ~50%


def test_unequal_sample_sizes_interpolate_rather_than_fail():
    x = gaussian_sample(3000, [0.0, 0.0], 1.0, seed=80)
    y = gaussian_sample(1700, [1.0, 0.0], 1.0, seed=81)
    g = torch.Generator().manual_seed(0)
    got = sliced_wasserstein(x, y, n_projections=512, generator=g)
    exact = gaussian_sliced_wasserstein2(torch.zeros(2), 1.0, torch.tensor([1.0, 0.0]), 1.0)
    assert got.distance == pytest.approx(exact, rel=0.05)


def test_sliced_w1_differs_from_sliced_w2():
    """``p`` is honoured: for a scale difference the two orders disagree."""
    x = gaussian_sample(8000, [0.0, 0.0], 1.0, seed=90)
    y = gaussian_sample(8000, [0.0, 0.0], 2.0, seed=91)
    g = torch.Generator().manual_seed(0)
    w1 = sliced_wasserstein(x, y, n_projections=256, p=1, generator=g).distance
    w2 = sliced_wasserstein(x, y, n_projections=256, p=2, generator=g).distance
    assert w1 < w2  # Jensen: the p-norm over directions grows with p
    # W_1 between N(0,1) and N(0,4) in 1-D is E|z| * (2-1) = sqrt(2/pi).
    assert w1 == pytest.approx(math.sqrt(2.0 / math.pi), rel=0.05)


# --------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_projections": 0}, "n_projections must be >= 1"),
        ({"p": 0}, "p must be >= 1"),
    ],
)
def test_sliced_wasserstein_rejects_bad_arguments(kwargs, message):
    x = gaussian_sample(10, [0.0, 0.0], 1.0, seed=0)
    with pytest.raises(ValueError, match=message):
        sliced_wasserstein(x, x, **kwargs)


def test_mismatched_dimensions_are_refused():
    x = gaussian_sample(10, [0.0, 0.0], 1.0, seed=0)
    y = gaussian_sample(10, [0.0, 0.0, 0.0], 1.0, seed=1)
    with pytest.raises(ValueError, match="differ in dimension"):
        rbf_mmd2(x, y)
    with pytest.raises(ValueError, match="differ in dimension"):
        sliced_wasserstein(x, y)
    with pytest.raises(ValueError, match="differ in dimension"):
        mmd_permutation_test(x, y)


def test_a_one_dimensional_sample_array_is_refused():
    with pytest.raises(ValueError, match=r"x must be \(n, d\)"):
        rbf_mmd2(torch.zeros(10, dtype=DT), torch.zeros(10, 1, dtype=DT))


def test_a_degenerate_pooled_sample_is_refused():
    x = torch.zeros(5, 2, dtype=DT)
    with pytest.raises(ValueError, match="median pairwise distance is zero"):
        rbf_mmd2(x, x)
