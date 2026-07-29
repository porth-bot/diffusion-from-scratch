"""The two summary statistics the sampling experiment reports in prose.

``weight_tv`` and ``mode_spread`` are the numbers that end up in the README's
table and in every figure title, and neither is checked by anything in the
package, so they are checked here: against a target whose mode weights and
mode width are known, and against a deliberately reweighted one whose TV
distance from uniform can be written down.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))

import sampling  # noqa: E402


def test_exact_draws_report_uniform_weights_and_the_target_width():
    p = sampling.target()
    x = p.sample(40_000, generator=torch.Generator().manual_seed(0))
    # 8 modes, 40k draws -> each weight has s.e. 0.0016, so TV (a sum of 8
    # half-deviations) sits around 0.005 and 0.02 is far above the noise.
    assert sampling.weight_tv(sampling.mode_weights(x)) < 0.02
    assert sampling.mode_spread(x) == pytest.approx(sampling.RING_STD, rel=0.02)


def test_the_tv_statistic_reads_a_known_reweighting():
    """A mixture with one mode at double weight has a computable TV distance."""
    from diffusion.targets import GaussianMixture

    p = sampling.target()
    weights = [2.0] + [1.0] * 7
    total = sum(weights)
    tilted = GaussianMixture(p.means, p.covs, torch.tensor(weights) / total, dtype=p.dtype)
    x = tilted.sample(60_000, generator=torch.Generator().manual_seed(1))
    exact_tv = 0.5 * sum(abs(w / total - 1.0 / 8) for w in weights)
    assert sampling.weight_tv(sampling.mode_weights(x)) == pytest.approx(exact_tv, rel=0.05)


def test_mode_spread_tracks_a_wider_target():
    """It measures the component width, not the ring's."""
    from diffusion.targets import GaussianMixture

    wide = GaussianMixture.ring(n_components=8, radius=4.0, std=0.7)
    x = wide.sample(40_000, generator=torch.Generator().manual_seed(2))
    assert sampling.mode_spread(x) == pytest.approx(0.7, rel=0.03)


def test_mode_assignment_is_the_nearest_mode_by_angle():
    """Points placed on the modes are assigned to them, one each."""
    ang = torch.arange(8, dtype=torch.float32) * (2 * math.pi / 8)
    centres = 4.0 * torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
    weights = sampling.mode_weights(centres)
    assert weights == pytest.approx([1 / 8] * 8)
    assert sampling.mode_spread(centres) == pytest.approx(0.0, abs=1e-5)
