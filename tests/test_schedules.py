"""Tests for the noise schedules, and for the claim that a schedule is only node placement.

Two separate things are checked here and they are worth keeping apart.

1. **The ladders are what they say they are.** Each is uniform spacing in a
   warp of ``sigma`` (``log``, the identity, ``arctan``), they hit the
   requested endpoints exactly, they are decreasing, and the cosine one really
   is Nichol & Dhariwal's ``alphabar = cos^2`` schedule rewritten for a
   variance-exploding process -- checked against that definition rather than
   against the ``arctan`` form it was derived into.

2. **The schedule does not change the flow, only its discretization.**
   ``theory/derivations.md`` Sec. 5.2 shows the schedule cancels out of the
   probability-flow ODE once ``sigma`` is the clock, so three different ladders
   with the same endpoints must integrate to the *same* map. On a Gaussian that
   map is closed-form, so the claim can be tested to 1e-6 rather than by
   comparing two sets of samples -- and the same test at 3 nodes shows the
   three answers a full six orders of magnitude apart, which is what makes the
   fine-ladder agreement evidence rather than a tautology.

The last test records something that surprised this repo: on that Gaussian the
*ranking* of the schedules flips with the node count, because which end of the
range the flow does its moving in is a property of the target, not of the
ladder. It is asserted rather than narrated so it cannot quietly stop being
true.
"""

from __future__ import annotations

import math

import pytest
import torch

from diffusion.langevin import (
    SIGMA_SCHEDULES,
    cosine_sigmas,
    geometric_sigmas,
    linear_sigmas,
    sigma_ladder,
)
from diffusion.ode import gaussian_pf_ode_scale, pf_ode_sample
from diffusion.targets import Gaussian

DT = torch.float64
SIGMA_MIN, SIGMA_MAX = 0.05, 8.0


# --------------------------------------------------------------------------
# the ladders are the warps they claim to be
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(SIGMA_SCHEDULES))
def test_every_schedule_hits_the_endpoints_and_decreases(name):
    ladder = sigma_ladder(name, SIGMA_MAX, SIGMA_MIN, 17)
    assert len(ladder) == 17
    assert ladder[0] == pytest.approx(SIGMA_MAX)
    assert ladder[-1] == pytest.approx(SIGMA_MIN)
    assert ladder == sorted(ladder, reverse=True)
    assert all(s > 0 for s in ladder)


@pytest.mark.parametrize(
    "builder, warp",
    [
        (geometric_sigmas, math.log),
        (linear_sigmas, lambda s: s),
        (cosine_sigmas, math.atan),
    ],
)
def test_each_ladder_is_uniform_in_its_own_warp(builder, warp):
    """The defining property: equal spacing after the warp, and only after it."""
    ladder = builder(SIGMA_MAX, SIGMA_MIN, 13)
    gaps = [warp(a) - warp(b) for a, b in zip(ladder, ladder[1:])]
    assert max(gaps) == pytest.approx(min(gaps), rel=1e-9)


def test_the_three_warps_are_genuinely_different_ladders():
    """Guards against a refactor collapsing them; the middle level is the tell."""
    mid = {name: sigma_ladder(name, SIGMA_MAX, SIGMA_MIN, 9)[4] for name in SIGMA_SCHEDULES}
    # geometric sits lowest (uniform in log pulls the middle toward sigma_min),
    # linear highest, cosine between them.
    assert mid["geometric"] < mid["cosine"] < mid["linear"]
    assert mid["linear"] / mid["geometric"] > 5


def test_cosine_matches_the_variance_preserving_definition_it_came_from():
    """``sigma^2 = (1 - alphabar)/alphabar`` with ``alphabar = cos^2`` of a uniform angle.

    The implementation is ``uniform in arctan(sigma)``, which is the *derived*
    form. This checks the derivation by going back to Nichol & Dhariwal's own
    parameterization: recover ``alphabar`` from each level and assert that its
    arccos is arithmetically spaced.
    """
    ladder = cosine_sigmas(SIGMA_MAX, SIGMA_MIN, 21)
    alphabar = [1.0 / (1.0 + s * s) for s in ladder]  # invert sigma^2 = (1-a)/a
    angles = [math.acos(math.sqrt(a)) for a in alphabar]  # alphabar = cos^2(angle)
    gaps = [b - a for a, b in zip(angles, angles[1:])]
    assert max(gaps) == pytest.approx(min(gaps), rel=1e-9)
    # ... and the angle really is the arctan of the level, which is the identity
    # the docstring derives (tan of the VP angle is the VE noise level).
    for s, ang in zip(ladder, angles):
        assert math.tan(ang) == pytest.approx(s, rel=1e-9)


@pytest.mark.parametrize("name", sorted(SIGMA_SCHEDULES))
def test_a_two_level_ladder_is_the_endpoints_whatever_the_warp(name):
    assert sigma_ladder(name, SIGMA_MAX, SIGMA_MIN, 2) == pytest.approx([SIGMA_MAX, SIGMA_MIN])
    assert sigma_ladder(name, SIGMA_MAX, SIGMA_MIN, 1) == pytest.approx([SIGMA_MAX])


@pytest.mark.parametrize("builder", [linear_sigmas, cosine_sigmas])
def test_the_new_ladders_refuse_impossible_ranges(builder):
    with pytest.raises(ValueError, match="n_levels"):
        builder(SIGMA_MAX, SIGMA_MIN, 0)
    with pytest.raises(ValueError, match="sigma_max"):
        builder(SIGMA_MAX, -1.0, 5)
    with pytest.raises(ValueError, match="sigma_max"):
        builder(0.1, 1.0, 5)


def test_an_unknown_schedule_name_is_refused_and_names_the_known_ones():
    with pytest.raises(ValueError, match="unknown schedule"):
        sigma_ladder("cosine_v2", SIGMA_MAX, SIGMA_MIN, 5)


# --------------------------------------------------------------------------
# the schedule is only node placement
# --------------------------------------------------------------------------
VAR = 0.6
X0 = torch.tensor([[1.0, -2.0], [3.0, 0.5], [-4.0, 4.0]], dtype=DT)


def _flow_outputs(n_nodes: int) -> dict[str, torch.Tensor]:
    """Run the PF-ODE on a Gaussian down each ladder, same endpoints, same start."""
    p = Gaussian([0.0, 0.0], VAR * torch.eye(2, dtype=DT), dtype=DT)

    def score(x, sigma):
        return p.noised(sigma).score(x)

    return {
        name: pf_ode_sample(
            score, X0, sigma_ladder(name, SIGMA_MAX, SIGMA_MIN, n_nodes),
            method="rk4", final_step=False,
        )
        for name in SIGMA_SCHEDULES
    }


def _flow_errors(n_nodes: int) -> dict[str, float]:
    """Max pointwise error against the closed-form Gaussian flow (Sec. 5.2).

    The exact solution depends on the *endpoints* only, so it is the same
    reference for all three ladders -- which is the claim being tested.
    """
    scale = gaussian_pf_ode_scale(VAR, SIGMA_MAX, SIGMA_MIN)
    return {
        name: float((got - X0 * scale).abs().max())
        for name, got in _flow_outputs(n_nodes).items()
    }


def _pairwise_spread(outputs: dict[str, torch.Tensor]) -> float:
    return max(float((a - b).abs().max()) for a in outputs.values() for b in outputs.values())


def test_a_fine_ladder_gives_the_same_flow_whatever_the_schedule():
    """All three converge to the one trajectory the endpoints define."""
    errors = _flow_errors(128)
    assert max(errors.values()) < 1e-6
    assert _pairwise_spread(_flow_outputs(128)) < 1e-6


def test_the_schedules_do_disagree_when_the_ladder_is_coarse():
    """The control that makes the test above mean something.

    If the ladders agreed at every resolution, agreement at 128 nodes would say
    nothing. At 3 nodes the three answers differ from each other by ~1e-1 --
    six orders of magnitude more than at 128 -- so the fine-ladder agreement is
    convergence to a shared limit, not a tautology about the code.
    """
    coarse = _pairwise_spread(_flow_outputs(3))
    fine = _pairwise_spread(_flow_outputs(128))
    assert coarse > 1e-2
    assert coarse > 1e5 * fine


def test_which_schedule_wins_depends_on_how_many_nodes_there_are():
    """A recorded finding, not a preference: the ranking flips with resolution.

    On this Gaussian the flow's action is concentrated at the *coarse* end (the
    contraction factor sqrt((v+s^2)/(v+8^2)) does most of its moving while sigma
    is large), so with only 3 nodes the ladders that put nodes up there -- linear
    and cosine -- beat the geometric one the repo uses by 2x. Give the solver 17
    nodes and it reverses: geometric and cosine are ~6x more accurate than
    linear, whose fine end is a single enormous step. So "which schedule is
    best" is not a property of the schedule alone.
    """
    coarse = _flow_errors(3)
    assert coarse["geometric"] > 1.5 * min(coarse["linear"], coarse["cosine"])

    fine = _flow_errors(17)
    assert fine["linear"] > 3 * max(fine["geometric"], fine["cosine"])
