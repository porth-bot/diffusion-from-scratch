"""The ODE-vs-SDE comparison is only worth reading if the budgets really match.

``experiments/ode.py`` claims to put three samplers on one "score evaluations"
axis. RK4 spends four evaluations per step, so its node count has to be divided
down -- and if that arithmetic drifts, the figure quietly becomes a comparison
at 4x the cost with no error anywhere. Checked twice: on the helper, and on the
committed log the README's numbers come from.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import ode as experiment  # noqa: E402

from diffusion.ode import pf_ode_nfe  # noqa: E402


@pytest.mark.parametrize("budget", experiment.NFE_BUDGETS)
def test_the_rk4_node_count_fits_the_budget_and_nearly_fills_it(budget):
    nodes = experiment.rk4_nodes_for(budget)
    spent = pf_ode_nfe(nodes, "rk4")
    assert spent <= budget
    assert spent > budget - 4  # one more node would overrun, so this is tight
    assert nodes >= 2  # an integration needs a step


def test_every_arm_in_the_committed_sweep_spent_its_budget():
    """The log itself, not just the helper that produced it."""
    with (ROOT / "logs" / "ode_sweep.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "logs/ode_sweep.csv is empty"
    for r in rows:
        budget, nfe = int(r["budget"]), int(r["nfe"])
        assert budget - 3 <= nfe <= budget, f"{r['sampler']} at budget {budget} spent {nfe}"
    # ...and all three dynamics are actually present at every budget, so no
    # curve is silently missing from the figure.
    for budget in experiment.NFE_BUDGETS:
        names = {r["sampler"] for r in rows if int(r["budget"]) == budget}
        assert names == {"ancestral", "pf-ode euler", "pf-ode rk4"}


def test_the_exact_score_exact_prior_likelihood_row_is_the_methods_own_zero():
    """The control that makes the KL numbers readable.

    Transporting ``p_{sigma_min}`` with the exact score and the exact
    top-of-ladder prior must return the target's own density, so that row's
    mean gap is a pure error bar for everything else in the table. If it drifts
    away from zero, the other two rows stop being interpretable as KLs.
    """
    with (ROOT / "logs" / "ode_likelihood.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["kind"] == "kl"]
    exact = [r for r in rows if r["score"] == "exact" and r["prior"] == "p_sigma_max"]
    assert len(exact) == 1
    assert abs(float(exact[0]["mean_gap"])) < 1e-3
    # and the two departures from truth cost strictly more than that floor
    others = [r for r in rows if r not in exact]
    assert len(others) == 2
    for r in others:
        assert -float(r["mean_gap"]) > 1e-3  # KL = -mean_gap, positive
