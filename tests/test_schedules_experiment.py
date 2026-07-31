"""The schedule sweep makes one theoretical claim and one empirical one. Pin both.

The theoretical claim (``theory/derivations.md`` Sec. 5.2) is that a
variance-exploding noise schedule cancels out of the probability-flow ODE, so
three ladders sharing endpoints describe the same trajectory and differ only in
where they place nodes. ``tests/test_schedules.py`` checks that on a Gaussian
where the flow is closed-form; here it is checked on the log the README quotes,
on the actual ring target with 16,000 samples per cell — the exact-score
PF-ODE arms have to converge onto each other as the budget grows, and they do,
to four decimals.

The empirical claim is the part that is *not* implied by the theory: the
ancestral sampler keeps a schedule dependence at every budget measured, and at
the very coarsest budget the ranking of the schedules is opposite between the
two samplers. Both are asserted here so that a later change to the ladders or
the samplers cannot quietly turn the README's paragraph into fiction.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import schedules as experiment  # noqa: E402

from diffusion.ode import pf_ode_nfe  # noqa: E402


def _rows() -> list[dict]:
    with (ROOT / "logs" / "schedule_sweep.csv").open() as f:
        rows = [
            {k: (v if k in ("sampler", "score", "schedule") else float(v)) for k, v in r.items()}
            for r in csv.DictReader(f)
        ]
    assert rows, "logs/schedule_sweep.csv is empty"
    return rows


def _cell(rows, budget, sampler, score) -> dict[str, float]:
    """``{schedule: sliced W2}`` for one (budget, sampler, score)."""
    return {
        r["schedule"]: r["sw2"]
        for r in rows
        if r["budget"] == budget and r["sampler"] == sampler and r["score"] == score
    }


def test_the_sweep_covers_every_cell_and_averages_over_seeds():
    rows = _rows()
    for budget in experiment.NFE_BUDGETS:
        for sampler in ("ancestral", "pf-ode rk4"):
            for score in ("exact", "learned"):
                cell = _cell(rows, budget, sampler, score)
                assert set(cell) == set(experiment.SCHEDULES), (budget, sampler, score)
    assert {r["n_seeds"] for r in rows} == {float(experiment.N_SEEDS)}
    assert all(r["sw2_sd"] > 0 for r in rows)  # a zero sd would mean one seed reused


@pytest.mark.parametrize("budget", experiment.NFE_BUDGETS)
def test_every_arm_spent_the_budget_it_was_given(budget):
    """A schedule comparison at unequal cost would not be a schedule comparison."""
    for r in _rows():
        if r["budget"] != budget:
            continue
        if r["sampler"] == "ancestral":
            assert r["nfe"] == budget
        else:
            assert r["nfe"] == pf_ode_nfe(int(r["nodes"]), "rk4")
        assert budget - 4 < r["nfe"] <= budget


def test_the_ode_forgets_the_schedule_once_the_ladder_resolves_the_flow():
    """Sec. 5.2's claim, on the real target: the three arms converge onto one.

    At 9 evaluations (3 RK4 nodes) the schedules disagree by 0.45 in sliced
    W_2 — 57x the seed-to-seed spread. By 41 they agree to 5e-4, and by 101 to
    1e-4, which is below the resolution of the statistic. The gap has to shrink
    monotonically across the three coarsest budgets for that to be convergence
    rather than a coincidence at one point.
    """
    rows = _rows()

    def spread(budget):
        c = _cell(rows, budget, "pf-ode rk4", "exact")
        return max(c.values()) - min(c.values())

    assert spread(9) > 0.3
    assert spread(9) > spread(17) > spread(41)
    assert spread(41) < 1e-3
    assert spread(101) < 5e-4
    assert spread(201) < 5e-4


def test_the_ancestral_sampler_does_not_forget_it():
    """The honest counterpart: the theory covers the ODE, not the chain.

    The ancestral update discretizes the reverse *chain*, whose per-level
    plug-in error depends on the gap between adjacent levels — and a linear
    ladder's finest ratio is still 1.8 at 201 levels where the geometric one is
    1.03. So its schedule dependence survives every budget measured, and it is
    the linear ladder that pays.
    """
    rows = _rows()
    for budget in (41, 101, 201):
        c = _cell(rows, budget, "ancestral", "exact")
        assert max(c.values()) - min(c.values()) > 1e-2
        assert c["linear"] == max(c.values())


def test_at_the_coarsest_budget_the_two_samplers_rank_the_schedules_oppositely():
    """Recorded because it is the kind of thing a single-sampler study would miss.

    At 9 evaluations the ancestral sampler has 9 levels and is hurt most by the
    linear ladder's enormous final ratio; RK4 has 3 nodes and is hurt most by
    the geometric ladder spending them where this target's flow has stopped
    moving. Same budget, same target, opposite verdict.
    """
    rows = _rows()
    ancestral = _cell(rows, 9, "ancestral", "exact")
    ode = _cell(rows, 9, "pf-ode rk4", "exact")
    assert max(ancestral, key=ancestral.get) == "linear"
    assert min(ode, key=ode.get) == "linear"
    assert max(ode, key=ode.get) == "geometric"


def test_a_learned_score_swamps_the_schedule_at_a_resolved_budget():
    """Which is the practical conclusion, so it should be a test and not a mood.

    At 201 evaluations every learned-score PF-ODE arm sits within 0.0015 of the
    others while sitting 0.037 above the exact-score arms. The schedule is a
    rounding error next to the estimation error it is being asked to compete
    with.
    """
    rows = _rows()
    learned = _cell(rows, 201, "pf-ode rk4", "learned")
    exact = _cell(rows, 201, "pf-ode rk4", "exact")
    assert max(learned.values()) - min(learned.values()) < 2e-3
    assert min(learned.values()) - max(exact.values()) > 0.03
