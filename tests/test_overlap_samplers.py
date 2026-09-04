"""What the overlapping-regime sampler sweep measured, pinned to its log.

`experiments/overlap.py` established what `q_w` does once the components
overlap and stopped there, leaving a bullet in the README's Limitations saying
in as many words that "whether ancestral sampling still overshoots once the
components overlap is untested". `experiments/overlap_samplers.py` runs that
test. These assertions are the conclusions it reached, held against
`logs/overlap_samplers.csv` so a later edit cannot quietly soften them --
particularly the one that cuts *against* the section's own headline (the
overshoot is not monotone in overlap, and once ancestral's guidance-free error
is divided out it barely moves at all).

The derived read-outs are recomputed from the logged columns rather than read
back, so the arithmetic the README quotes is under test too.
"""

import csv
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import overlap_samplers as OS  # noqa: E402

ROWS = [
    {k: float(v) for k, v in r.items()}
    for r in csv.DictReader((ROOT / "logs" / "overlap_samplers.csv").open())
]
D = OS.derived(ROWS)
SEPS = sorted({r["separation_sd"] for r in D}, reverse=True)


def at(sep, w):
    return next(r for r in D
                if r["separation_sd"] == pytest.approx(sep) and r["w"] == w)


def test_the_sweep_covers_the_regimes_it_claims_to():
    """Separated (the guidance study's own target) through heavily overlapping."""
    assert len(SEPS) == 3
    assert SEPS[0] == pytest.approx(7.65, abs=0.01)   # experiments/guidance.py's ring
    assert SEPS[-1] < 2.0
    assert {r["w"] for r in D} == {0.0, 1.0, 2.0, 4.0}


def test_langevin_reaches_q_w_at_every_separation():
    """The answer to the open bullet, positive half.

    Annealed Langevin equilibrates at each level, so its output is the
    stationary law of the last one -- and that reading was established where
    the tilt did nothing. It survives the regime where the tilt does something:
    every cell sits at or below the two-independent-draws floor, and the radial
    landing point matches q_w to under 0.02 component sd.
    """
    for r in D:
        assert r["langevin_ratio"] < 1.2, r
        assert abs(r["langevin_excess_sd"]) < 0.02, r


def test_ancestral_still_overshoots_when_the_components_overlap():
    """The answer to the open bullet, negative half -- and it is not absorbed.

    q_w itself displaces outward at overlap (0.55 and 1.80 in absolute units at
    w = 4), and the open question was whether a sampler that drifts outward is
    drifting toward a target that has already moved. It is not: the excess
    *past* q_w is positive and growing in w at every separation, so the two
    add.
    """
    for sep in SEPS:
        prev = -1.0
        for w in (1.0, 2.0, 4.0):
            e = at(sep, w)["ancestral_excess_sd"]
            assert e > 0.15, (sep, w, e)
            assert e > prev
            prev = e
    assert at(SEPS[1], 4.0)["ancestral_excess_sd"] == pytest.approx(0.772, abs=0.01)


def test_ancestral_overshoot_is_not_monotone_in_overlap():
    """The result that cuts against the section's own headline.

    "Overlap makes it worse" is the reading the raw distance column invites,
    and it does not survive: the most overlapping cell overshoots *less* than
    the middle one. Asserted so that a later tidy-up cannot round the sweep
    into a trend it does not have.
    """
    excess = [at(sep, 4.0)["ancestral_excess_sd"] for sep in SEPS]
    assert excess[1] > excess[2] > excess[0]


def test_dividing_out_the_guidance_free_error_flattens_the_comparison():
    """Ancestral's w = 0 error is not constant across the sweep, and matters.

    At w = 0 there is no tilt, so that row is ancestral's ordinary
    discretization error on a target whose components are getting wider -- and
    it nearly doubles across the sweep (1.85 -> 3.11 floors). Divide it out and
    the guidance-attributable inflation at w = 4 is ~9x at *both* the separated
    and the most overlapping cell. Reporting the undivided column as evidence
    that overlap hurts ancestral would be attributing that baseline to
    guidance.
    """
    w0 = [at(sep, 0.0)["ancestral_ratio"] for sep in SEPS]
    assert max(w0) / min(w0) > 1.6
    infl = [at(sep, 4.0)["ancestral_inflation"] for sep in SEPS]
    assert infl[0] == pytest.approx(infl[2], rel=0.05)     # 9.01 vs 9.02
    assert all(i == pytest.approx(1.0) for i in
               (at(sep, 0.0)["ancestral_inflation"] for sep in SEPS))


def test_the_w0_control_drifts_inward_not_outward():
    """Which is why the outward drift at w > 0 is guidance and not the sampler.

    Ancestral's guidance-free error on this target lands *short* of the
    component mean at every separation. So the raw radial offset at w > 0
    understates the guided drift rather than inheriting it.
    """
    for sep in SEPS:
        assert at(sep, 0.0)["ancestral_excess_sd"] < 0.0


def test_the_ladder_top_matches_the_guidance_study_at_the_separated_cell():
    """sigma_max is scaled with the component width, which could confound the
    sweep -- so the cell that reproduces the guidance study's setting must
    reproduce its ladder exactly."""
    import sampling  # noqa: E402

    assert OS.sigma_max_for(0.4) == pytest.approx(sampling.SIGMA_MAX)
    assert OS.sigma_max_for(2.0) > OS.sigma_max_for(1.2) > OS.sigma_max_for(0.4)
    for r in ROWS:
        assert r["tail_mass"] < 1e-4, r      # the quadrature grid holds q_w
