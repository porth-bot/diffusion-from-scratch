"""The tilt identity, and what the overlap sweep measured with it.

Three groups. The identity ``log q_w - log q_0 = w (log r_k - log w_k)`` is
checked against ``tilted_log_density`` directly, since everything the
experiment concludes is read off it. The log-space responsibility is checked
against the linear one *and* against the regime where the linear one fails,
because that failure is the reason the helper exists. And the measured
conclusions are pinned against the committed log -- including the spread
reversal, which refutes the "guidance sharpens the conditional" reading the
sweep was built to demonstrate, and is therefore exactly what a later tidy-up
would smooth away.
"""

import json
import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from diffusion.guidance import TiltedGrid, log_tilt, tilted_log_density  # noqa: E402
from diffusion.targets import GaussianMixture  # noqa: E402

import overlap as O  # noqa: E402

LOG = json.loads((ROOT / "logs" / "overlap.json").read_text())
CELLS = LOG["cells"]


def rows(cell):
    return {r["w"]: r for r in cell["rows"]}


# -- the identity ------------------------------------------------------------

@pytest.mark.parametrize("std", [0.4, 1.2, 2.0])
@pytest.mark.parametrize("w", [1.0, 2.0, 4.0])
def test_the_tilt_is_the_responsibility_raised_to_the_guidance_weight(std, w):
    """``log q_w - log q_0`` equals ``log_tilt`` up to a constant, exactly.

    Both sides are unnormalized, so the claim is that their difference is
    *constant in x* -- which is the whole content of "the tilt cancels against
    the normalizer when the responsibility does not vary".
    """
    mix = GaussianMixture.ring(8, 4.0, std)
    x = mix.sample(2000, generator=torch.Generator().manual_seed(5))
    lhs = tilted_log_density(mix, 0, x, w) - tilted_log_density(mix, 0, x, 0.0)
    dev = lhs - log_tilt(mix, 0, x, w)
    assert float((dev - dev.mean()).abs().max()) < 1e-3


def test_the_tilt_vanishes_at_weight_zero():
    mix = GaussianMixture.ring(8, 4.0, 1.2)
    x = mix.sample(64, generator=torch.Generator().manual_seed(1))
    assert torch.equal(log_tilt(mix, 0, x, 0.0), torch.zeros(64))


def test_log_responsibilities_agrees_with_the_linear_form_where_that_works():
    mix = GaussianMixture.ring(8, 4.0, 2.0)   # overlapping: no underflow
    x = mix.sample(1000, generator=torch.Generator().manual_seed(2))
    lin = mix.responsibilities(x).log()
    assert torch.allclose(mix.log_responsibilities(x), lin, atol=1e-5)


def test_the_linear_responsibility_underflows_exactly_where_the_tilt_matters():
    """Why ``log_responsibilities`` exists rather than ``responsibilities().log()``.

    On the *default* ring -- the one every other experiment in this repo uses --
    the off-component responsibilities of a point near a mode are below
    float32's smallest normal, so the linear form returns exact zeros. Any tilt
    diagnostic built on it reads -inf (or the clamp) on precisely the rows that
    establish the separated regime.
    """
    mix = GaussianMixture.ring(8, 4.0, 0.4)
    x = mix.sample(2000, generator=torch.Generator().manual_seed(3))
    lin = mix.responsibilities(x)
    assert (lin == 0).any(), "expected underflow on the default ring"
    assert torch.isinf(lin.log()).any()
    assert torch.isfinite(mix.log_responsibilities(x)).all()


def test_separation_in_component_sds_is_the_chord_over_the_width():
    for std in O.STDS:
        assert O.separation_sd(std) == pytest.approx(
            2 * O.RADIUS * math.sin(math.pi / O.N_COMPONENTS) / std
        )


# -- the sweep, as committed -------------------------------------------------

def test_the_sweep_covers_the_default_target_and_widens_from_it():
    assert CELLS[0]["std"] == 0.4                    # the repo's ring
    assert CELLS[0]["separation_sd"] == pytest.approx(7.65, abs=0.01)
    seps = [c["separation_sd"] for c in CELLS]
    assert seps == sorted(seps, reverse=True)
    assert seps[-1] < 2.0


def test_the_grid_is_not_clipping_any_cell():
    """Every distance is measured against a normalizer; a clipped box would
    make all of them functions of ``half_width``."""
    worst = max(r["tail_mass"] for c in CELLS for r in c["rows"])
    assert worst < 1e-6, worst


def test_guidance_does_nothing_at_all_on_the_separated_ring():
    """The reproduction of the earlier guidance study's negative, now with the
    reason attached: the tilt is a near-constant, so ``q_w = q_0``."""
    c = CELLS[0]
    assert c["sd_log_tilt"] < 0.1
    assert c["mean_responsibility"] > 0.999
    for w, r in rows(c).items():
        assert r["distance_to_q0"] / c["floor"] < 1.5, (w, r)


def test_the_tilt_spread_grows_as_the_components_close_in():
    spreads = [c["sd_log_tilt"] for c in CELLS]
    assert spreads == sorted(spreads)
    assert spreads[-1] / spreads[0] > 10


def test_q_w_departs_from_the_conditional_monotonically_in_both_variables():
    for c in CELLS[1:]:
        d = [rows(c)[w]["distance_to_q0"] for w in LOG["weights"]]
        assert d == sorted(d), c["separation_sd"]
        assert d[-1] / c["floor"] > 5, c["separation_sd"]
    strongest = [rows(c)[4.0]["distance_to_q0"] / c["floor"] for c in CELLS]
    assert strongest == sorted(strongest)


def test_purity_is_saturated_when_separated_and_informative_when_not():
    """The read-out the earlier study discarded is the one this regime needs."""
    assert min(r["purity"] for r in CELLS[0]["rows"]) > 0.9998
    worst = CELLS[-1]
    p = [rows(worst)[w]["purity"] for w in LOG["weights"]]
    assert p == sorted(p)
    assert p[0] < 0.6 and p[-1] > 0.9


# -- the part that refutes the section's own premise -------------------------

def test_the_displacement_is_unbounded_while_the_narrowing_is_not():
    """The measured correction to "guidance sharpens the conditional".

    Across the whole sweep the mean moves by up to ~0.9 component sd and keeps
    moving with ``w``, while the spread never falls below ~0.82 sd. Asserted
    with both bounds because the interesting statement is the ratio between
    them, not either one.
    """
    disp = [r["displacement"] for c in CELLS for r in c["rows"]]
    spread = [r["spread"] for c in CELLS for r in c["rows"]]
    assert max(disp) > 0.85
    assert min(spread) > 0.80, "narrowing deeper than 20% would change the story"
    for c in CELLS[1:]:
        d = [rows(c)[w]["displacement"] for w in LOG["weights"]]
        assert d == sorted(d), c["separation_sd"]


def test_the_narrowing_reverses_at_high_weight_in_the_overlapping_cells():
    """Not monotone, and the sweep was built expecting it to be.

    In the two most overlapping cells the spread has an interior minimum: more
    guidance past that point re-widens the distribution while continuing to
    displace it. Pinned so that a later edit cannot quietly report "guidance
    sharpens, monotonically in w".
    """
    for c in CELLS[-2:]:
        s = [rows(c)[w]["spread"] for w in LOG["weights"]]
        assert s[-1] > min(s), (c["separation_sd"], s)
        assert s.index(min(s)) not in (0, len(s) - 1), (c["separation_sd"], s)
