"""The VP-coordinates network, and what the experiment measured with it.

Three groups. The wrapper is checked against the algebra it implements (it is
one multiplication, so there is no excuse for it not to be exact). The
closed-form input-scale prediction is checked against its own definition and
against the identity ``v/(1 + sigma_max^2)``, which is the quantity the result
turned out to be about. And the measured conclusions are pinned against the
committed CSV -- including the ones that refute the prediction the experiment
was designed around, since those are what a later edit would be tempted to
tidy.
"""

import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from diffusion.model import ScoreNet, count_parameters  # noqa: E402
from diffusion.targets import GaussianMixture  # noqa: E402
from diffusion.vp import (  # noqa: E402
    VPInputScoreNet,
    alphabar_from_sigma,
    input_sd,
    to_vp,
)

import vp_coords as V  # noqa: E402


# -- the wrapper -------------------------------------------------------------

def test_the_vp_network_is_the_ve_network_on_the_rescaled_input():
    """The definition, exactly: one multiplication by sqrt(abar) and nothing
    else, including a parameter count that cannot differ."""
    torch.manual_seed(0)
    ve = ScoreNet(dim=2, width=32, depth=2)
    torch.manual_seed(0)
    vp = VPInputScoreNet(dim=2, width=32, depth=2)
    assert count_parameters(ve) == count_parameters(vp)

    x = torch.randn(6, 2)
    sig = torch.tensor([0.05, 0.2, 1.0, 2.0, 4.0, 8.0])
    scaled = x * torch.sqrt(alphabar_from_sigma(sig)).unsqueeze(1)
    assert torch.allclose(vp.epsilon(x, sig), ve.epsilon(scaled, sig), atol=0)


def test_the_two_networks_agree_where_the_coordinates_do():
    """``abar -> 1`` as ``sigma -> 0``, so the coordinate change vanishes at the
    fine end -- which is also why Sec. 6's per-level table shows no VP effect
    there."""
    torch.manual_seed(0)
    ve = ScoreNet(dim=2, width=32, depth=2)
    torch.manual_seed(0)
    vp = VPInputScoreNet(dim=2, width=32, depth=2)
    x = torch.randn(4, 2)
    assert torch.allclose(vp.epsilon(x, 1e-6), ve.epsilon(x, 1e-6), atol=1e-5)
    # and disagree where they should
    assert not torch.allclose(vp.epsilon(x, 8.0), ve.epsilon(x, 8.0), atol=1e-3)


def test_the_score_follows_from_epsilon_with_no_further_change():
    """``forward`` is written in terms of ``epsilon``, so the VP network returns
    a score in *VE* coordinates and every downstream consumer is unchanged."""
    torch.manual_seed(0)
    vp = VPInputScoreNet(dim=2, width=32, depth=2)
    x, s = torch.randn(5, 2), 2.0
    assert torch.allclose(vp(x, s), -vp.epsilon(x, s) / s)


# -- the closed-form premise -------------------------------------------------

def test_input_sd_matches_its_definition():
    v, sig = 8.16, torch.tensor([0.05, 1.0, 8.0])
    assert torch.allclose(input_sd(sig, v, "ve"), torch.sqrt(v + sig * sig))
    assert torch.allclose(input_sd(sig, v, "vp"),
                          torch.sqrt((v + sig * sig) / (1 + sig * sig)))
    with pytest.raises(ValueError):
        input_sd(sig, v, "vq")
    with pytest.raises(ValueError):
        input_sd(sig, 0.0, "ve")


def test_vp_inputs_are_exactly_unit_scale_only_on_unit_variance_data():
    """The premise the experiment was built to test, and the reason it is
    conditional: at v = 1 the VP input sd is exactly 1 at every level; at this
    repo's v = 8.16 it spans as far as VE does, in the other direction."""
    sig = torch.logspace(-2, 1, 25)
    assert torch.allclose(input_sd(sig, 1.0, "vp"), torch.ones_like(sig))
    native = input_sd(sig, V.DATA_VAR, "vp")
    assert float(native.max() / native.min()) > 2.5


def test_the_ring_variance_is_the_closed_form_the_experiment_assumes():
    """``r^2/2 + s^2``, checked against the sampler rather than trusted."""
    p = GaussianMixture.ring(n_components=8, radius=4.0, std=0.4)
    x = p.sample(200_000, generator=torch.Generator().manual_seed(0))
    assert float(x.var(dim=0).mean()) == pytest.approx(V.DATA_VAR, rel=0.02)
    assert V.DATA_VAR == pytest.approx(4.0**2 / 2 + 0.4**2)


def test_standardizing_holds_the_snr_ladder_fixed():
    """The control that makes the two targets comparable: dividing the data and
    the whole sigma ladder by the same constant is an exact rescaling, so every
    rung sits at the same signal-to-noise ratio."""
    for a, b in zip(V.eval_sigmas("native"), V.eval_sigmas("unit")):
        assert b == pytest.approx(a / V.DATA_SD)
    assert V.top_of_ladder_snr("native", "full") == pytest.approx(
        V.top_of_ladder_snr("unit", "full"), rel=0.15)


def test_top_of_ladder_snr_is_the_vp_identity_maps_error():
    """The identity that turns a formula into an explanation.

    ``x_vp - eps = sqrt(abar) x_0 + (sqrt(1-abar) - 1) eps``, so predicting
    ``eps`` by *copying the VP input* costs ``abar v`` relative to
    ``E||eps||^2`` -- which is exactly ``top_of_ladder_snr``. Measured against
    a sample rather than restated.
    """
    for scale, ladder in V.cells():
        p = V.target(scale)
        smax = V.eval_sigmas(scale, ladder)[-1]
        g = torch.Generator().manual_seed(12345)
        x0 = p.sample(40_000, generator=g)
        eps = torch.randn(x0.shape, generator=g)
        xvp = to_vp(x0 + smax * eps, smax)
        measured = float(((xvp - eps) ** 2).sum(1).mean()
                         / (eps ** 2).sum(1).mean())
        assert measured == pytest.approx(V.top_of_ladder_snr(scale, ladder),
                                         rel=0.05)


# -- what the runs measured --------------------------------------------------

@pytest.fixture(scope="module")
def rows():
    r = V.load_rows()
    if len(r) != V.n_runs():
        pytest.skip(f"{len(r)} of {V.n_runs()} runs cached; "
                    "run experiments/vp_coords.py --train")
    return r


def test_the_csv_covers_every_cell(rows):
    assert len(rows) == V.n_runs() == 40
    for scale, ladder in V.cells():
        for coords in V.COORDS:
            assert len(V.arm(rows, coords, scale, ladder)) == len(V.SEEDS)


def test_vp_helps_and_the_aggregate_metric_hides_it(rows):
    """Two claims at once, because they are the same claim.

    Mean relative MSE over the ladder says VP helps on the native target and
    does nothing on the other three. On the two full-ladder cells that metric
    is 41-47% a single level -- the error at the finest sigma is 44x and 57x
    the error at the coarsest -- so it is very nearly "relative MSE at
    sigma_min", where the two coordinate systems are identical. Weighting the
    levels equally, VP wins in 5/5, 5/5 and 4/5 seeds on the three deep-ladder
    cells.
    """
    import statistics as st

    def level(scale, ladder, i):
        return st.median(V.arm(rows, "ve", scale, ladder, f"rel_mse_L{i}"))

    for scale, ladder, lo in (("native", "full", 40), ("unit", "full", 50)):
        fine = level(scale, ladder, 0)
        coarse = level(scale, ladder, V.N_EVAL_LEVELS - 1)
        share = fine / sum(level(scale, ladder, i)
                           for i in range(V.N_EVAL_LEVELS))
        assert fine / coarse > lo, (scale, ladder)
        assert 0.40 < share < 0.50, (scale, ladder, share)

    for scale, ladder, wins in (("native", "full", 5), ("unit", "full", 5),
                                ("native", "short", 4)):
        gm = V.log_mean_ratios(rows, scale, ladder)
        assert sum(1 for g in gm if g < 1) == wins, (scale, ladder)
        assert sorted(gm)[len(gm) // 2] < 0.95


def test_the_o1_input_argument_is_refuted(rows):
    """The prediction this experiment was designed around, and its refutation.

    Standardizing the data takes VP's input span from 2.71x to *exactly* 1.00x
    -- the O(1)-input ideal -- and the argument says that is where VP should
    help most. It is where VP helps *least* of the three deep-ladder cells, and
    at the coarsest level the two targets are indistinguishable (0.54 against
    0.53) across an 8.2x change in data variance.
    """
    spans = {(r["scale"], r["ladder"]): r["span"]
             for r in V.predicted_input_spans() if r["coords"] == "vp"}
    assert spans[("unit", "full")] == pytest.approx(1.0, abs=1e-6)
    assert spans[("native", "full")] == pytest.approx(2.71, abs=0.02)

    def coarse_ratio(scale, ladder):
        ve = V.arm(rows, "ve", scale, ladder, f"rel_mse_L{V.N_EVAL_LEVELS - 1}")
        vp = V.arm(rows, "vp", scale, ladder, f"rel_mse_L{V.N_EVAL_LEVELS - 1}")
        r = sorted(b / a for a, b in zip(ve, vp))
        return r[len(r) // 2]

    native, unit = coarse_ratio("native", "full"), coarse_ratio("unit", "full")
    assert native == pytest.approx(0.53, abs=0.03)
    assert unit == pytest.approx(0.54, abs=0.03)
    assert abs(native - unit) < 0.05              # the 8.2x in v buys nothing
    # And on the aggregate the standardized arm is the *weaker* of the two.
    gm = V.log_mean_ratios
    assert (sorted(gm(rows, "unit", "full"))[2]
            > sorted(gm(rows, "native", "full"))[2])


def test_the_ladder_length_does_not_explain_it_either(rows):
    """The second hypothesis, also refuted. Cutting the ladder from 160x to 16x
    with the coarse end untouched leaves the effect where it was."""
    assert V.LADDERS["short"] == (10.0, 1.0)
    full = sorted(V.log_mean_ratios(rows, "native", "full"))
    short = sorted(V.log_mean_ratios(rows, "native", "short"))
    assert abs(full[2] - short[2]) < 0.1


def test_the_shallow_ladder_loses_most_of_the_effect(rows):
    """The one knob that moves the whole-ladder number. Three cells sit at the
    same top-of-ladder SNR *by construction* -- which is a flaw in the first
    design and the reason the shallow ladder exists -- and the fourth, 13x
    higher, is the only one where VP stops winning."""
    snrs = {(s, ell): V.top_of_ladder_snr(s, ell) for s, ell in V.cells()}
    deep = [snrs[c] for c in (("native", "full"), ("unit", "full"),
                              ("native", "short"))]
    assert max(deep) / min(deep) < 1.2
    assert snrs[("native", "shallow")] / max(deep) > 10

    shallow = sorted(V.log_mean_ratios(rows, "native", "shallow"))
    assert sum(1 for g in shallow if g < 1) == 3      # against 5, 5, 4
    assert shallow[2] > 0.9                           # against 0.78-0.90


# -- and the confound in that comparison ------------------------------------

def test_the_cells_are_only_comparable_in_units_of_the_data_sd(rows):
    """``sigma`` means different things in the two targets, and the ladders
    reach different distances. Both are removed by the same normalization: the
    three deep cells stop at 2.80 sd and the shallow one at 0.70."""
    ends = {(s, ell): V.normalized_sigmas(s, ell)[-1] for s, ell in V.cells()}
    for cell in (("native", "full"), ("unit", "full"), ("native", "short")):
        assert ends[cell] == pytest.approx(2.801, rel=1e-3)
    assert ends[("native", "shallow")] == pytest.approx(0.700, rel=1e-3)
    # unit/full is the cell that makes this a normalization and not a relabel.
    # In *absolute* terms its ladder tops out at 2.80 against the shallow
    # ladder's 2.00 -- a factor of 1.4, closer to it than to the other two
    # deep cells at 8.00 -- and yet it behaves like the deep cells and not
    # like the shallow one. Absolute sigma is not the variable; sigma/sd is.
    assert V.sigma_range("unit", "full")[1] / V.sigma_range(
        "native", "shallow")[1] == pytest.approx(1.40, abs=0.02)
    assert ends[("unit", "full")] / ends[("native", "shallow")] == \
        pytest.approx(4.0, abs=0.02)


def test_the_shallow_ladder_is_not_different_it_just_stops_short(rows):
    """The correction to the reading above, and the reason the figure's last
    panel is a band split rather than a scatter against top-of-ladder SNR.

    Scored on the band all four ladders cover, the shallow cell is
    indistinguishable from the other three (0.92-0.99, and shallow sits in the
    middle at 0.95). Scored on the band above it, the three that reach it all
    drop to 0.68-0.78. So the controlling variable is the *noise level*, and
    the ladder enters only through how much of that level it covers -- the
    whole-ladder numbers differ because they average different bands.
    """
    shared = {c: V.band_summary(rows, *c, 0.0, V.SHARED_BAND) for c in V.cells()}
    deep = {c: V.band_summary(rows, *c, V.SHARED_BAND, float("inf"))
            for c in V.cells()}

    meds = [shared[c]["median"] for c in V.cells()]
    assert min(meds) > 0.90 and max(meds) < 1.0
    assert max(meds) - min(meds) < 0.08
    sh = shared[("native", "shallow")]["median"]
    assert min(meds) <= sh <= max(meds)               # inside, not an outlier

    assert deep[("native", "shallow")] is None        # never gets there
    for c in (("native", "full"), ("unit", "full"), ("native", "short")):
        assert deep[c]["median"] < 0.80
        assert deep[c]["median"] < shared[c]["median"]
    # Every level is scored exactly once, in one band or the other.
    for c in V.cells():
        n_deep = 0 if deep[c] is None else deep[c]["levels"]
        assert shared[c]["levels"] + n_deep == V.N_EVAL_LEVELS


def test_the_band_split_is_a_restriction_of_the_same_statistic(rows):
    """``log_mean_ratios`` with no band is the whole ladder, so the split
    cannot be measuring something else."""
    for scale, ladder in V.cells():
        whole = V.log_mean_ratios(rows, scale, ladder)
        assert whole == V.log_mean_ratios(rows, scale, ladder, 0.0,
                                          float("inf"))
        lo = V.log_mean_ratios(rows, scale, ladder, 0.0, V.SHARED_BAND)
        hi = V.log_mean_ratios(rows, scale, ladder, V.SHARED_BAND,
                               float("inf"))
        n_lo = len(V.band_summary(rows, scale, ladder, 0.0,
                                  V.SHARED_BAND)["levels"] * [0])
        if not hi:
            assert lo == pytest.approx(whole)
            continue
        n_hi = V.N_EVAL_LEVELS - n_lo
        for w, a, b in zip(whole, lo, hi):
            assert w == pytest.approx(a ** (n_lo / V.N_EVAL_LEVELS)
                                      * b ** (n_hi / V.N_EVAL_LEVELS))


def test_the_sign_test_reports_its_own_resolution_floor():
    """Five paired seeds cannot produce p < 0.0625, so a null result here has
    to be read as 'not resolvable', and the floor is returned so it can be."""
    t = V.paired_sign_test([1.0] * 5, [0.5] * 5)
    assert t["wins_for_b"] == 5
    assert t["p_two_sided"] == pytest.approx(2 / 32)
    assert t["min_p"] == pytest.approx(2 / 32)
    assert V.paired_sign_test([1.0] * 5, [1.0] * 5)["p_two_sided"] == 1.0
    # Ties drop out of the denominator rather than counting as evidence.
    assert V.paired_sign_test([1.0, 1.0], [1.0, 0.5])["min_p"] == pytest.approx(1.0)


def test_the_paired_design_is_actually_paired():
    """Same seed, same initial weights, in both arms -- which is what makes a
    5/5 win a statement about the input transform."""
    a = V.net_for("ve", 3)
    b = V.net_for("vp", 3)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    assert not torch.equal(next(V.net_for("ve", 3).parameters()),
                           next(V.net_for("ve", 4).parameters()))


def test_the_reported_numbers_come_from_the_committed_csv(rows):
    """The report is a pure function of the CSV, so the README's numbers cannot
    drift from the runs without this failing."""
    s = V.report(rows)
    assert set(s["measured"]) == {f"{sc}/{la}" for sc, la in V.cells()}
    m = s["measured"]["native/full"]
    assert m["ve_mean"] == pytest.approx(0.0771, abs=5e-4)
    assert m["vp_mean"] == pytest.approx(0.0619, abs=5e-4)
    assert m["wins_for_b"] == 5
    assert m["p_two_sided"] == pytest.approx(2 / 32)
    assert m["log_mean_ratio_median"] == pytest.approx(0.78, abs=0.02)
    assert math.isclose(m["top_of_ladder_snr"], V.DATA_VAR / 65, rel_tol=1e-9)
