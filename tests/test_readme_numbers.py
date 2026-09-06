"""Every number the README tabulates, recomputed from the committed logs.

The repo already has two structural guards, and neither of them looks at a
number. ``tests/test_reproduction.py`` checks that every artifact the figure
replay reads is tracked by git, so no figure can be built from something that
is not shipped; ``reproduce.sh`` regenerates all fourteen figures from those
artifacts, so no figure can go stale. What neither touches is a number *typed
into a table* -- the README has more than a hundred of them, in fifteen tables
and in the prose around them, every one transcribed by hand from a sweep that
has since been rerun, extended, or refactored.

That gap is not hypothetical. grokking-transformer added the same instrument
to its own README and it found three drifted numbers immediately. Here it
found seven on the first run, and none of them was a typo:

* the schedule section quoted the ancestral sampler's spread at 201
  evaluations as 0.016, which is the *learned*-score arm's spread, while the
  41- and 101-evaluation numbers beside it are the exact-score arm's -- one
  three-number sequence read from two different rows of the sweep;
* the same paragraph's ODE spread at 41 evaluations read 0.0005 for a measured
  0.00043;
* the sampler table's reference row quoted the *other* exact draw's mode width
  (0.401, the reference the arms are scored against) instead of its own
  (0.400), and its sliced W2 read 0.043 for a measured 0.0425;
* the ancestral row started from p_sigma_max read 0.394 for a measured 0.3935;
* "once the ladder is fine enough to resolve a mode it is right to under 1%"
  was true from 100 evaluations on and 2.5% out at 50, which is a column of
  the table it introduces;
* and the last of the three guidance inflations read 9.01x for a measured
  9.02x.

None is a typo and none is large: three are roundings that went up rather
than down, two quote a neighbouring quantity instead of the one in the row,
and two state a claim a little stronger than the sweep supports. That is what
this class of drift looks like -- which is why reading the tables again would
not have caught them and recomputing them did.

Where a claim depends on a convention (which window a standard deviation is
taken over, which arm a spread is read from), the convention is written into
the test, because that is the part a reader cannot recover from the number.

Pure stdlib on purpose -- no torch, no matplotlib -- so this runs anywhere the
rest of the suite does.
"""

import csv
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
README = (ROOT / "README.md").read_text()


# -- reading the README ------------------------------------------------------

def section(heading_starts_with, level=3):
    """The text of the section whose heading starts with the given text."""
    parts = re.split(r"^#{%d} " % level, README, flags=re.M)
    hits = [p for p in parts if p.startswith(heading_starts_with)]
    assert len(hits) == 1, f"{heading_starts_with!r} matched {len(hits)} sections"
    return hits[0]


def plain(cell):
    """A table cell stripped of markdown emphasis, ticks and LaTeX noise."""
    cell = cell.replace("**", "").replace("`", "").strip()
    return cell.replace("$", "").replace("\\", "").strip()


class Table:
    """A markdown table's header cells and its body rows, both as cell lists."""

    def __init__(self, header, body):
        self.header, self.body = header, body

    def __getitem__(self, i):
        return self.body[i]

    def __len__(self):
        return len(self.body)

    def __iter__(self):
        return iter(self.body)


def table(text, index=0):
    """The ``index``-th markdown table in ``text``.

    Several of the README's tables put the quantity being swept in the header
    (noise levels, evaluation budgets, seed counts), so the header is kept
    rather than discarded -- reading a row without checking which column it
    landed in is how a table drifts silently.
    """
    tables, block = [], []
    for line in text.splitlines() + [""]:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = stripped.strip("|").replace("\\|", "\x00").split("|")
            block.append([plain(c.replace("\x00", "|")) for c in cells])
            continue
        rule = [i for i, cells in enumerate(block)
                if all(set(c) <= set("-: ") and c for c in cells)]
        if rule:
            cut = rule[0]
            header = block[cut - 1] if cut else []
            tables.append(Table(header, block[cut + 1:]))
        block = []
    assert len(tables) > index, f"wanted table {index}, found {len(tables)}"
    return tables[index]


def rounds_to(value, text):
    """Does ``value`` print as the README's cell, at the cell's own precision?

    The README writes numbers three ways -- 0.0906, 1.1e-1, 10.5x -- and each
    carries its own claim about precision. Comparing at the README's precision
    is what makes a failure informative: the test fires when the digits shown
    would change, not when a float moves in its last bit.
    """
    text = plain(text).replace("−", "-").replace("–", "-").rstrip("x×")
    if "e" in text.lower() and "e-" in text.lower() or re.search(r"e\+?-?\d", text):
        mantissa, _, exponent = text.lower().partition("e")
        digits = len(mantissa.partition(".")[2])
        return f"{value:.{digits}e}" == f"{float(text):.{digits}e}"
    digits = len(text.partition(".")[2])
    return f"{value:.{digits}f}" == f"{float(text):.{digits}f}"


def prose(text):
    """Section text with its line breaks flattened, so a claim can be asserted
    as the sentence it is rather than as the line-wrapping it happens to have."""
    return re.sub(r"\s+", " ", text)


def scientific(text):
    """A cell written as ``-9\times10^{-6}`` (or ``0.0089``) as a float."""
    text = plain(text).replace("−", "-").replace("–", "-").strip()
    m = re.fullmatch(r"(-?[\d.]+)times10\{?\^?\{?(-?\d+)\}?\}?", text)
    return float(m.group(1)) * 10.0 ** int(m.group(2)) if m else float(text)


def column_number(cell):
    """The number in a header cell that also carries a label -- ``8 levels``,
    ``sigma_{max} 2``, ``NFE 9``."""
    hits = re.findall(r"-?\d+(?:\.\d+)?", plain(cell))
    assert len(hits) == 1, f"{cell!r} does not name exactly one column"
    return float(hits[0])


def check(value, text, what):
    assert rounds_to(value, text), f"{what}: README says {text!r}, logs say {value!r}"


# -- reading the logs --------------------------------------------------------

def rows(name):
    with open(LOGS / name) as f:
        return list(csv.DictReader(f))


def summary(name):
    return json.loads((LOGS / name).read_text())


def floats(row, *keys):
    return [float(row[k]) for k in keys]


# -- score matching ----------------------------------------------------------

def test_the_score_error_table_matches_the_sweep():
    """Five of the thirteen logged levels, three statistics each. The README
    labels its columns with rounded sigmas, so each is matched to the nearest
    logged level -- and the match has to be within 1%, or the label is wrong."""
    logged = rows("score_error.csv")
    body = table(section("What a learned score is actually worth"))
    sigmas = [float(c) for c in body.header[1:]]
    for j, sigma in enumerate(sigmas):
        level = min(logged, key=lambda r: abs(float(r["sigma"]) - sigma))
        assert abs(float(level["sigma"]) - sigma) < 0.02 * max(sigma, 0.05)
        check(float(level["relative_mse"]), body[0][j + 1], f"rel MSE at {sigma}")
        check(float(level["cosine"]), body[1][j + 1], f"cosine at {sigma}")
        check(float(level["cosine_weighted"]), body[2][j + 1], f"weighted at {sigma}")


def test_the_cosine_dip_is_the_near_zero_fraction():
    """The measurement trap: the plain cosine's mid-ladder dip is the fraction
    of draws in the field's near-zero interior, not a defect of the model."""
    level = min(rows("score_error.csv"), key=lambda r: abs(float(r["sigma"]) - 2.25))
    text = section("One measurement trap, recorded")
    check(float(level["cosine"]), "0.955", "dip")
    check(float(level["cosine_weighted"]), "0.997", "weighted at the dip")
    assert f"{round(100 * float(level['near_zero_fraction']))}% of the draws" in text


def test_the_loss_floor_numbers_and_the_window_the_sd_is_taken_over():
    """Section 'The training loss cannot tell you when to stop'.

    The one number here that is not a lookup is the minibatch sd, and it is
    only meaningful with its window attached: taken over the whole run it is
    0.084, because the first 2k steps are still descending. The claim is about
    the flat stretch, so the window starts where the loss flattens.
    """
    s = summary("score_matching.json")
    ev = {int(float(r["step"])): r for r in rows("training_eval.csv")}
    text = section("The training loss cannot tell you when to stop")

    check(s["eval_batch_floor"], "1.203", "the exact floor")
    check(float(ev[60000]["eval_loss"]), "1.212", "well-trained loss")
    check(float(ev[2000]["eval_loss"]), "1.228", "barely-trained loss")

    losses = [float(r["loss"]) for r in rows("training.csv")
              if int(r["step"]) >= 2000]
    sd = statistics.stdev(losses)
    check(sd, "0.073", "minibatch sd from step 2000 on")

    excess_2k = float(ev[2000]["eval_loss"]) - s["eval_batch_floor"]
    excess_60k = float(ev[60000]["eval_loss"]) - s["eval_batch_floor"]
    check(excess_2k, "0.025", "excess risk at 2k")
    check(excess_60k, "0.0093", "excess risk at 60k")
    assert round(sd / excess_60k) == 8, "the sd is eight times the excess"
    assert "eight times" in text

    fall = float(ev[2000]["relative_mse"]) / float(ev[60000]["relative_mse"])
    check(fall, "4.5", "score error over the same stretch")
    assert s["parameters"] == 87298 and "87k parameters" in README


# -- EMA ---------------------------------------------------------------------

EMA_SECTION = "Averaging the weights is worth more than the last 15k steps"


def test_the_ema_table_matches_the_ablation():
    per = summary("ema.json")["per_decay"]
    for row in table(section(EMA_SECTION)):
        d = per[row[0]]
        check(d["raw_mean"], row[1].split("pm")[0], f"raw mean at {row[0]}")
        check(d["raw_sd"], row[1].split("pm")[1], f"raw sd at {row[0]}")
        check(d["ema_mean"], row[2].split("pm")[0], f"EMA mean at {row[0]}")
        check(d["ema_sd"], row[2].split("pm")[1], f"EMA sd at {row[0]}")
        check(d["mean_ratio"], row[3], f"ratio at {row[0]}")
        assert row[4] == f"{d['wins']}/{d['n_seeds']}"


def test_the_ema_prose_and_the_warmup_caveat():
    j = summary("ema.json")
    best = j["per_decay"]["0.999"]
    text = section(EMA_SECTION)

    check(1 / best["mean_ratio"], "6.3", "EMA's advantage in score error")
    check(1 / best["sd_ratio"], "17", "EMA's advantage in seed spread")
    check(100 * best["raw_sd"] / best["raw_mean"], "43", "raw sd as % of its mean")
    check(100 * best["ema_sd"] / best["ema_mean"], "16", "EMA sd as % of its mean")

    finals = [float(r["raw_relative_mse"]) for r in rows("ema_ablation.csv")
              if int(r["step"]) == j["n_steps"] and r["decay"] == "0.999"]
    assert len(finals) == len(j["seeds"])
    check(max(finals), "0.150", "worst raw seed")
    check(min(finals), "0.061", "best raw seed")
    assert max(finals) - min(finals) > best["raw_mean"] - best["ema_mean"]

    assert all(d["crossover_step"] == 500 for d in j["per_decay"].values())
    assert "step 500 of 20,000" in text

    # the caveat: the warmup ramp min(d, (1+t)/(10+t)) reaches d only at
    # t = (10d-1)/(1-d), which is past the end of the run for the top two arms
    for decay, reach in (("0.999", "8,990"), ("0.9995", "17,990")):
        d = float(decay)
        assert f"step {reach}" in text
        assert round((10 * d - 1) / (1 - d)) == int(reach.replace(",", ""))
    d = 0.9999
    assert (10 * d - 1) / (1 - d) > j["n_steps"], "0.9999 must not reach its decay"
    effective = (1 + j["n_steps"]) / (10 + j["n_steps"])
    check(effective, "0.99955", "the decay 0.9999 is actually run at")

    measured_variance_ratio = best["sd_ratio"] ** 2
    check(measured_variance_ratio, "3.4e-3", "measured variance reduction")
    check(best["variance_ratio_bound"], "5e-4", "the i.i.d. bound")
    assert round(measured_variance_ratio / best["variance_ratio_bound"]) == 7
    assert "seven times short of the bound" in prose(text)


# -- sampling ----------------------------------------------------------------

def test_the_sampler_arms_table_matches_the_run():
    logged = rows("sampler_arms.csv")
    body = table(section("Sampling: three errors, taken apart", level=2))
    assert len(body) == len(logged) == 8
    for row, r in zip(body, logged):
        assert row[3] == r["nfe"], f"{r['arm']}: NFE"
        check(float(r["weight_tv"]), row[4], f"{r['arm']}: weight TV")
        check(float(r["mode_sd"]), row[5], f"{r['arm']}: mode sd")
        check(float(r["sw2"]), row[6], f"{r['arm']}: sliced W2")
        # the permutation p lives on a 1/200 grid, so several of its values
        # (0.725, 0.235, 0.245) sit exactly on the boundary of the two decimals
        # the README shows -- which digit appears there is a rounding
        # convention, not a measurement, so this column is checked to within
        # half of its last shown digit rather than reformatted
        for column, key in ((7, "mmd_p_wide"), (8, "mmd_p_fine")):
            quoted = float(plain(row[column]))
            step = 10.0 ** -len(plain(row[column]).partition(".")[2])
            assert abs(quoted - float(r[key])) <= step / 2 + 1e-12, \
                f"{r['arm']}: {key}: README says {row[column]!r}, logs say {r[key]}"


def test_the_floor_is_the_reference_row_and_not_zero():
    j = summary("sampling.json")
    text = section("Sampling: three errors, taken apart", level=2)
    check(j["sw2_floor_mean"], "0.066", "sliced W2 floor")
    check(j["sw2_floor_sd"], "0.020", "floor sd")
    check(min(j["sw2_floor_values"]), "0.036", "floor low")
    check(max(j["sw2_floor_values"]), "0.092", "floor high")
    assert len(j["sw2_floor_values"]) == 8 and "eight pairs" in prose(text)
    check(j["reference_mode_sd"], "0.401", "an exact draw's mode sd")
    check(j["bandwidth_median_heuristic"], "5.71",
          "the median-heuristic bandwidth")


def test_the_mode_sd_prediction_table():
    """`ddpm_gaussian_variance` predicts the within-mode width, and the check
    nobody ordered is that it is right to under 1% once the ladder resolves a
    mode -- which is why NFE 20 is in the table and is the one that is off."""
    logged = {int(float(r["nfe"])): r for r in rows("sampler_sweep.csv")}
    body = table(section("Discretization error closes; estimation error does not"))
    budgets = [int(c) for c in body.header[1:]]
    for j, nfe in enumerate(budgets):
        check(float(logged[nfe]["mode_sd_predicted"]), body[0][j + 1],
              f"predicted mode sd at NFE {nfe}")
        check(float(logged[nfe]["mode_sd_exact"]), body[1][j + 1],
              f"measured mode sd at NFE {nfe}")
    def gap(nfe):
        pred, meas = floats(logged[nfe], "mode_sd_predicted", "mode_sd_exact")
        return abs(pred - meas) / meas

    # the claim is graded, because the recursion arrives gradually: it is not
    # inside 1% at NFE 50, which the section said it was until this test ran
    for nfe in (n for n in budgets if n >= 100):
        assert gap(nfe) < 0.01, f"NFE {nfe} should agree to 1%"
    assert 0.02 < gap(50) < 0.03 and 0.07 < gap(20) < 0.08, (gap(50), gap(20))
    text = section("Discretization error closes; estimation error does not")
    assert "7.5% out at NFE 20, 2.5% at 50, then under 1% from 100 on" in prose(text)


def test_the_langevin_arm_degrades_with_budget_and_the_exact_arm_does_not():
    arms = {(r["arm"], r["nfe"]): r for r in rows("sampler_arms.csv")}
    text = section("More compute makes the Langevin sampler *worse*")
    learned_200 = arms[("annealed Langevin, learned score", "200")]
    learned_2000 = arms[("annealed Langevin, learned score", "2000")]
    exact_200 = arms[("annealed Langevin, exact score", "200")]
    exact_2000 = arms[("annealed Langevin, exact score", "2000")]
    check(float(learned_200["sw2"]), "0.274", "learned Langevin at 200")
    check(float(learned_2000["sw2"]), "0.807", "learned Langevin at 2000")
    check(float(learned_2000["weight_tv"]), "0.150", "its mode-weight TV")
    check(float(exact_200["sw2"]), "0.047", "exact Langevin at 200")
    check(float(exact_2000["sw2"]), "0.064", "exact Langevin at 2000")
    assert round(float(learned_2000["sw2"]) / float(learned_200["sw2"])) == 3
    assert "three times the error" in text


# -- the probability-flow ODE ------------------------------------------------

ODE_ROWS = [("ancestral", "exact"), ("pf-ode euler", "exact"),
            ("pf-ode rk4", "exact"), ("ancestral", "learned"),
            ("pf-ode euler", "learned"), ("pf-ode rk4", "learned")]


def ode_sweep():
    out = {}
    for r in rows("ode_sweep.csv"):
        out.setdefault((r["sampler"], r["score"], r["init"]), {})[int(r["budget"])] = r
    return out


def test_the_ode_budget_table_matches_the_sweep():
    sweep = ode_sweep()
    body = table(section("Same budget, three dynamics"))
    budgets = [int(column_number(c)) for c in body.header[1:]]
    assert budgets == [9, 17, 41, 81, 201, 401]
    for row, key in zip(body, ODE_ROWS):
        for j, budget in enumerate(budgets):
            r = sweep[(key[0], key[1], "N(0,s_max^2)")][budget]
            check(float(r["sw2"]), row[j + 1], f"{key} at NFE {budget}")


def test_the_deterministic_sampler_inherits_its_prior():
    """The claim is an equality across budgets, not a small number: a
    deterministic map cannot re-roll which mode a prior draw lands in, so the
    ODE's mode-weight TV is the *same* at every budget, to four decimals."""
    sweep = ode_sweep()
    text = section("A deterministic sampler inherits its prior, permanently")
    tvs = {b: float(r["weight_tv"])
           for b, r in sweep[("pf-ode rk4", "exact", "N(0,s_max^2)")].items()}
    assert len({round(v, 4) for v in tvs.values()}) == 1, tvs
    check(next(iter(tvs.values())), "0.0121", "the ODE's frozen mode-weight TV")

    stochastic = [float(r["weight_tv"])
                  for r in sweep[("ancestral", "exact", "N(0,s_max^2)")].values()]
    check(min(stochastic), "0.004", "the stochastic sampler's TV, low")
    check(max(stochastic), "0.012", "the stochastic sampler's TV, high")

    body = table(section("A deterministic sampler inherits its prior, permanently"))
    for row, init in zip(body, ["N(0,s_max^2)", "p_sigma_max"]):
        r = sweep[("pf-ode rk4", "exact", init)][201]
        check(float(r["weight_tv"]), row[1], f"{init}: TV")
        check(float(r["mode_sd"]), row[2], f"{init}: mode sd")
        check(float(r["sw2"]), row[3], f"{init}: sliced W2")
    assert "0.3932" in text and "flat from 81 evaluations on" in text
    flat = [float(sweep[("pf-ode rk4", "exact", "N(0,s_max^2)")][b]["mode_sd"])
            for b in (81, 201, 401)]
    assert len({round(v, 4) for v in flat}) == 1, flat


def test_the_learned_score_ordering_and_the_mode_weight_medians():
    """The ODE's advantage under a learned score is quoted as a pair of
    medians over the budgets where the claim is made (17 and up), not as a
    single budget's numbers."""
    sweep = ode_sweep()
    text = section("Same budget, three dynamics")
    budgets = [17, 41, 81, 201, 401]
    ode = [float(sweep[("pf-ode rk4", "learned", "N(0,s_max^2)")][b]["sw2"])
           for b in budgets]
    anc = [float(sweep[("ancestral", "learned", "N(0,s_max^2)")][b]["sw2"])
           for b in budgets]
    assert all(o < a for o, a in zip(ode, anc)), "the ODE wins at every budget"
    ode_tv = [float(sweep[("pf-ode rk4", "learned", "N(0,s_max^2)")][b]["weight_tv"])
              for b in budgets]
    anc_tv = [float(sweep[("ancestral", "learned", "N(0,s_max^2)")][b]["weight_tv"])
              for b in budgets]
    check(statistics.median(ode_tv), "0.019", "the ODE's mode-weight TV")
    check(statistics.median(anc_tv), "0.031", "the ancestral sampler's")
    assert "TV 0.019 against 0.031" in text


# -- noise schedules ---------------------------------------------------------

def schedule_sweep():
    out = {}
    for r in rows("schedule_sweep.csv"):
        key = (r["sampler"], r["score"])
        out.setdefault(key, {}).setdefault(int(r["budget"]), {})[r["schedule"]] = r
    return out


def test_the_schedule_table_matches_the_sweep():
    sweep = schedule_sweep()
    body = table(section("The noise schedule cancels"))
    budgets = [int(column_number(c)) for c in body.header[1:]]
    wanted = [("ancestral", "geometric"), ("ancestral", "cosine"),
              ("ancestral", "linear"), ("pf-ode rk4", "geometric"),
              ("pf-ode rk4", "cosine"), ("pf-ode rk4", "linear")]
    for row, (sampler, schedule) in zip(body, wanted):
        for j, budget in enumerate(budgets):
            r = sweep[(sampler, "exact")][budget][schedule]
            check(float(r["sw2"]), row[j + 1],
                  f"{sampler}/{schedule} at NFE {budget}")


def test_the_schedule_spreads_are_read_from_the_arm_they_are_quoted_for():
    """The ODE forgets the schedule and the ancestral sampler does not, and
    both halves are spreads across the three ladders at a fixed budget.

    The trap this test exists for: ``schedules.json``'s ``gap_by_budget`` is
    the worst spread over *every* cell, so at 201 evaluations it reports the
    learned-score ancestral arm (0.016) while at 41 and 101 it reports the
    exact-score one. Quoting the three together as one arm's numbers -- which
    the README did -- silently reads one number from a different row of the
    sweep than the two beside it.
    """
    sweep = schedule_sweep()
    j = summary("schedules.json")
    text = section("The noise schedule cancels")

    def spread(sampler, score, budget):
        cells = sweep[(sampler, score)][budget]
        vals = [float(r["sw2"]) for r in cells.values()]
        return max(vals) - min(vals)

    check(j["seed_spread_sw2"], "0.008", "seed-to-seed spread")
    check(spread("pf-ode rk4", "exact", 9), "0.451", "ODE spread at 9")
    assert round(spread("pf-ode rk4", "exact", 9) / j["seed_spread_sw2"]) == 57
    check(spread("pf-ode rk4", "exact", 17), "0.058", "ODE spread at 17")
    check(spread("pf-ode rk4", "exact", 41), "0.0004", "ODE spread at 41")
    for budget in (101, 201):
        assert spread("pf-ode rk4", "exact", budget) <= 1e-4, budget

    for budget, quoted in ((41, "0.021"), (101, "0.015"), (201, "0.012")):
        check(spread("ancestral", "exact", budget), quoted,
              f"ancestral spread at {budget}")
        assert quoted in text
    # and the ordering that goes with them
    for budget in (41, 101, 201):
        cells = sweep[("ancestral", "exact")][budget]
        worst = max(cells, key=lambda s: float(cells[s]["sw2"]))
        assert worst == "linear", f"at {budget} the worst ladder is {worst}"


def test_the_schedule_is_a_smaller_knob_than_the_score():
    """The section's closing claim, and it compares two different things: a
    schedule swap within the learned-score ODE, against exact-vs-learned at
    the same budget and schedule."""
    sweep = schedule_sweep()
    text = section("The noise schedule cancels")
    for budget in (41, 101, 201):
        cells = sweep[("pf-ode rk4", "learned")][budget]
        vals = [float(r["sw2"]) for r in cells.values()]
        assert max(vals) - min(vals) < 0.0015, (budget, vals)
        learned = float(cells["geometric"]["sw2"])
        exact = float(sweep[("pf-ode rk4", "exact")][budget]["geometric"]["sw2"])
        assert learned - exact > 0.035, (budget, learned, exact)
    check(float(sweep[("pf-ode rk4", "learned")][201]["geometric"]["sw2"])
          - float(sweep[("pf-ode rk4", "exact")][201]["geometric"]["sw2"]),
          "0.037", "what a learned score costs")
    assert "twenty-five times as much" in prose(text)


# -- likelihoods -------------------------------------------------------------

def test_the_likelihood_error_tables():
    logged = rows("ode_likelihood.csv")
    integration = {int(r["nodes"]): r for r in logged if r["kind"] == "integration"}
    hutchinson = {int(r["probes"]): r for r in logged if r["kind"] == "hutchinson"}
    text = section("Likelihoods, checked against a closed form", level=2)

    body = table(text, 0)
    for j, nodes in enumerate(int(c) for c in body.header[1:]):
        check(float(integration[nodes]["mean_abs_err"]), body[0][j + 1],
              f"integration error at {nodes} nodes")
    body = table(text, 1)
    for j, probes in enumerate(int(c) for c in body.header[1:]):
        check(float(hutchinson[probes]["mean_abs_err"]), body[0][j + 1],
              f"Hutchinson error at {probes} probes")

    # the section's point: at 41 nodes the estimator is four orders of
    # magnitude above the integration error it is added to
    ratio = float(hutchinson[32]["mean_abs_err"]) / float(integration[41]["mean_abs_err"])
    assert round(math.log10(ratio)) == 4, ratio
    assert "four orders of magnitude above" in prose(text)


def test_the_kl_table_and_what_a_learned_score_costs():
    j = summary("ode.json")
    body = table(section("What the model's likelihood is worth, in nats"))
    kl = j["kl_nats"]
    names = ["exact score, exact prior", "exact score, Gaussian prior",
             "learned score, Gaussian prior"]
    logq = {r["score"] + r["prior"]: float(r["mean_log_prob"])
            for r in rows("ode_likelihood.csv") if r["kind"] == "kl"}
    for row, name in zip(body, names):
        quoted = scientific(row[2].split("pm")[0])
        assert abs(quoted - kl[name]) < 5e-5, \
            f"KL for {name}: README says {row[2]!r}, logs say {kl[name]!r}"
    assert [float(row[1].replace("−", "-")) for row in body] == \
        [round(v, 4) for v in (logq["exactp_sigma_max"],
                               logq["exactN(0,s_max^2)"],
                               logq["learnedN(0,s_max^2)"])]

    prior_gap = kl["exact score, Gaussian prior"]
    learned_extra = kl["learned score, Gaussian prior"] - prior_gap
    check(prior_gap, "0.0089", "what the Gaussian prior costs")
    check(learned_extra, "0.0585", "what the learned score costs on top")
    assert round(learned_extra / prior_gap) == 7
    assert "about seven times the prior gap" in section(
        "What the model's likelihood is worth, in nats")


# -- classifier-free guidance ------------------------------------------------

def test_the_guidance_table_and_the_target_that_does_not_move():
    logged = {float(r["w"]): r for r in rows("guidance.csv")}
    text = section("Conditioning: what classifier-free guidance actually samples from",
                   level=2)
    shifts = [float(logged[w]["tilt_shift"]) for w in (0.5, 1, 2, 4)]
    for shift in shifts:
        assert f"{shift:.4f}" in text, shift
    floors = [float(r["floor"]) for r in logged.values()]
    assert max(shifts) < min(floors), "the tilt must be below the two-draw floor"

    body = table(text)
    for row in body:
        r = logged[float(row[0])]
        check(float(r["floor"]), row[1], f"floor at w={row[0]}")
        check(float(r["ancestral_sw"]), row[2], f"ancestral at w={row[0]}")
        check(float(r["ancestral_over_floor"]), row[3], f"ancestral ratio")
        check(float(r["langevin_sw"]), row[4], f"Langevin at w={row[0]}")
        check(float(r["langevin_over_floor"]), row[5], f"Langevin ratio")


def test_the_guidance_read_outs_that_do_not_show_a_collapse():
    logged = {float(r["w"]): r for r in rows("guidance.csv")}
    j = summary("guidance.json")
    text = section("Conditioning: what classifier-free guidance actually samples from",
                   level=2)
    assert all(float(r["ancestral_purity"]) >= 0.99995 for r in logged.values())
    assert "1.0000 at every weight" in prose(text)

    widths = [float(logged[w]["ancestral_spread"]) for w in (0, 0.5, 1, 2, 4)]
    assert ", ".join(f"{v:.3f}" for v in widths) in text
    check(float(logged[4.0]["spread_target"]), "0.399", "q_w's own width")
    assert min(widths) / widths[0] > 0.94, "a 5% dip, not a collapse"
    assert widths[-1] > min(widths), "and it comes back"

    check(float(logged[0.0]["ancestral_radial"]), "-0.012", "radial at w=0")
    check(float(logged[4.0]["ancestral_radial"]), "0.181", "radial at w=4")
    check(float(logged[0.0]["radial_target"]), "0.007", "q_w's own radial offset")
    check(float(logged[4.0]["ancestral_radial"]) / float(logged[4.0]["spread_target"]),
          "0.45", "the radial offset in component sd")

    control = j["equilibration_control"]
    check(control["standard"], "0.0153", "Langevin at its standard budget")
    check(control["doubled"], "0.0116", "Langevin at twice the steps")
    assert control["doubled"] < float(logged[4.0]["floor"]), "doubled must beat the floor"


def test_the_overshoot_law_is_reported_as_fitted():
    body = table(section("Conditioning: what classifier-free guidance actually samples from",
                         level=2), 1)
    logged = {float(r["w"]): r for r in rows("guidance_overshoot.csv")}
    for j, w in enumerate(float(c) for c in body.header[1:]):
        check(float(logged[w]["radial_offset"]), body[0][j + 1], f"offset at w={w}")
        if body[1][j + 1] != "—":
            check(float(logged[w]["log_law"]), body[1][j + 1], f"law at w={w}")
            check(float(logged[w]["ratio"]), body[2][j + 1], f"ratio at w={w}")
    within = max(abs(float(r["ratio"]) - 1) for w, r in logged.items() if w <= 8)
    assert round(100 * within) == 2, within
    assert round(100 * abs(float(logged[16.0]["ratio"]) - 1)) == 5
    assert "The log law is fitted, not derived" in README


# -- guidance where the components overlap -----------------------------------

def test_the_overlap_sweep_is_the_tilt_becoming_available():
    logged = rows("overlap.csv")
    text = section("The other half: what guidance does once the components overlap")
    by_sep = {}
    for r in logged:
        by_sep.setdefault(float(r["separation_sd"]), {})[float(r["w"])] = r
    seps = sorted(by_sep)

    tilts = [float(by_sep[s][0.0]["sd_log_tilt"]) for s in seps]
    check(min(tilts), "0.049", "sd of the log tilt, separated")
    check(max(tilts), "0.816", "sd of the log tilt, most overlapping")

    widest = by_sep[seps[0]]
    check(float(widest[4.0]["distance_to_q0"]) / float(widest[0.0]["floor"]),
          "41", "how far q_w moves at the most overlapping cell")

    displacements = [float(r["displacement"]) for r in logged]
    check(max(displacements), "0.90", "the largest mean displacement")
    spreads = [float(r["spread"]) for r in logged]
    check(min(spreads), "0.83", "the narrowest q_w gets")
    assert round(100 * (1 - min(spreads))) == 17, "17% at best"
    for sep in seps[:2]:                     # the two most overlapping cells
        widths = [float(by_sep[sep][w]["spread"]) for w in (0.0, 1.0, 2.0, 4.0)]
        assert widths[-1] > min(widths), f"{sep}: the narrowing must reverse"
    assert "then reversing" in README

    purities = [float(by_sep[seps[0]][w]["purity"]) for w in (0.0, 4.0)]
    check(purities[0], "0.558", "purity at w=0, most overlapping")
    check(purities[1], "0.953", "purity at w=4, most overlapping")
    assert "0.558 -> 0.953" in text.replace("$\\to$", "->").replace("→", "->")


def test_the_overlap_sampler_table_is_measured_past_the_target():
    """The table's two overshoot columns are landing points *minus* q_w's own,
    in component sd -- the correction that keeps a sampler drifting outward
    from being credited to a target that moved out to meet it."""
    logged = {(float(r["std"]), float(r["w"])): r
              for r in rows("overlap_samplers.csv")}
    body = table(section("And the samplers there: ancestral still overshoots"))
    for row in body:
        sep, w = float(row[0]), float(row[1])
        r = next(v for (std, weight), v in logged.items()
                 if weight == w and abs(float(v["separation_sd"]) - sep) < 0.01)
        std = float(r["std"])
        anc = (float(r["ancestral_radial"]) - float(r["target_radial"])) / std
        lan = (float(r["langevin_radial"]) - float(r["target_radial"])) / std
        check(anc, row[2], f"ancestral past q_w at sep {sep}, w={w}")
        check(lan, row[3], f"Langevin past q_w at sep {sep}, w={w}")
        check(float(r["ancestral_ratio"]), row[4], f"ancestral/floor at {sep}, {w}")
        check(float(r["langevin_ratio"]), row[5], f"Langevin/floor at {sep}, {w}")
        assert abs(lan) <= 0.02, "Langevin arrives in every cell"

    # dividing out the w=0 control, which is not a fixed baseline
    inflations = []
    for std in sorted({k[0] for k in logged}):
        base = float(logged[(std, 0.0)]["ancestral_ratio"])
        inflations.append(float(logged[(std, 4.0)]["ancestral_ratio"]) / base)
    text = section("And the samplers there: ancestral still overshoots")
    assert ", ".join(f"${v:.2f}\\times$" for v in inflations) in prose(text)
    assert max(inflations) / min(inflations) < 1.4, "flat across the sweep"


# -- VP vs VE ----------------------------------------------------------------

def test_the_vp_ve_difference_table_resolves_nothing():
    logged = {}
    for r in rows("vp_ve_sweep.csv"):
        key = (r["target"], r["score"])
        logged.setdefault(key, {}).setdefault(float(r["sigma_max"]), []).append(r)
    body = table(section("VP and VE are one chain in two coordinate systems"), 1)
    wanted = [("ring_standardized", "exact"), ("ring", "exact"), ("ring", "learned")]
    sigmas = [column_number(c) for c in body.header[1:]]
    for row, key in zip(body, wanted):
        for j, sigma in enumerate(sigmas):
            draws = logged[key][sigma]
            assert len(draws) == 3, f"{key} at {sigma}: {len(draws)} prior draws"
            gap = statistics.mean(float(r["vp_sw2"]) - float(r["ve_sw2"])
                                  for r in draws)
            check(gap, row[j + 1], f"{key} at sigma_max={sigma}")

    floors = summary("vp_ve.json")["floors"]
    check(floors["ring_standardized"], "0.023", "standardized floor")
    check(floors["ring"], "0.066", "ring floor")
    worst = max(abs(statistics.mean(float(r["vp_sw2"]) - float(r["ve_sw2"])
                                    for r in draws))
                for cells in logged.values() for draws in cells.values())
    assert round(100 * worst / floors["ring"]) == 13, "13% of the floor"
    assert "13% of the floor" in prose(section(
        "VP and VE are one chain in two coordinate systems"))


def test_the_prior_accounting_table():
    logged = {(float(r["sigma_max"]), int(r["n_levels"])): r
              for r in rows("vp_ve_tradeoff.csv")}
    body = table(section("VP and VE are one chain in two coordinate systems"), 2)
    levels = [int(column_number(c)) for c in body.header[1:]]
    for row, (sigma, field) in zip(body, [(3.0, "prior_contribution"),
                                              (3.0, "ladder_deficit"),
                                              (8.0, "prior_contribution"),
                                              (8.0, "ladder_deficit")]):
        for j, n in enumerate(levels):
            check(float(logged[(sigma, n)][field]), row[j + 1],
                  f"{field} at sigma_max={sigma}, {n} levels")
    # the prior's contribution does not depend on the ladder -- that is the claim
    for sigma in (3.0, 8.0):
        vals = {round(float(logged[(sigma, n)]["prior_contribution"]), 5)
                for n in levels}
        assert len(vals) == 1, (sigma, vals)


def test_the_pathwise_agreement_and_the_prior_ratios():
    j = summary("vp_ve.json")
    text = section("VP and VE are one chain in two coordinate systems")
    check(j["pathwise"]["max_abs_gap"], "2e-4", "float32 pathwise gap")
    check(j["pathwise"]["state_scale"], "5.2", "the states it is measured on")
    v = j["ring_var"]
    check(v, "8.16", "the ring's per-coordinate variance")
    # the prior ratios at sigma_max = 2, quoted in the section's prose
    ve = 2.0 ** 2 / (v + 2.0 ** 2)
    alpha_bar = 1 / (1 + 2.0 ** 2)
    vp = 1 / (alpha_bar * v + (1 - alpha_bar))
    check(ve, "0.329", "VE's prior ratio at sigma_max=2")
    check(vp, "0.411", "VP's prior ratio at sigma_max=2")
    assert "0.411 against VE's 0.329" in text


# -- training in VP coordinates ----------------------------------------------

VP_COORDS = "Training in VP coordinates"


def test_the_input_span_table_is_the_premise_being_checked():
    predicted = {(p["scale"], p["ladder"], p["coords"]): p
                 for p in summary("vp_coords.json")["predicted"]}
    body = table(section(VP_COORDS))
    wanted = [("native", "ve"), ("native", "vp"), ("unit", "ve"), ("unit", "vp")]
    for row, (scale, coords) in zip(body, wanted):
        p = predicted[(scale, "full", coords)]
        lo, hi = [s.strip() for s in row[1].replace("→", "->").split("->")]
        check(p["sd_at_sigma_min"], lo, f"{scale}/{coords}: sd at sigma_min")
        check(p["sd_at_sigma_max"], hi, f"{scale}/{coords}: sd at sigma_max")
        check(p["span"], row[2], f"{scale}/{coords}: span")
    assert round(predicted[("unit", "full", "vp")]["span"], 6) == 1.0


def test_the_vp_coords_tables_match_the_paired_runs():
    measured = summary("vp_coords.json")["measured"]
    cells = ["native/full", "unit/full", "native/short", "native/shallow"]

    aggregate = table(section(VP_COORDS), 1)
    for row, key in zip(aggregate, cells):
        m = measured[key]
        for value, text in ((m["ve_mean"], row[1]), (m["vp_mean"], row[2])):
            body, _, rng = text.partition("[")
            check(value, body, f"{key}: mean")
            lo, hi = rng.rstrip("]").replace("–", "-").split("-")
            source = "ve" if text is row[1] else "vp"
            check(m[f"{source}_min"], lo, f"{key}: {source} min")
            check(m[f"{source}_max"], hi, f"{key}: {source} max")
        check(m["ratio_median"], row[3], f"{key}: VP/VE")
        assert row[4] == f"{int(m['wins_for_b'])}/{int(m['n'])}", key

    per_level = table(section(VP_COORDS), 2)
    for row, key in zip(per_level, cells):
        m = measured[key]
        check(m["log_mean_ratio_median"], row[1], f"{key}: ladder-mean ratio")
        assert row[2] == f"{m['log_mean_ratio_wins']}/{int(m['n'])}", key
        check(m["coarsest_ratio_median"], row[3], f"{key}: coarsest level")
        assert row[4] == f"{m['coarsest_wins']}/{int(m['n'])}", key

    bands = table(section(VP_COORDS), 3)
    for row, key in zip(bands, cells):
        m = measured[key]
        for text, band in ((row[1], "shared_band"), (row[2], "deep_band")):
            if m[band] is None:
                assert "does not reach" in text, key
                continue
            b = m[band]
            head, _, rest = text.partition("[")
            check(b["median"], head, f"{key}/{band}: median")
            lo, hi = rest.partition("]")[0].replace("–", "-").split("-")
            check(b["min"], lo, f"{key}/{band}: min")
            check(b["max"], hi, f"{key}/{band}: max")
            assert rest.partition("]")[2].strip(", ") == \
                f"{b['wins']}/{int(m['n'])}", (key, band)


def test_the_vp_coords_conclusions_follow_from_those_tables():
    measured = summary("vp_coords.json")["measured"]
    text = section(VP_COORDS)
    # standardizing gives the O(1) ideal and is where VP helps least
    deep = {k: measured[k]["deep_band"]["median"]
            for k in ("native/full", "unit/full", "native/short")}
    assert max(deep, key=deep.get) == "unit/full", deep
    check(min(deep.values()), "0.68", "the best deep-band gain")
    check(max(deep.values()), "0.78", "the worst deep-band gain")
    assert "0.68-0.78x" in README.replace("–", "-").replace("$", "")

    shared = [measured[k]["shared_band"]["median"] for k in measured]
    check(min(shared), "0.92", "the shared band, low")
    check(max(shared), "0.99", "the shared band, high")
    assert "0.92-0.99x" in text.replace("–", "-").replace("$", "")

    coarse = {k: measured[k]["coarsest_ratio_median"] for k in ("native/full", "unit/full")}
    assert abs(coarse["native/full"] - coarse["unit/full"]) < 0.01
    assert "0.54 against 0.53" in text
    # five paired seeds cannot resolve below 2/2^5
    assert all(m["min_p"] == 0.0625 for m in measured.values())
    assert "2/2^5 = 0.0625" in text.replace("$", "").replace("{", "").replace("}", "")
    snr = measured["native/full"]["top_of_ladder_snr"]
    check(snr, "0.126", "the top-of-ladder identity's error")


# --- the counts the documents quote about the repo itself -----------------
#
# Every test above recomputes a measurement from a committed sweep. These two
# do not: they check what the README and reproduce.sh say about *this
# repository* -- how many tests there are, how many figures ship -- which is
# the one class of claim no instrument in this series was looking at.
# gp-from-scratch's Reproduce block had drifted to 340 tests against a real
# 366, unnoticed until someone ran the suite from a fresh clone and read the
# total off the bottom.

REPRODUCE = (ROOT / "reproduce.sh").read_text()


def _whole_suite(config) -> bool:
    """True when this run collected everything, so a total means something.

    ``pytest tests/test_ode.py`` or ``-k foo`` collects a subset; asserting a
    total against that would fail for a reason unrelated to the README. With no
    path argument pytest records the rootdir as its one arg.
    """
    if config.option.keyword or config.option.markexpr:
        return False
    return [Path(a).resolve() for a in config.args] == [ROOT]


def test_the_documents_quote_the_number_of_tests_there_are(request):
    claimed = {int(n) for n in re.findall(r"reproduce\.sh\s+# (\d+) tests", README + REPRODUCE)}
    assert claimed, "neither document quotes a test count any more"
    if not _whole_suite(request.config):
        pytest.skip("a partial run's count says nothing about the README's")
    collected = request.session.testscollected
    assert claimed == {collected}, (
        f"the documents claim {sorted(claimed)} tests, this run collected {collected}"
    )


def test_the_documents_agree_on_how_many_figures_ship():
    tracked = subprocess.run(
        ["git", "ls-files", "figures/*.png"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    claimed = {
        int(n)
        for n in re.findall(r"all (\d+) (?:committed )?figures", README + REPRODUCE)
    }
    assert claimed, "neither document states a figure count any more"
    assert claimed == {len(tracked)}, (
        f"the documents claim {sorted(claimed)} figures, git tracks {len(tracked)}"
    )
