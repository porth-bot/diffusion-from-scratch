r"""Training a network in VP coordinates: the last piece of the VP/VE question.

`experiments/vp_ve.py` settled the process and the prior. VP and VE are one
chain in two coordinate systems, agreeing pathwise to 1e-11, so nothing about
the *process* is a modelling choice; and VP's conventional ``N(0, I)`` prior is
exactly right on standardized data and wrong otherwise, worth one to two orders
of magnitude less than a halving of the step size. The README's remaining
caveat named the third thing, which is the one the literature actually argues
from: **nothing here trains a model in VP coordinates**, which is where the
O(1)-input argument for VP would be tested.

This runs it, and the design is unusually tight because of Eq. (2) of
``diffusion/vp.py``: the VP state is the VE state times ``sqrt(abar)``, a
per-level scalar, and the regression target ``eps`` is the *same draw* in both
coordinate systems. So "train in VP coordinates" is exactly one multiplication
at the network's input (``diffusion.vp.VPInputScoreNet``), and the two arms can
share their data stream, their targets, and their initialization. Same seed in
both arms means the same weights at step 0 and the same batches throughout --
a paired comparison of one input transform, not two runs that differ by
everything.

The prediction, written down first
----------------------------------
The argument is about scale, so it is checkable in closed form
(``diffusion.vp.input_sd``). For data of per-coordinate variance ``v``, the
network's input has per-coordinate sd ``sqrt(v + sigma^2)`` in VE coordinates
and ``sqrt((v + sigma^2)/(1 + sigma^2))`` in VP. VP is O(1) *at every level*
only when ``v = 1``, where it is exactly 1.0 everywhere. This repo's ring has
``v = 8.16``, and there the ranges are:

    VE   2.857 -> 8.495   (a factor of 2.97)
    VP   2.853 -> 1.054   (a factor of 2.71)

-- the same span, pointed the other way. So the prediction is **no advantage on
the native target**, and an advantage only on standardized data, where VP's
input sd is constant by construction. Both are run.

Standardizing has to be done without changing the problem, or the two scales
are not comparable. Dividing the target by its own sd and dividing the whole
sigma ladder by the same factor is an exact rescaling: identical SNR at every
rung, identical mixture geometry in units of its own width. What changes is
only the absolute scale of what the network sees -- which is the variable under
study.

Artifacts and cost. Each cell is one training run; the four cells x five seeds
are cached one row per run in ``logs/vp_coords.csv`` and only missing rows are
trained, so an interrupted pass resumes (a run is ~2 minutes on CPU here and
the pass is ~35). The figure and every number replay from the CSV.

Run:  python experiments/vp_coords.py            (table + figure from the CSV)
      python experiments/vp_coords.py --figures  (the figure alone, no table)
      python experiments/vp_coords.py --train    (fill in missing runs)
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from diffusion.dsm import score_diagnostics, train_score_model  # noqa: E402
from diffusion.langevin import geometric_sigmas  # noqa: E402
from diffusion.model import ScoreNet  # noqa: E402
from diffusion.targets import GaussianMixture  # noqa: E402
from diffusion.vp import VPInputScoreNet, input_sd  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"

# Matched to experiments/score_matching.py so the numbers sit on the repo's
# existing scale, except for the step budget: this is a paired comparison of a
# single input transform, and 8k steps is where the EMA ablation showed the
# arms of a paired comparison already separated.
SIGMA_MIN, SIGMA_MAX = 0.05, 8.0
WIDTH, DEPTH = 128, 4
N_STEPS, BATCH, LR = 8_000, 512, 2e-3
EVAL_EVERY, EVAL_POINTS = 1_000, 4_096
N_EVAL_LEVELS = 13
SEEDS = (0, 1, 2, 3, 4)
COORDS = ("ve", "vp")
SCALES = ("native", "unit")
# Ladders, as (sigma_min, sigma_max) multipliers on the base range. The short
# ladder exists to separate two explanations of whatever the full one shows:
# it keeps the coarse end (and therefore the largest input-scale difference
# between the two coordinate systems) exactly where it was, and cuts the
# *range of sigma-dependent gain* the network has to implement from 160x to
# 16x. An effect that is about the coarse end survives it; an effect that is
# about spanning the ladder does not.
#
# The shallow ladder is the control the other three cannot be, and it exists
# because of a flaw in this design that only shows up once it is written out.
# Rescaling the data and the sigma ladder together (the "unit" arm) holds the
# signal-to-noise ratio fixed at every rung *by construction*, and the short
# ladder does not touch the coarse end at all -- so all three hold the SNR at
# the top of the ladder, v / (1 + sigma_max^2), at the same value. Any effect
# that tracks *that* would look invariant across all three and be mistaken for
# universal. The shallow ladder stops at sigma_max / 4, which moves the
# top-of-ladder SNR by 13x with the data untouched.
#
# That is how the cells were designed and it is *not* how they should be read.
# A ladder that stops early does not only change its top-of-ladder SNR, it
# stops covering the noise levels above it -- so the four whole-ladder numbers
# average four different bands. Scored on the band all four cover, the shallow
# cell is indistinguishable from the rest; scored above it, everything that
# reaches drops. See `report`'s last table: the variable is the noise level,
# and the ladder enters only through how much of it gets averaged.
LADDERS = {"full": (1.0, 1.0), "short": (10.0, 1.0), "shallow": (1.0, 0.25)}

RING = dict(n_components=8, radius=4.0, std=0.4)
# Per-coordinate variance of the ring: r^2/2 + s^2, exactly (the modes are
# equally spaced on the circle, so the angular average of cos^2 is 1/2).
DATA_VAR = RING["radius"] ** 2 / 2 + RING["std"] ** 2
DATA_SD = math.sqrt(DATA_VAR)


def target(scale: str) -> GaussianMixture:
    """The ring, natively or divided by its own standard deviation."""
    c = 1.0 if scale == "native" else 1.0 / DATA_SD
    return GaussianMixture.ring(n_components=RING["n_components"],
                                radius=RING["radius"] * c,
                                std=RING["std"] * c)


def sigma_range(scale: str, ladder: str = "full") -> tuple[float, float]:
    """The ladder, rescaled with the data so the SNR at every rung is fixed."""
    c = 1.0 if scale == "native" else 1.0 / DATA_SD
    lo_mult, hi_mult = LADDERS[ladder]
    return SIGMA_MIN * c * lo_mult, SIGMA_MAX * c * hi_mult


def data_var(scale: str) -> float:
    return DATA_VAR if scale == "native" else 1.0


def eval_sigmas(scale: str, ladder: str = "full") -> list[float]:
    lo, hi = sigma_range(scale, ladder)
    return sorted(geometric_sigmas(hi, lo, N_EVAL_LEVELS))


def net_for(coords: str, seed: int) -> ScoreNet:
    """A fresh network of the requested coordinate convention.

    ``torch.manual_seed`` before construction is what makes the pairing real:
    the VE and VP arms of one seed start from *the same weights*, so a
    difference between them at the end is the input transform and not the draw.
    """
    torch.manual_seed(seed)
    cls = ScoreNet if coords == "ve" else VPInputScoreNet
    return cls(dim=2, width=WIDTH, depth=DEPTH)


def cells() -> list[tuple[str, str]]:
    """``(scale, ladder)`` pairs. The short ladder is a control, run on the
    native target only: its job is to vary the ladder length with the coarse
    end held fixed, and doing that twice would double the cost for nothing."""
    return [("native", "full"), ("unit", "full"), ("native", "short"),
            ("native", "shallow")]


def top_of_ladder_snr(scale: str, ladder: str) -> float:
    """``v / (1 + sigma_max^2)``: how much signal survives at the coarsest rung.

    Also, exactly, the relative error of the *identity map* in VP coordinates
    there -- ``x_vp - eps = sqrt(abar) x_0 + (sqrt(1-abar) - 1) eps``, whose
    leading term has squared norm ``abar v`` against ``E||eps||^2 = 1`` per
    coordinate. So this number says how close a VP network's coarse-end input
    already is to its own regression target, before any training.
    """
    _, hi = sigma_range(scale, ladder)
    return data_var(scale) / (1.0 + hi * hi)


# The three deep cells reach sigma / sd = 2.80 and the shallow one stops at
# 0.70, so comparing whole ladders compares different *bands* of noise level.
# That is a confound in the design, not a result, and the split below is what
# removes it: every cell also gets scored on the band all four cover.
SHARED_BAND = 0.71


def normalized_sigmas(scale: str, ladder: str) -> list[float]:
    """The ladder in units of the data's own standard deviation.

    The native and unit targets differ by ``sqrt(v)`` in *both* the data and
    the ladder, so this is the only scale on which a level in one cell is the
    same thing as a level in another -- and it is what makes the four cells
    comparable level by level instead of only cell by cell.
    """
    sd = math.sqrt(data_var(scale))
    return [s / sd for s in eval_sigmas(scale, ladder)]


# -- the closed-form prediction ---------------------------------------------

def predicted_input_spans() -> list[dict[str, float]]:
    """Input-sd range of each cell, before any training happens."""
    rows = []
    for scale, ladder in cells():
        levels = torch.tensor(eval_sigmas(scale, ladder))
        for coords in COORDS:
            sd = input_sd(levels, data_var(scale), coords)
            # Reported at the two ends of the ladder rather than as min/max,
            # so the *direction* is visible: VE grows with sigma and VP shrinks,
            # which is why their spans can match while the argument for VP
            # sounds like it should not allow that.
            rows.append({"scale": scale, "ladder": ladder, "coords": coords,
                         "sd_at_sigma_min": float(sd[0]),
                         "sd_at_sigma_max": float(sd[-1]),
                         "span": float(sd.max() / sd.min())})
    return rows


# -- training ----------------------------------------------------------------

CSV = LOGS / "vp_coords.csv"


def columns() -> list[str]:
    return (["coords", "scale", "ladder", "seed", "relative_mse", "cosine",
             "seconds"] + [f"rel_mse_L{i}" for i in range(N_EVAL_LEVELS)])


def load_rows() -> dict[tuple[str, str, str, int], dict]:
    if not CSV.exists():
        return {}
    with CSV.open(newline="") as f:
        out = {}
        for r in csv.DictReader(f):
            row = dict(r)
            row["seed"] = int(row["seed"])
            for k in columns()[4:]:
                row[k] = float(row[k])
            out[(row["coords"], row["scale"], row["ladder"], row["seed"])] = row
    return out


def write_rows(rows: dict[tuple[str, str, str, int], dict]) -> None:
    LOGS.mkdir(exist_ok=True)
    ordered = [rows[(c, s, ell, d)]
               for s, ell in cells() for c in COORDS for d in SEEDS
               if (c, s, ell, d) in rows]
    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns())
        w.writeheader()
        w.writerows(ordered)


def train_missing(budget_seconds: float = float("inf")) -> None:
    """Train whatever the CSV lacks, stopping when the budget is spent.

    The budget exists because this host cannot hold a foreground process for
    long, and a killed pass that has written nothing has done nothing. Each
    finished run is flushed immediately, so the work survives the interruption
    and the next call picks up where this one stopped.
    """
    import time

    rows = load_rows()
    t0 = time.perf_counter()
    for scale, ladder in cells():
        p, (lo, hi) = target(scale), sigma_range(scale, ladder)
        levels = eval_sigmas(scale, ladder)
        for coords in COORDS:
            for seed in SEEDS:
                if (coords, scale, ladder, seed) in rows:
                    continue
                if time.perf_counter() - t0 > budget_seconds:
                    print(f"budget spent; {len(rows)} of {n_runs()} runs done",
                          flush=True)
                    return
                net = net_for(coords, seed)
                hist = train_score_model(
                    net, p, lo, hi, n_steps=N_STEPS, batch_size=BATCH, lr=LR,
                    seed=seed, log_every=2000, eval_every=EVAL_EVERY,
                    eval_sigmas=[lo, math.sqrt(lo * hi), hi],
                    eval_points=EVAL_POINTS,
                )
                diag = score_diagnostics(
                    net, p, levels, n_points=EVAL_POINTS,
                    generator=torch.Generator().manual_seed(12345),
                )
                n = len(diag.relative_mse)
                row = {"coords": coords, "scale": scale, "ladder": ladder,
                       "seed": seed,
                       "relative_mse": sum(diag.relative_mse) / n,
                       "cosine": sum(diag.cosine) / n,
                       "seconds": hist.seconds}
                row.update({f"rel_mse_L{i}": v
                            for i, v in enumerate(diag.relative_mse)})
                rows[(coords, scale, ladder, seed)] = row
                write_rows(rows)
                print(f"{scale:7s} {ladder:5s} {coords}  seed {seed}: "
                      f"relative mse {row['relative_mse']:.5f}  "
                      f"({hist.seconds:.0f}s)", flush=True)


# -- reporting ---------------------------------------------------------------

def n_runs() -> int:
    return len(cells()) * len(COORDS) * len(SEEDS)


def arm(rows: dict, coords: str, scale: str, ladder: str = "full",
        field: str = "relative_mse") -> list[float]:
    return [rows[(coords, scale, ladder, s)][field] for s in SEEDS
            if (coords, scale, ladder, s) in rows]


def log_mean_ratios(rows: dict, scale: str, ladder: str,
                    lo: float = 0.0, hi: float = float("inf")) -> list[float]:
    """Per-seed geometric mean of the per-level VP/VE ratio.

    The summary the aggregate cannot be: mean relative MSE over the ladder is
    47% one level, because the error at the finest sigma is 70x the error at
    the coarsest. Averaging the *log ratio* gives every level equal weight,
    which is what "does the coordinate change help across the ladder" means.

    ``lo`` and ``hi`` restrict it to a band of *normalized* noise level
    (``sigma / sd``, see :func:`normalized_sigmas`). The default is the whole
    ladder; the band is what compares cells whose ladders do not cover the
    same range of noise, which three of these four do not.
    """
    idx = [i for i, x in enumerate(normalized_sigmas(scale, ladder))
           if lo < x <= hi]
    out = []
    for seed in SEEDS:
        key_ve, key_vp = ("ve", scale, ladder, seed), ("vp", scale, ladder, seed)
        if key_ve not in rows or key_vp not in rows or not idx:
            continue
        logs = [math.log(rows[key_vp][f"rel_mse_L{i}"]
                         / rows[key_ve][f"rel_mse_L{i}"])
                for i in idx]
        out.append(math.exp(sum(logs) / len(logs)))
    return out


def band_summary(rows: dict, scale: str, ladder: str,
                 lo: float, hi: float) -> dict | None:
    """Median and seed range of the VP/VE ratio inside one band, or None if
    this cell's ladder does not reach into it."""
    g = sorted(log_mean_ratios(rows, scale, ladder, lo, hi))
    if not g:
        return None
    idx = [i for i, x in enumerate(normalized_sigmas(scale, ladder))
           if lo < x <= hi]
    return {"levels": len(idx), "median": g[len(g) // 2],
            "min": g[0], "max": g[-1], "wins": sum(1 for x in g if x < 1)}


def paired_sign_test(a: list[float], b: list[float]) -> dict[str, float]:
    """Two-sided sign test on paired differences, exact.

    The arms share seed, initialization and data stream, so the comparison is
    paired and the question is only how many seeds move which way. With five
    pairs a clean sweep is p = 2/32 = 0.0625, so ``min_p`` is reported next to
    ``p`` -- five paired seeds *cannot* reach 0.05, and reading a
    non-significant result as "no effect" would be reading the sample size.
    """
    from math import comb

    n = len(a)
    wins = sum(1 for x, y in zip(a, b) if y < x)      # b better than a
    ties = sum(1 for x, y in zip(a, b) if y == x)
    k = min(wins, n - ties - wins)
    tail = sum(comb(n - ties, i) for i in range(k + 1)) / 2 ** (n - ties)
    return {"n": float(n), "wins_for_b": float(wins),
            "p_two_sided": min(1.0, 2 * tail), "min_p": 2.0 / 2 ** (n - ties)}


def report(rows: dict) -> dict:
    print("\nPredicted input scale (closed form, before any training):")
    print(f"  {'target':8s} {'ladder':7s} {'coords':6s} "
          f"{'input sd over the ladder':30s} span")
    for r in predicted_input_spans():
        span = "{:.3f} -> {:.3f}".format(r["sd_at_sigma_min"],
                                         r["sd_at_sigma_max"])
        print(f"  {r['scale']:8s} {r['ladder']:7s} {r['coords']:6s} "
              f"{span:30s} {r['span']:.2f}x")

    summary: dict = {"n_steps": N_STEPS, "seeds": list(SEEDS),
                     "sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX,
                     "ladders": {k: list(v) for k, v in LADDERS.items()},
                     "width": WIDTH, "depth": DEPTH, "batch_size": BATCH,
                     "lr": LR, "data_var": DATA_VAR,
                     "predicted": predicted_input_spans(), "measured": {}}

    print(f"\nMean relative MSE over {N_EVAL_LEVELS} levels, "
          f"after {N_STEPS:,} steps:")
    print(f"  {'target':8s} {'ladder':7s} {'VE':24s} {'VP':24s} "
          f"{'VP/VE':>7} {'VP wins':>8} {'p':>7}")
    for scale, ladder in cells():
        ve, vp = arm(rows, "ve", scale, ladder), arm(rows, "vp", scale, ladder)
        if len(ve) != len(SEEDS) or len(vp) != len(SEEDS):
            print(f"  {scale:8s} {ladder:7s} incomplete "
                  f"({len(ve)} VE, {len(vp)} VP)")
            continue
        t = paired_sign_test(ve, vp)
        ratios = sorted(b / a for a, b in zip(ve, vp))
        gm = sorted(log_mean_ratios(rows, scale, ladder))
        coarse_ve = arm(rows, "ve", scale, ladder, f"rel_mse_L{N_EVAL_LEVELS - 1}")
        coarse_vp = arm(rows, "vp", scale, ladder, f"rel_mse_L{N_EVAL_LEVELS - 1}")
        coarse = sorted(b / a for a, b in zip(coarse_ve, coarse_vp))
        cell = {
            "ve_mean": sum(ve) / len(ve), "ve_min": min(ve), "ve_max": max(ve),
            "vp_mean": sum(vp) / len(vp), "vp_min": min(vp), "vp_max": max(vp),
            "ratio_median": ratios[len(ratios) // 2],
            "log_mean_ratio_median": gm[len(gm) // 2],
            "log_mean_ratio_wins": sum(1 for g in gm if g < 1),
            "top_of_ladder_snr": top_of_ladder_snr(scale, ladder),
            "coarsest_ratio_median": coarse[len(coarse) // 2],
            "coarsest_wins": sum(1 for r in coarse if r < 1),
            "shared_band": band_summary(rows, scale, ladder, 0.0, SHARED_BAND),
            "deep_band": band_summary(rows, scale, ladder, SHARED_BAND,
                                      float("inf")),
            "sigma_max_normalized": normalized_sigmas(scale, ladder)[-1], **t,
        }
        summary["measured"][f"{scale}/{ladder}"] = cell
        ve_cell = "{:.4f} [{:.4f}-{:.4f}]".format(cell["ve_mean"], min(ve), max(ve))
        vp_cell = "{:.4f} [{:.4f}-{:.4f}]".format(cell["vp_mean"], min(vp), max(vp))
        wins = "{}/{}".format(int(t["wins_for_b"]), len(SEEDS))
        print(f"  {scale:8s} {ladder:7s} {ve_cell:24s} {vp_cell:24s} "
              f"{cell['ratio_median']:6.2f}x {wins:>8s} {t['p_two_sided']:7.3f}")

    print(f"\nThat aggregate is 41-47% a single level (the error at the finest "
          f"sigma is ~70x\nthe error at the coarsest), so every level gets equal "
          f"weight here instead --\nthe per-seed geometric mean of the "
          f"per-level VP/VE ratio:")
    print(f"  {'target':8s} {'ladder':7s} {'top SNR':>8} {'median VP/VE':>13} "
          f"{'VP wins':>9}  {'coarsest level only':>21} {'VP wins':>9}")
    for scale, ladder in cells():
        c = summary["measured"].get(f"{scale}/{ladder}")
        if c is None:
            continue
        print(f"  {scale:8s} {ladder:7s} "
              f"{top_of_ladder_snr(scale, ladder):8.3f} "
              f"{c['log_mean_ratio_median']:12.2f}x "
              f"{c['log_mean_ratio_wins']:>6}/{len(SEEDS)}  "
              f"{c['coarsest_ratio_median']:20.2f}x "
              f"{c['coarsest_wins']:>6}/{len(SEEDS)}")
    print(f"  (five paired seeds cannot produce a sign-test p below "
          f"{2 / 2 ** len(SEEDS):.4f}, so 'not significant'\n   here means "
          f"'not resolvable at five seeds')")

    print(f"\nThose four rows compare different *bands* of noise level: the "
          f"shallow ladder\nstops at sigma/sd = "
          f"{normalized_sigmas('native', 'shallow')[-1]:.2f} where the other "
          f"three reach "
          f"{normalized_sigmas('native', 'full')[-1]:.2f}. Split at "
          f"{SHARED_BAND}, so like is\ncompared with like:")
    print(f"  {'target':8s} {'ladder':7s} "
          f"{'sigma/sd <= ' + str(SHARED_BAND) + ' (all four cover it)':40s} "
          f"{'sigma/sd > ' + str(SHARED_BAND):30s}")
    for scale, ladder in cells():
        c = summary["measured"].get(f"{scale}/{ladder}")
        if c is None:
            continue

        def fmt(b):
            if b is None:
                return f"{'ladder does not reach it':30s}"
            return ("{:.2f}x [{:.2f}-{:.2f}]  {}/{} seeds, {:2d} levels"
                    .format(b["median"], b["min"], b["max"], b["wins"],
                            len(SEEDS), b["levels"]))

        print(f"  {scale:8s} {ladder:7s} {fmt(c['shared_band']):40s} "
              f"{fmt(c['deep_band']):30s}")
    return summary


def main() -> None:
    if "--figures" in sys.argv:
        figure(load_rows())
        return
    if "--train" in sys.argv:
        budget = float("inf")
        if "--budget" in sys.argv:
            budget = float(sys.argv[sys.argv.index("--budget") + 1])
        train_missing(budget_seconds=budget)
    rows = load_rows()
    n_want = n_runs()
    if len(rows) < n_want:
        print(f"{len(rows)} of {n_want} runs cached; "
              f"run with --train to fill the rest")
    summary = report(rows)
    if len(rows) == n_want:
        LOGS.mkdir(exist_ok=True)
        with (LOGS / "vp_coords.json").open("w") as f:
            json.dump(summary, f, indent=2)
        figure(rows)


CELL_STYLE = {
    ("native", "full"): ("C0", "o", "the ring, as the repo uses it"),
    ("unit", "full"): ("C1", "s", r"standardized ($v = 1$)"),
    ("native", "short"): ("C2", "^", r"short ladder ($16\times$, not $160\times$)"),
    ("native", "shallow"): ("C3", "D", r"shallow ($\sigma_{\max}/4$)"),
}


def figure(rows: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.6), constrained_layout=True)

    # (a) the premise being tested, in closed form.
    ax = axes[0, 0]
    for scale in ("native", "unit"):
        levels = eval_sigmas(scale, "full")
        for coords, colour in (("ve", "C0"), ("vp", "C3")):
            sd = input_sd(torch.tensor(levels), data_var(scale), coords)
            ax.plot(levels, sd, "-" if scale == "native" else "--",
                    color=colour, lw=1.6,
                    label=f"{coords.upper()}, " +
                          ("v = 8.16 (this repo)" if scale == "native"
                           else "v = 1 (standardized)"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma$ (each ladder in its own units)")
    ax.set_ylabel("per-coordinate sd of the input")
    ax.set_title("(a) The premise. VP inputs are O(1) at every level only when\n"
                 "the data is standardized -- otherwise the span is the same,\n"
                 "pointed the other way.", fontsize=9, loc="left")
    ax.legend(fontsize=7.5)

    # (b) the paired measurement, one column per cell.
    ax = axes[0, 1]
    for j, (scale, ladder) in enumerate(cells()):
        colour, marker, _ = CELL_STYLE[(scale, ladder)]
        gm = log_mean_ratios(rows, scale, ladder)
        if not gm:
            continue
        ax.scatter([j] * len(gm), gm, s=34, color=colour, marker=marker, zorder=3)
        ax.hlines(sorted(gm)[len(gm) // 2], j - 0.22, j + 0.22, color=colour, lw=1.6)
    ax.axhline(1.0, color="0.4", lw=1)
    ax.set_xticks(range(len(cells())),
                  [f"{s}\n{ell}" for s, ell in cells()], fontsize=8)
    ax.set_ylabel("VP / VE, geometric mean over levels")
    ax.set_title(f"(b) The measurement. {len(SEEDS)} paired seeds (same init, same\n"
                 f"batches, one input transform); below 1 means VP is better.\n"
                 f"Bar is the median.", fontsize=9, loc="left")

    # (c) where along the ladder it lives -- x normalized so the four overlay.
    ax = axes[1, 0]
    for (scale, ladder), (colour, marker, label) in CELL_STYLE.items():
        levels = eval_sigmas(scale, ladder)
        per = [[rows[("vp", scale, ladder, s)][f"rel_mse_L{i}"]
                / rows[("ve", scale, ladder, s)][f"rel_mse_L{i}"]
                for s in SEEDS if ("vp", scale, ladder, s) in rows]
               for i in range(N_EVAL_LEVELS)]
        if not per[0]:
            continue
        med = [sorted(v)[len(v) // 2] for v in per]
        ax.plot(normalized_sigmas(scale, ladder), med, "-", color=colour,
                marker=marker, ms=3.5, lw=1.3, label=label)
    ax.axhline(1.0, color="0.4", lw=1)
    ax.axvline(SHARED_BAND, color="0.4", lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma\,/\,\mathrm{sd}(x_0)$  "
                  "(the only axis on which the cells are comparable)")
    ax.set_ylabel("VP / VE relative MSE (median over seeds)")
    ax.set_title("(c) Where it lives: the coarse end, and nowhere else --\n"
                 "which is why the aggregate, 47% one fine level, hides it.\n"
                 "Dotted line: how far the shallow ladder reaches.",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="lower left")

    # (d) the same measurement with the coverage confound taken out.
    ax = axes[1, 1]
    bands = [("all four ladders cover this", 0.0, SHARED_BAND),
             (f"only the deep ladders reach here", SHARED_BAND, float("inf"))]
    for j, (_, lo, hi) in enumerate(bands):
        for k, ((scale, ladder), (colour, marker, label)) in enumerate(
                CELL_STYLE.items()):
            g = sorted(log_mean_ratios(rows, scale, ladder, lo, hi))
            if not g:
                continue
            x = j + 0.20 * (k - 1.5)
            ax.errorbar([x], [g[len(g) // 2]],
                        yerr=[[g[len(g) // 2] - g[0]],
                              [g[-1] - g[len(g) // 2]]],
                        fmt=marker, color=colour, ms=7, capsize=3,
                        label=label if j == 0 else None)
    ax.axhline(1.0, color="0.4", lw=1)
    ax.set_xticks(range(len(bands)),
                  [f"$\\sigma/\\mathrm{{sd}} \\leq {SHARED_BAND}$\n{b[0]}"
                   if i == 0 else
                   f"$\\sigma/\\mathrm{{sd}} > {SHARED_BAND}$\n{b[0]}"
                   for i, b in enumerate(bands)], fontsize=8)
    ax.set_xlim(-0.6, len(bands) - 0.4)
    ax.set_ylabel("VP / VE, geometric mean over the band")
    ax.set_title("(d) And the shallow ladder is not different -- it just never\n"
                 "reaches the band. Inside the shared band all four agree;\n"
                 "outside it the three that get there all drop.",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle("Training in VP coordinates: the O(1)-input argument is "
                 "refuted, and what is left is a statement about the noise level",
                 y=1.03)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "vp_coords.png", bbox_inches="tight", dpi=150)
    print("saved figures/vp_coords.png")


if __name__ == "__main__":
    main()
