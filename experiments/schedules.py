"""Does the noise schedule matter, and to whom?

``theory/derivations.md`` Sec. 5.2 shows that once ``sigma`` is the clock the
schedule *cancels out* of the probability-flow ODE: every variance-exploding
schedule is a reparameterization of one fixed trajectory, so the only thing a
schedule can change is where the integration nodes are placed. That is a claim
about the continuous limit. This experiment asks what it is worth at a finite
budget, which is the only regime anyone samples in.

Three ladders over the identical ``[sigma_min, sigma_max]``, each uniform
spacing in a different warp of ``sigma`` (:mod:`diffusion.langevin`):

- **geometric** -- uniform in ``log sigma``. This repo's default everywhere
  else, and the one Song & Ermon's overlap argument recommends.
- **linear** -- uniform in ``sigma``. The naive choice, whose adjacent-level
  *ratio* blows up at the fine end.
- **cosine** -- uniform in ``arctan sigma``, which is Nichol & Dhariwal's
  ``alphabar = cos^2`` schedule rewritten for a variance-exploding process
  (derivation in :func:`diffusion.langevin.cosine_sigmas`).

Each is run through both samplers -- ancestral (Sec. 4.4) and PF-ODE with RK4
(Sec. 5) -- with both the exact and the learned score, at matched score
evaluations. The exact-score arms answer "how much does node placement cost the
solver"; the learned-score arms answer the question that actually matters,
which is whether a schedule can move nodes into the part of the ladder where
the network is *weak* and make the estimation error worse.

Every cell is repeated over ``N_SEEDS`` prior draws and reported as a mean with
the spread across seeds, because the differences being looked for here are the
size of the sampling floor: two independent exact draws of this size differ by
sliced ``W_2`` 0.066 +- 0.020 (Day 25), and a single-seed table would not be
able to tell a schedule effect from that.

Run
---
    python experiments/schedules.py            # sweep, log, and plot
    python experiments/schedules.py --figures  # replot from committed logs
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import sampling  # noqa: E402  -- Day 25's target, net loader, and metrics

from diffusion.langevin import SIGMA_SCHEDULES, sigma_ladder  # noqa: E402
from diffusion.ode import STEP_EVALUATIONS, pf_ode_nfe, pf_ode_sample  # noqa: E402
from diffusion.samplers import ddpm_ancestral  # noqa: E402

LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"

SIGMA_MIN, SIGMA_MAX = sampling.SIGMA_MIN, sampling.SIGMA_MAX
N_SAMPLES = sampling.N_SAMPLES

SCHEDULES = ["geometric", "cosine", "linear"]

# Budgets chosen to straddle the interesting range: Day 26 found every
# exact-score arm inside the reference floor from 41 evaluations on, so a
# schedule effect, if there is one, has to show up below that.
NFE_BUDGETS = [9, 17, 41, 101, 201]

# Prior draws per cell. Three is enough to see whether a gap survives the
# seed-to-seed spread and cheap enough that the whole sweep is under a minute.
N_SEEDS = 3
PRIOR_SEEDS = [12, 23, 34]


def rk4_nodes_for(nfe: int) -> int:
    """Largest node count whose RK4 cost fits in ``nfe`` evaluations."""
    return max(2, (nfe - 1) // STEP_EVALUATIONS["rk4"] + 1)


def run_one(
    sampler: str, score_fn, ladder: list[float], seed: int
) -> tuple[torch.Tensor, int, float]:
    x0 = sampling.prior_init(N_SAMPLES, seed)
    t0 = time.perf_counter()
    if sampler == "ancestral":
        x = ddpm_ancestral(score_fn, x0, ladder, generator=torch.Generator().manual_seed(seed))
        nfe = len(ladder)
    else:
        x = pf_ode_sample(score_fn, x0, ladder, method="rk4")
        nfe = pf_ode_nfe(len(ladder), "rk4")
    return x, nfe, time.perf_counter() - t0


def sweep(scores: dict, reference: torch.Tensor) -> list[dict]:
    rows: list[dict] = []
    print("schedule sweep: mean over", N_SEEDS, "prior draws")
    print("  budget  sampler     score     schedule    nfe      SW2 (sd)        TV(w)    mode sd")
    for budget in NFE_BUDGETS:
        for sampler in ("ancestral", "pf-ode rk4"):
            nodes = budget if sampler == "ancestral" else rk4_nodes_for(budget)
            for schedule in SCHEDULES:
                ladder = sigma_ladder(schedule, SIGMA_MAX, SIGMA_MIN, nodes)
                for label, s_fn in scores.items():
                    per_seed = []
                    nfe = 0
                    seconds = 0.0
                    for seed in PRIOR_SEEDS:
                        x, nfe, dt = run_one(sampler, s_fn, ladder, seed)
                        seconds += dt
                        per_seed.append(sampling.cheap_measure(x, reference, seed=7))
                    row = {
                        "budget": budget, "sampler": sampler, "score": label,
                        "schedule": schedule, "nodes": nodes, "nfe": nfe,
                        "n_seeds": len(PRIOR_SEEDS), "seconds": seconds,
                    }
                    for key in ("sw2", "weight_tv", "mode_sd"):
                        vals = [m[key] for m in per_seed]
                        row[key] = statistics.fmean(vals)
                        row[f"{key}_sd"] = statistics.stdev(vals)
                    rows.append(row)
                    print(
                        f"  {budget:6d}  {sampler:11s} {label:8s}  {schedule:9s} {nfe:5d}  "
                        f"{row['sw2']:.4f} ({row['sw2_sd']:.4f})   "
                        f"{row['weight_tv']:.4f}   {row['mode_sd']:.4f}"
                    )
    return rows


def summarize(rows: list[dict]) -> dict:
    """The two numbers the README section is allowed to claim.

    ``worst_gap`` is the largest spread between schedules within a single
    (budget, sampler, score) cell -- the honest headline for "how much does the
    schedule matter". ``seed_spread`` is the mean of the within-cell standard
    deviations across prior draws, which is the noise that gap has to beat.
    """
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        cells.setdefault((r["budget"], r["sampler"], r["score"]), []).append(r)
    gaps = {
        key: max(r["sw2"] for r in group) - min(r["sw2"] for r in group)
        for key, group in cells.items()
    }
    worst = max(gaps, key=gaps.get)
    return {
        "schedules": SCHEDULES,
        "budgets": NFE_BUDGETS,
        "n_seeds": N_SEEDS,
        "seed_spread_sw2": statistics.fmean(r["sw2_sd"] for r in rows),
        "worst_cell": {"budget": worst[0], "sampler": worst[1], "score": worst[2]},
        "worst_gap_sw2": gaps[worst],
        "gap_by_budget": {
            str(b): max(g for k, g in gaps.items() if k[0] == b) for b in NFE_BUDGETS
        },
    }


def run() -> None:
    LOGS.mkdir(exist_ok=True)
    p = sampling.target()
    net = sampling.load_net()
    scores = {"exact": sampling.exact_score(p), "learned": sampling.learned_score(net)}
    reference = p.sample(N_SAMPLES, generator=torch.Generator().manual_seed(4321))

    rows = sweep(scores, reference)
    with (LOGS / "schedule_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = summarize(rows)
    with (LOGS / "schedules.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"\nlargest between-schedule gap in sliced W2: {meta['worst_gap_sw2']:.4f} "
        f"at {meta['worst_cell']}; mean seed-to-seed sd {meta['seed_spread_sw2']:.4f}"
    )
    figures()


def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    with (LOGS / "schedule_sweep.csv").open() as f:
        rows = [
            {k: (v if k in ("sampler", "score", "schedule") else float(v)) for k, v in r.items()}
            for r in csv.DictReader(f)
        ]
    with (LOGS / "sampling.json").open() as f:
        day25 = json.load(f)

    colours = {"geometric": "C0", "cosine": "C2", "linear": "C3"}

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4))

    # -- panel 1: where the nodes actually sit --------------------------------
    for schedule in SCHEDULES:
        ladder = sigma_ladder(schedule, SIGMA_MAX, SIGMA_MIN, 17)
        ax[0].semilogy(range(len(ladder)), ladder, "o-", color=colours[schedule],
                       ms=4, lw=1.3, label=schedule)
    ax[0].set_xlabel("level index (coarse to fine)")
    ax[0].set_ylabel(r"$\sigma$")
    ax[0].set_title("Three ladders, same endpoints, 17 levels")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    # -- panels 2 and 3: what that costs each sampler -------------------------
    for j, sampler in enumerate(("ancestral", "pf-ode rk4")):
        a = ax[j + 1]
        for schedule in SCHEDULES:
            for score, style in (("exact", "-"), ("learned", "--")):
                sel = sorted(
                    (r for r in rows if r["sampler"] == sampler
                     and r["schedule"] == schedule and r["score"] == score),
                    key=lambda r: r["nfe"],
                )
                a.errorbar(
                    [r["nfe"] for r in sel], [r["sw2"] for r in sel],
                    yerr=[r["sw2_sd"] for r in sel], fmt="o" + style,
                    color=colours[schedule], ms=4, lw=1.3, capsize=2,
                    label=f"{schedule}, {score}",
                )
        a.axhline(day25["sw2_floor_mean"], color="k", ls=":", lw=1)
        a.axhspan(min(day25["sw2_floor_values"]), max(day25["sw2_floor_values"]),
                  color="k", alpha=0.10)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("score evaluations (NFE)")
        a.set_ylabel(r"sliced $W_2$ against an exact draw")
        a.set_title(f"{sampler}: schedule against budget")
        a.legend(fontsize=7, ncol=2)
        a.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(FIGURES / "schedules.png", dpi=150)
    plt.close(fig)
    print(f"figures -> {FIGURES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", action="store_true", help="replot from committed logs only")
    args = ap.parse_args()
    if args.figures:
        figures()
    else:
        run()
