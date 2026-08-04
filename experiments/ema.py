"""Does an EMA of the weights actually help? Ablated across decays and seeds.

Every diffusion codebase averages its weights and none of them shows the
number. The claim is easy to state and easy to check on this repo's target,
because the exact noised score is available at every level: hold the
optimization fixed and evaluate the *same run* under raw and averaged
weights. That makes it a paired comparison -- the two arms differ by the
average and nothing else, not by a seed (``tests/test_ema.py`` asserts the
raw arm is bit-for-bit the EMA-off run).

Three things get measured, and only the first is the one people quote:

1. **Final score error**, relative MSE against the exact noised score,
   averaged over the ladder, raw vs EMA, at four decays and five seeds.
2. **Seed spread.** The variance argument (``diffusion/ema.py``, eq. 2) says
   EMA should shrink run-to-run scatter as well as the mean, and that is the
   half a single-seed comparison cannot see.
3. **The crossover.** A decay is an averaging window; while the weights are
   still travelling, averaging them is averaging worse parameters. The step
   at which the EMA arm overtakes the raw arm is a real cost of a long decay
   and it is plotted rather than asserted away.

Run
---
    python experiments/ema.py              # run the ablation, log, plot
    python experiments/ema.py --figures    # replot from the committed log
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from diffusion.dsm import train_score_model  # noqa: E402
from diffusion.model import ScoreNet  # noqa: E402
from diffusion.targets import GaussianMixture  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"

# The same target and noise range as experiments/score_matching.py, so the
# numbers here sit on the same scale as the repo's main training run.
SIGMA_MIN, SIGMA_MAX = 0.05, 8.0
WIDTH, DEPTH = 128, 4
N_STEPS, BATCH, LR = 20_000, 512, 2e-3
EVAL_EVERY = 500
EVAL_POINTS = 4096
SEEDS = (0, 1, 2, 3, 4)
# 0.99 averages ~100 steps, 0.9995 averages ~2000 -- a tenth of the run, which
# is where the lag should start costing something.
DECAYS = (0.99, 0.999, 0.9995, 0.9999)


def target() -> GaussianMixture:
    return GaussianMixture.ring(n_components=8, radius=4.0, std=0.4)


def run() -> None:
    LOGS.mkdir(exist_ok=True)
    p = target()
    rows: list[dict[str, float]] = []

    for decay in DECAYS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = ScoreNet(dim=2, width=WIDTH, depth=DEPTH)
            hist = train_score_model(
                net, p, SIGMA_MIN, SIGMA_MAX,
                n_steps=N_STEPS, batch_size=BATCH, lr=LR, seed=seed,
                log_every=1000, eval_every=EVAL_EVERY, eval_points=EVAL_POINTS,
                ema_decay=decay,
            )
            for step, raw, ema, raw_cos, ema_cos in zip(
                hist.eval_steps, hist.eval_relative_mse, hist.ema_relative_mse,
                hist.eval_cosine, hist.ema_cosine,
            ):
                rows.append({
                    "decay": decay, "seed": seed, "step": step,
                    "raw_relative_mse": raw, "ema_relative_mse": ema,
                    "raw_cosine": raw_cos, "ema_cosine": ema_cos,
                })
            print(
                f"decay {decay:<7} seed {seed}: "
                f"raw {hist.eval_relative_mse[-1]:.5f}  "
                f"ema {hist.ema_relative_mse[-1]:.5f}  "
                f"({hist.seconds:.0f}s)",
                flush=True,
            )

    with (LOGS / "ema_ablation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # The raw arm does not depend on the decay -- the shadow never touches the
    # gradient -- so the same (seed, step) must give the same raw error at
    # every decay. Re-running it four times is the price of the simple loop,
    # and checking that it lands on the same number turns that redundancy into
    # a consistency check on the claim the whole paired design rests on.
    for seed in SEEDS:
        for step in sorted({r["step"] for r in rows}):
            vals = {r["raw_relative_mse"] for r in rows
                    if r["seed"] == seed and r["step"] == step}
            assert len(vals) == 1, (seed, step, vals)

    summary = summarize(rows)
    with (LOGS / "ema.json").open("w") as f:
        json.dump(summary, f, indent=2)
    report(summary)
    figures()


def summarize(rows: list[dict[str, float]]) -> dict:
    """Final-step means and spreads per decay, plus the crossover step."""
    final_step = max(r["step"] for r in rows)
    out: dict = {
        "n_steps": N_STEPS, "seeds": list(SEEDS), "decays": list(DECAYS),
        "eval_every": EVAL_EVERY, "width": WIDTH, "depth": DEPTH,
        "batch_size": BATCH, "lr": LR,
        "sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX,
        "per_decay": {},
    }
    for decay in DECAYS:
        final = [r for r in rows if r["decay"] == decay and r["step"] == final_step]
        raw = [r["raw_relative_mse"] for r in final]
        ema = [r["ema_relative_mse"] for r in final]

        # Crossover: the first evaluation step at which the EMA arm is better
        # on a majority of seeds and stays that way. None if it never is.
        steps = sorted({r["step"] for r in rows if r["decay"] == decay})
        crossover = None
        for i, step in enumerate(steps):
            later = steps[i:]
            if all(
                sum(
                    1 for r in rows
                    if r["decay"] == decay and r["step"] == s
                    and r["ema_relative_mse"] < r["raw_relative_mse"]
                ) > len(SEEDS) / 2
                for s in later
            ):
                crossover = step
                break

        out["per_decay"][str(decay)] = {
            "raw_mean": statistics.mean(raw),
            "raw_sd": statistics.stdev(raw),
            "ema_mean": statistics.mean(ema),
            "ema_sd": statistics.stdev(ema),
            "mean_ratio": statistics.mean(ema) / statistics.mean(raw),
            "sd_ratio": statistics.stdev(ema) / statistics.stdev(raw),
            "wins": sum(1 for a, b in zip(ema, raw) if a < b),
            "n_seeds": len(raw),
            "crossover_step": crossover,
            # The i.i.d.-jitter bound (eq. 2), for reference: real iterates
            # are correlated, so the measured reduction should fall short.
            "variance_ratio_bound": (1 - decay) / (1 + decay),
        }
    return out


def report(summary: dict) -> None:
    print(
        f"\nEMA ablation: {N_STEPS:,} steps, {len(SEEDS)} seeds per decay, "
        f"relative MSE against the exact score at step {N_STEPS:,}\n"
    )
    print("  decay      raw mean +- sd        EMA mean +- sd     ratio   wins   crossover")
    for decay in DECAYS:
        s = summary["per_decay"][str(decay)]
        cross = "never" if s["crossover_step"] is None else f"{s['crossover_step']:,}"
        print(
            f"  {decay:<8}  {s['raw_mean']:.5f} +- {s['raw_sd']:.5f}   "
            f"{s['ema_mean']:.5f} +- {s['ema_sd']:.5f}   "
            f"{s['mean_ratio']:.3f}   {s['wins']}/{s['n_seeds']}   {cross:>9}"
        )
    print(
        "\n  'ratio' is EMA mean / raw mean: below 1 means the average is better.\n"
        "  'crossover' is the first eval step from which EMA leads on a majority\n"
        "  of seeds and never falls back."
    )


def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    with (LOGS / "ema_ablation.csv").open() as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    with (LOGS / "ema.json").open() as f:
        summary = json.load(f)

    decays = sorted({r["decay"] for r in rows})
    steps = sorted({r["step"] for r in rows})

    def curve(decay: float, column: str) -> tuple[list[float], list[float]]:
        med, lo_hi = [], []
        for s in steps:
            vals = sorted(r[column] for r in rows if r["decay"] == decay and r["step"] == s)
            med.append(statistics.median(vals))
            lo_hi.append((vals[0], vals[-1]))
        return med, lo_hi

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # -- left: the trajectories, one decay per colour ------------------------
    ax = axes[0]
    raw_med, raw_band = curve(decays[0], "raw_relative_mse")   # identical per decay
    ax.fill_between(steps, [b[0] for b in raw_band], [b[1] for b in raw_band],
                    color="k", alpha=0.12)
    ax.plot(steps, raw_med, "k-", lw=2, label="raw weights (seed range shaded)")
    for i, decay in enumerate(decays):
        med, _ = curve(decay, "ema_relative_mse")
        ax.plot(steps, med, "-", color=f"C{i}", label=f"EMA $d$ = {decay}")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"relative MSE vs exact score")
    ax.set_yscale("log")
    ax.set_title(f"Median over {len(SEEDS)} seeds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # -- right: final mean and spread ----------------------------------------
    ax = axes[1]
    x = range(len(decays))
    raw_mean = [summary["per_decay"][str(d)]["raw_mean"] for d in decays]
    raw_sd = [summary["per_decay"][str(d)]["raw_sd"] for d in decays]
    ema_mean = [summary["per_decay"][str(d)]["ema_mean"] for d in decays]
    ema_sd = [summary["per_decay"][str(d)]["ema_sd"] for d in decays]
    w = 0.35
    ax.bar([i - w / 2 for i in x], raw_mean, w, yerr=raw_sd, capsize=4,
           color="0.6", label="raw")
    ax.bar([i + w / 2 for i in x], ema_mean, w, yerr=ema_sd, capsize=4,
           color="C0", label="EMA")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(d) for d in decays])
    ax.set_xlabel("EMA decay $d$")
    ax.set_ylabel("relative MSE at the final step")
    ax.set_title("Error bars are the seed sd, not a fit")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIGURES / "ema_ablation.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURES / 'ema_ablation.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", action="store_true",
                    help="replot from the committed log without retraining")
    args = ap.parse_args()
    if args.figures:
        figures()
    else:
        run()
