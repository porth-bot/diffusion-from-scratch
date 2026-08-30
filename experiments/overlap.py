"""Guidance where the components actually overlap, which is the half not run.

`experiments/guidance.py` measured what classifier-free guidance samples from
and found, on this repo's default ring, that it samples from the conditional at
every weight: 7.7 component standard deviations of separation make the tilt a
constant where `p(x|k)` has mass, so `q_w = p(x|k)` and the sharpening guidance
is *for* cannot appear. The README has said since then that showing that half
"would need overlapping components -- a small radius or a wide s -- and that
sweep is not run here". This is that sweep.

The variable, and why it is the right one
-----------------------------------------
`diffusion.guidance.log_tilt` writes the tilt exactly:

    log q_w(x) - log q_0(x) = w (log r_k(x) - log w_k),

with `r_k` the posterior responsibility of the guided component. So guidance
does something precisely to the extent that `r_k` *varies* over the mass of
`p(.|k)` -- if the label is already determined by x, the tilt is a constant and
cancels against the normalizer. That makes the controlling variable a property
of the target rather than of the sampler, and it is one scalar: the spread of
`log r_k` under the conditional.

The sweep holds the ring's radius and component count fixed and widens the
components, so nearest-neighbour separation runs 7.65 down to 1.53 component
standard deviations. Everything is exact -- both scores, `q_w` up to the same
2-D quadrature `TiltedGrid` already uses, and draws from it -- so no sampler
and no network sits between the claim and the measurement. The distance between
two independent draws from the same `q_w` is carried as the floor, exactly as
in the guidance and VP/VE studies.

Read-outs, and which one this regime needs
------------------------------------------
Three, because "sharpened" is not one thing:

* **displacement** of `q_w`'s mean from the component mean, in units of the
  component sd -- guidance pushing the mode away from its neighbours;
* **spread** about the component mean -- guidance narrowing it;
* **purity**, the fraction of draws whose nearest mean is `k`.

Purity was useless in the separated study (a saturated 1.0000 at every weight).
It is the informative read-out here, which is the point: a diagnostic is only
uninformative relative to a regime.

Run
---
    python experiments/overlap.py            # sweep, log, and plot  (~3 min)
    python experiments/overlap.py --figures  # replot from committed logs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from diffusion.guidance import (  # noqa: E402
    TiltedGrid,
    component_purity,
    log_tilt,
    spread_about,
)
from diffusion.metrics import sliced_wasserstein  # noqa: E402
from diffusion.targets import GaussianMixture  # noqa: E402

LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"

# Radius and component count are the repo's ring; only the component width
# moves, so "separation" below is measured in component standard deviations and
# the first row IS the default target every other experiment uses.
N_COMPONENTS = 8
RADIUS = 4.0
STDS = (0.4, 0.8, 1.2, 1.6, 2.0)
WEIGHTS = (0.0, 1.0, 2.0, 4.0)
COMPONENT = 0
N_SAMPLES = 40_000
N_TILT_SAMPLES = 40_000
GRID_RESOLUTION = 512
N_PROJECTIONS = 512
SEED = 0

# The grid has to hold q_w's mass at every cell of the sweep, and the widest
# components put mass much further out than the default half-width does. Scaled
# off the target rather than fixed, and tail_mass() is logged so the choice is
# checkable rather than asserted.
def half_width(std: float) -> float:
    return RADIUS + 6.0 * std


def separation_sd(std: float) -> float:
    """Nearest-neighbour mean gap, in component standard deviations."""
    return 2 * RADIUS * math.sin(math.pi / N_COMPONENTS) / std


def tilt_spread(mix: GaussianMixture, k: int, seed: int) -> tuple[float, float]:
    """``sd(log r_k)`` and ``E[r_k]`` under ``p(.|k)`` -- the controlling scalar.

    Under the conditional, because that is the measure the tilt has to vary
    over for guidance to move anything. Taken from ``log_tilt`` at ``w = 1`` so
    the number reported is literally the quantity the identity puts in the
    exponent, not a re-derivation of it.
    """
    x = mix.components[k].sample(
        N_TILT_SAMPLES, generator=torch.Generator().manual_seed(seed)
    )
    lt = log_tilt(mix, k, x, 1.0)
    return float(lt.std()), float(mix.responsibilities(x)[:, k].mean())


def sw(a: torch.Tensor, b: torch.Tensor, seed: int) -> float:
    return float(
        sliced_wasserstein(
            a, b, n_projections=N_PROJECTIONS,
            generator=torch.Generator().manual_seed(seed),
        ).distance
    )


def cell(std: float, seed: int = SEED) -> dict:
    """One separation: q_w for every weight, against q_0, against the floor."""
    mix = GaussianMixture.ring(N_COMPONENTS, RADIUS, std)
    mu = mix.means[COMPONENT]
    hw = half_width(std)

    grids, draws = {}, {}
    for w in WEIGHTS:
        g = TiltedGrid.build(mix, COMPONENT, w, half_width=hw,
                             resolution=GRID_RESOLUTION)
        grids[w] = g
        draws[w] = g.sample(N_SAMPLES, generator=torch.Generator().manual_seed(seed + 1))

    # Two independent draws from q_0 set the resolution: any distance below
    # this is sample size, not guidance.
    ref = grids[0.0].sample(N_SAMPLES, generator=torch.Generator().manual_seed(seed + 2))
    floor = sw(draws[0.0], ref, seed + 7)

    sd_log_r, mean_r = tilt_spread(mix, COMPONENT, seed + 3)
    rows = []
    for w in WEIGHTS:
        x = draws[w]
        rows.append({
            "w": w,
            "distance_to_q0": sw(x, ref, seed + 7),
            "displacement": float((grids[w].mean() - mu).norm()) / std,
            "spread": spread_about(mix, COMPONENT, x) / std,
            "purity": component_purity(mix, COMPONENT, x),
            "tail_mass": grids[w].tail_mass(),
        })
    return {
        "std": std,
        "separation_sd": separation_sd(std),
        "half_width": hw,
        "sd_log_tilt": sd_log_r,
        "mean_responsibility": mean_r,
        "floor": floor,
        "rows": rows,
    }


def run() -> None:
    LOGS.mkdir(exist_ok=True)
    t0 = time.time()
    cells = []
    for std in STDS:
        c = cell(std)
        cells.append(c)
        print(f"separation {c['separation_sd']:5.2f} sd  "
              f"sd(log tilt) {c['sd_log_tilt']:6.3f}  floor {c['floor']:.4f}")
        for r in c["rows"]:
            print(f"    w={r['w']:<4} d/floor {r['distance_to_q0'] / c['floor']:7.2f}  "
                  f"displacement {r['displacement']:6.3f} sd  "
                  f"spread {r['spread']:6.3f} sd  purity {r['purity']:.4f}")

    payload = {
        "n_components": N_COMPONENTS, "radius": RADIUS, "stds": list(STDS),
        "weights": list(WEIGHTS), "component": COMPONENT,
        "n_samples": N_SAMPLES, "grid_resolution": GRID_RESOLUTION,
        "n_projections": N_PROJECTIONS, "seed": SEED,
        "cells": cells, "wall_seconds": time.time() - t0,
    }
    (LOGS / "overlap.json").write_text(json.dumps(payload, indent=2))
    with (LOGS / "overlap.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["separation_sd", "sd_log_tilt", "floor", "w",
                         "distance_to_q0", "displacement", "spread", "purity"])
        for c in cells:
            for r in c["rows"]:
                writer.writerow([f"{c['separation_sd']:.4f}", f"{c['sd_log_tilt']:.6f}",
                                 f"{c['floor']:.6f}", r["w"],
                                 f"{r['distance_to_q0']:.6f}", f"{r['displacement']:.6f}",
                                 f"{r['spread']:.6f}", f"{r['purity']:.6f}"])
    print(f"\nwrote logs/overlap.json and logs/overlap.csv "
          f"({payload['wall_seconds']:.1f} s)")


def figures() -> None:
    import matplotlib.pyplot as plt

    payload = json.loads((LOGS / "overlap.json").read_text())
    cells = payload["cells"]
    seps = [c["separation_sd"] for c in cells]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    colors = plt.cm.viridis([i / (len(WEIGHTS) - 1) for i in range(len(WEIGHTS))])

    ax = axes[0]
    for j, w in enumerate(payload["weights"]):
        y = [next(r for r in c["rows"] if r["w"] == w)["distance_to_q0"] / c["floor"]
             for c in cells]
        ax.plot(seps, y, "o-", color=colors[j], label=f"w = {w:g}")
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.text(seps[0], 1.15, "two-draw floor", color="0.4", fontsize=8)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("nearest-neighbour separation (component sd)")
    ax.set_ylabel(r"$W_2^{sl}(q_w, q_0)$ / floor")
    hi = max(r["distance_to_q0"] / c["floor"] for c in cells for r in c["rows"])
    lo = max(r["distance_to_q0"] / cells[0]["floor"] for r in cells[0]["rows"])
    ax.set_title(f"{lo:.2f}x the floor at {seps[0]:.1f} sd, "
                 f"{hi:.0f}x at {seps[-1]:.1f} sd", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    for j, w in enumerate(payload["weights"][1:], start=1):
        y = [next(r for r in c["rows"] if r["w"] == w)["displacement"] for c in cells]
        ax.plot(seps, y, "o-", color=colors[j], label=f"displacement, w = {w:g}")
        y = [next(r for r in c["rows"] if r["w"] == w)["spread"] for c in cells]
        ax.plot(seps, y, "s--", color=colors[j], alpha=0.7)
    y0 = [next(r for r in c["rows"] if r["w"] == 0.0)["spread"] for c in cells]
    ax.plot(seps, y0, "s--", color="0.4", label="spread, w = 0")
    ax.invert_xaxis()
    ax.set_xlabel("nearest-neighbour separation (component sd)")
    ax.set_ylabel("component sd")
    top_w = payload["weights"][-1]
    max_disp = max(next(r for r in c["rows"] if r["w"] == top_w)["displacement"]
                   for c in cells)
    min_spread = min(r["spread"] for c in cells for r in c["rows"])
    ax.set_title(f"solid: displaced up to {max_disp:.2f} sd.  "
                 f"dashed: spread never below {min_spread:.2f} sd",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    for j, w in enumerate(payload["weights"]):
        y = [next(r for r in c["rows"] if r["w"] == w)["purity"] for c in cells]
        ax.plot(seps, y, "o-", color=colors[j], label=f"w = {w:g}")
    ax.invert_xaxis()
    ax.set_xlabel("nearest-neighbour separation (component sd)")
    ax.set_ylabel("fraction nearest to the guided mean")
    sep_purity = min(r["purity"] for r in cells[0]["rows"])
    ov = {r["w"]: r["purity"] for r in cells[-1]["rows"]}
    ax.set_title(f"{sep_purity:.4f} at every w when separated; "
                 f"{ov[0.0]:.2f} to {ov[top_w]:.2f} when not",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        "Classifier-free guidance against component overlap "
        f"({N_COMPONENTS} Gaussians, radius {RADIUS:g}, exact scores)",
        fontsize=11,
    )
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "overlap.png", dpi=150)
    print("wrote figures/overlap.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", action="store_true",
                    help="replot from logs/overlap.json without rerunning")
    args = ap.parse_args()
    if not args.figures:
        run()
    figures()


if __name__ == "__main__":
    main()
