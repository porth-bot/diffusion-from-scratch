"""Do the samplers still miss q_w once the components overlap?

`experiments/guidance.py` separated two questions and answered both on the
default ring: what guidance targets (`q_w`, the tilted *clean* pair) and which
sampler gets there (annealed Langevin does, ancestral sampling drifts ~10x the
floor and overshoots outward). `experiments/overlap.py` then widened the
components until the tilt actually bites, and measured what `q_w` does there --
but entirely by exact quadrature, with no sampler in the loop.

That left a bullet in the README's Limitations standing verbatim: "whether
ancestral sampling still overshoots once the components overlap is untested".
This is that test.

Why it is not obvious either way
--------------------------------
The ancestral drift in the separated regime has a mechanism
(`diffusion.guidance.guided_stationary_point`): at each noise level the guided
score points at a stationary point *outside* the component mean, and a
one-pass sampler chases a moving attractor it never catches, so the error
accumulates radially. Overlap changes both halves of that. The tilt stops
being a constant, so the attractor moves for a second reason; and `q_w` itself
now displaces outward (overlap.py: up to 0.90 component sd), so a sampler that
overshoots outward is overshooting toward a target that has already moved. The
two could add, or the second could absorb the first, and there is no argument
here that settles it -- which is what makes it worth running.

Design
------
The same three arms as the guidance study, on the same widened mixtures as the
overlap study, so this table joins those two rather than starting a third
setting:

* `q_w` by exact 2-D quadrature (`TiltedGrid`), with the sliced-Wasserstein
  distance between two independent draws from it as the floor;
* annealed Langevin on the guided score, which equilibrates per level;
* DDPM ancestral on the same guided score, one pass.

Both samplers see the *same* exact scores, so nothing here is estimation error.
The ladder's top is scaled with the component width -- `sigma_max` has to
dominate the data scale for the prior init to be a prior, and the widest cell's
data sd is five times the default's. That is a change from the guidance study's
fixed ladder and it is the one thing that could confound a comparison across
cells, so `sigma_max` is logged per cell and the separated cell (std = 0.4)
reproduces the guidance study's setting exactly.

Run
---
    python experiments/overlap_samplers.py            # sweep, log, plot
    python experiments/overlap_samplers.py --figures  # replot from logs
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

import overlap  # noqa: E402  -- the ring geometry and separation scale
import sampling  # noqa: E402  -- the ladder endpoints and Langevin step size

from diffusion.guidance import (  # noqa: E402
    TiltedGrid,
    component_purity,
    guided_score_fn,
    radial_offset,
    spread_about,
)
from diffusion.langevin import annealed_langevin, geometric_sigmas  # noqa: E402
from diffusion.metrics import sliced_wasserstein  # noqa: E402
from diffusion.samplers import ddpm_ancestral  # noqa: E402
from diffusion.targets import GaussianMixture  # noqa: E402

LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"

STDS = (0.4, 1.2, 2.0)          # separations 7.65, 2.55, 1.53 component sd
WEIGHTS = (0.0, 1.0, 2.0, 4.0)
COMPONENT = overlap.COMPONENT
N_SAMPLES = 16_000
DDPM_LEVELS = 200
LANGEVIN_LEVELS = 40
LANGEVIN_STEPS = 200
GRID_RESOLUTION = 512
N_PROJECTIONS = 512
SEED = 0


def sigma_max_for(std: float) -> float:
    """Ladder top, scaled so the prior init still dominates the data.

    The default ring has data sd ~ sqrt(radius^2/2 + std^2) ~ 2.9 against
    sigma_max = 8, i.e. 2.8x. Holding that ratio keeps the top of the ladder
    doing the same job at every width instead of quietly becoming a warm start
    at the widest one.
    """
    data_sd = math.sqrt(overlap.RADIUS**2 / 2.0 + std**2)
    default_sd = math.sqrt(overlap.RADIUS**2 / 2.0 + 0.4**2)
    return sampling.SIGMA_MAX * data_sd / default_sd


def sw(a, b, seed):
    return sliced_wasserstein(a, b, n_projections=N_PROJECTIONS,
                              generator=torch.Generator().manual_seed(seed))


def cell(std: float, seed: int = SEED) -> dict:
    mix = GaussianMixture.ring(overlap.N_COMPONENTS, overlap.RADIUS, std)
    hw = overlap.half_width(std)
    smax = sigma_max_for(std)
    ddpm_ladder = geometric_sigmas(smax, sampling.SIGMA_MIN, DDPM_LEVELS)
    lang_ladder = geometric_sigmas(smax, sampling.SIGMA_MIN, LANGEVIN_LEVELS)
    init = smax * torch.randn(N_SAMPLES, 2,
                              generator=torch.Generator().manual_seed(seed))

    rows = []
    for w in WEIGHTS:
        grid = TiltedGrid.build(mix, COMPONENT, w, half_width=hw,
                                resolution=GRID_RESOLUTION)
        ref = grid.sample(N_SAMPLES, generator=torch.Generator().manual_seed(seed + 1))
        twin = grid.sample(N_SAMPLES, generator=torch.Generator().manual_seed(seed + 2))
        floor = sw(ref, twin, seed + 7)

        score = guided_score_fn(mix, COMPONENT, w)
        anc = ddpm_ancestral(score, init, ddpm_ladder,
                             generator=torch.Generator().manual_seed(seed + 3))
        lang = annealed_langevin(score, init, lang_ladder,
                                 tau=sampling.LANGEVIN_TAU,
                                 n_steps_per_level=LANGEVIN_STEPS,
                                 generator=torch.Generator().manual_seed(seed + 4))

        row = {
            "std": std,
            "separation_sd": overlap.separation_sd(std),
            "w": w,
            "sigma_max": smax,
            "tail_mass": grid.tail_mass(),
            "floor": floor.distance,
            "floor_stderr": floor.stderr,
            "target_displacement": float((grid.mean() - mix.means[COMPONENT]).norm()) / std,
            "target_spread": spread_about(mix, COMPONENT, ref) / std,
            "target_purity": component_purity(mix, COMPONENT, ref),
        }
        for tag, x in (("ancestral", anc), ("langevin", lang)):
            d = sw(x, ref, seed + 7)
            row[f"{tag}_sw"] = d.distance
            row[f"{tag}_ratio"] = d.distance / floor.distance
            row[f"{tag}_radial"] = radial_offset(mix, COMPONENT, x)
            row[f"{tag}_spread"] = spread_about(mix, COMPONENT, x) / std
            row[f"{tag}_purity"] = component_purity(mix, COMPONENT, x)
        row["target_radial"] = radial_offset(mix, COMPONENT, ref)
        rows.append(row)
        print(f"  sep {row['separation_sd']:5.2f} sd  w={w:4.1f}  "
              f"floor {floor.distance:.4f}  "
              f"ancestral {row['ancestral_ratio']:6.2f}x  "
              f"langevin {row['langevin_ratio']:5.2f}x  "
              f"radial: target {row['target_radial']:+.3f} "
              f"anc {row['ancestral_radial']:+.3f} lang {row['langevin_radial']:+.3f}")
    return rows


def derived(rows):
    """Read-outs computed from the logged columns rather than logged again.

    Two of them, because the raw table invites two wrong readings.

    ``excess_sd`` is the sampler's radial offset *minus the target's*, in
    component standard deviations. The raw offset is not the drift: at overlap
    ``q_w`` has itself moved outward (up to 1.80 here), so a sampler landing at
    3.12 is not 3.12 wrong. And the raw number is in absolute units, so it is
    not comparable across cells whose components have different widths.

    ``inflation`` is the sampler's distance to ``q_w`` in floors divided by its
    own ``w = 0`` value. At ``w = 0`` there is no guidance and no tilt, so that
    row is a pure control for ancestral's ordinary discretization error on this
    target -- which is not constant across the sweep (1.85, 3.00, 3.11 floors
    as the components widen). Dividing it out is what separates "guidance makes
    ancestral worse" from "wider components make ancestral worse".
    """
    base = {(r["separation_sd"], tag): next(
                q[f"{tag}_ratio"] for q in rows
                if q["separation_sd"] == r["separation_sd"] and q["w"] == 0.0)
            for r in rows for tag in ("ancestral", "langevin")}
    out = []
    for r in rows:
        d = dict(r)
        for tag in ("ancestral", "langevin"):
            d[f"{tag}_excess_sd"] = (r[f"{tag}_radial"] - r["target_radial"]) / r["std"]
            d[f"{tag}_inflation"] = r[f"{tag}_ratio"] / base[(r["separation_sd"], tag)]
        out.append(d)
    return out


def make_figure(rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = derived(rows)
    seps = sorted({r["separation_sd"] for r in rows}, reverse=True)
    styles = dict(zip(seps, ("-", "--", ":")))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    colors = {"ancestral": "#a11", "langevin": "#1f77b4"}

    def sub(sep):
        return sorted((r for r in rows if r["separation_sd"] == sep),
                      key=lambda r: r["w"])

    ax = axes[0]
    for tag, col in colors.items():
        for sep in seps:
            g = sub(sep)
            ax.semilogy([r["w"] for r in g], [r[f"{tag}_ratio"] for r in g],
                        styles[sep], color=col, marker="o", ms=3.5, lw=1.3,
                        label=f"{tag}, {sep:.1f} sd")
    ax.axhline(1.0, color="k", lw=1.0)
    ax.text(0.05, 1.06, "floor (two draws from $q_w$)", fontsize=6.5)
    ax.set_xlabel("guidance weight $w$")
    ax.set_ylabel("sliced $W$ to $q_w$, in floors")
    ax.set_title("Distance to the target each sampler is aimed at", fontsize=9)
    ax.legend(fontsize=6, ncol=2)

    # The finding, in the units that make the cells comparable: how far past
    # q_w the sampler lands, given that q_w has itself moved.
    ax = axes[1]
    for tag, col in colors.items():
        for sep in seps:
            g = sub(sep)
            ax.plot([r["w"] for r in g], [r[f"{tag}_excess_sd"] for r in g],
                    styles[sep], color=col, marker="o", ms=3.5, lw=1.3,
                    label=f"{tag}, {sep:.1f} sd")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xlabel("guidance weight $w$")
    ax.set_ylabel("radial offset past $q_w$ (component sd)")
    ax.set_title("Overshoot, net of the target's own displacement", fontsize=9)
    ax.legend(fontsize=6, ncol=2)

    ax = axes[2]
    for sep in seps:
        g = sub(sep)
        ax.plot([r["w"] for r in g], [r["target_purity"] for r in g],
                styles[sep], color="k", marker="s", ms=5.5, lw=3.0, alpha=0.35,
                label=f"$q_w$, {sep:.1f} sd")
        for tag, col in colors.items():
            ax.plot([r["w"] for r in g], [r[f"{tag}_purity"] for r in g],
                    styles[sep], color=col, marker="o", ms=3.0, lw=1.2)
    ax.set_xlabel("guidance weight $w$")
    ax.set_ylabel("component purity")
    ax.set_title("Purity: thick grey = $q_w$, red = ancestral, blue = Langevin",
                 fontsize=9)
    ax.legend(fontsize=6)

    fig.suptitle("Classifier-free guidance with overlapping components: "
                 "the samplers, not just the target", y=1.06)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "overlap_samplers.png", dpi=150, bbox_inches="tight")
    print(f"saved {FIGURES / 'overlap_samplers.png'}")


COLUMNS = [
    "std", "separation_sd", "w", "sigma_max", "tail_mass", "floor",
    "floor_stderr", "target_displacement", "target_spread", "target_purity",
    "target_radial",
    "ancestral_sw", "ancestral_ratio", "ancestral_radial", "ancestral_spread",
    "ancestral_purity",
    "langevin_sw", "langevin_ratio", "langevin_radial", "langevin_spread",
    "langevin_purity",
]


def run() -> None:
    LOGS.mkdir(exist_ok=True)
    t0 = time.time()
    rows = []
    for std in STDS:
        print(f"std {std}  (separation {overlap.separation_sd(std):.2f} sd, "
              f"sigma_max {sigma_max_for(std):.2f})")
        rows.extend(cell(std))

    with (LOGS / "overlap_samplers.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    (LOGS / "overlap_samplers.json").write_text(json.dumps({
        "n_samples": N_SAMPLES, "ddpm_levels": DDPM_LEVELS,
        "langevin_levels": LANGEVIN_LEVELS, "langevin_steps": LANGEVIN_STEPS,
        "langevin_tau": sampling.LANGEVIN_TAU, "sigma_min": sampling.SIGMA_MIN,
        "n_components": overlap.N_COMPONENTS, "radius": overlap.RADIUS,
        "seed": SEED, "elapsed_s": time.time() - t0,
    }, indent=2) + "\n")

    summarize(rows)
    print(f"\n[{time.time() - t0:.0f}s]")
    make_figure(rows)


def summarize(rows) -> None:
    d = derived(rows)
    print(f"\n{'sep':>6} {'w':>5} {'floor':>8} {'anc/fl':>7} {'lang/fl':>8} "
          f"{'anc infl':>9} {'q_w r':>7} {'anc past':>9} {'lang past':>10}")
    for r in d:
        print(f"{r['separation_sd']:6.2f} {r['w']:5.1f} {r['floor']:8.4f} "
              f"{r['ancestral_ratio']:7.2f} {r['langevin_ratio']:8.2f} "
              f"{r['ancestral_inflation']:9.2f} {r['target_radial']:+7.3f} "
              f"{r['ancestral_excess_sd']:+9.3f} {r['langevin_excess_sd']:+10.3f}")
    print("\n'anc infl' divides out ancestral's own w=0 error, which is not "
          "constant across\nthe sweep; 'past' columns are radial offset beyond "
          "q_w, in component sd.")


def replot() -> None:
    with (LOGS / "overlap_samplers.csv").open() as fh:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    make_figure(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true",
                    help="replot from the committed logs instead of rerunning")
    if ap.parse_args().figures:
        replot()
    else:
        run()
