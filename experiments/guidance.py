"""What classifier-free guidance samples from, and which sampler gets there.

Guidance is normally justified by a sentence: mixing the conditional and
unconditional scores as ``(1+w) s(x|y) - w s(x)`` samples from
``q_w ∝ p(x|y)^{1+w} p(x)^{-w}``, a sharpened conditional. On a ring of
Gaussians every term of that sentence is closed form -- both scores at every
noise level, and ``q_w`` itself up to a 2-D quadrature -- so the sentence can be
checked instead of repeated. Nothing here is learned; the scores are exact, so
what is measured is the method with no estimation error mixed in.

The claim under test
--------------------
``s_w(., sigma)`` is the score of ``r_{w,sigma} ∝ p_sigma(x|y)^{1+w}
p_sigma(x)^{-w}`` -- the tilt applied to the *noised* pair. What guidance is
said to target is ``q_w = r_{w,0}``, the tilt applied to the clean pair. These
agree at ``sigma = 0`` and nowhere else, because tilting and convolution do not
commute (``tests/test_guidance.py`` measures the disagreement directly). So:

- **annealed Langevin** equilibrates at each level, so its output is set by the
  stationary law of the last level, ``r_{w,sigma_min}``. It should reach
  ``q_w``.
- **ancestral sampling** makes one pass down the ladder and is exact only when
  the scores are the scores of a genuine noising path. Guidance breaks that
  premise, so it should not.

Both samplers are run on the *same* guided scores, against exact draws from
``q_w`` (:class:`diffusion.guidance.TiltedGrid`), with the sliced-Wasserstein
distance between two independent such draws as the floor that says when a gap
is real -- the same discipline the VP/VE study used, and for the same reason:
a gap smaller than the floor is a statement about sample size.

Run
---
    python experiments/guidance.py            # sweep, log, and plot  (~19 min)
    python experiments/guidance.py --figures  # replot from committed logs
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import sampling  # noqa: E402  -- the repo's ring target and ladder endpoints

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

LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"

# The same target and ladder endpoints as every other sampling experiment here,
# so the numbers sit next to Sec. 5's rather than beside them.
WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0)
COMPONENT = 0
N_SAMPLES = 16_000
DDPM_LEVELS = 200
# Annealed Langevin needs enough steps per level to actually equilibrate --
# which is the entire mechanism this experiment attributes its result to, so
# the budget is checked rather than assumed (see equilibration_check).
LANGEVIN_LEVELS = 40
LANGEVIN_STEPS = 200
GRID_HALF_WIDTH = 8.0
GRID_RESOLUTION = 512
N_PROJECTIONS = 512
SEED = 0


def ladders():
    lo, hi = sampling.SIGMA_MIN, sampling.SIGMA_MAX
    return (geometric_sigmas(hi, lo, DDPM_LEVELS),
            geometric_sigmas(hi, lo, LANGEVIN_LEVELS))


def reference(mix, w, seed):
    """Exact draws from ``q_w``, plus the floor between two independent draws.

    The floor is the whole reason this is a measurement and not a picture: at
    n = 16,000 on this ring the sliced-Wasserstein statistic between two draws
    from the *same* distribution is not zero, and any sampler gap below it is
    sample size rather than method.
    """
    grid = TiltedGrid.build(mix, COMPONENT, w, half_width=GRID_HALF_WIDTH,
                            resolution=GRID_RESOLUTION)
    a = grid.sample(N_SAMPLES, generator=torch.Generator().manual_seed(seed))
    b = grid.sample(N_SAMPLES, generator=torch.Generator().manual_seed(seed + 1000))
    floor = sliced_wasserstein(a, b, n_projections=N_PROJECTIONS,
                               generator=torch.Generator().manual_seed(seed + 7))
    return grid, a, floor


def distance(x, ref, seed):
    return sliced_wasserstein(x, ref, n_projections=N_PROJECTIONS,
                              generator=torch.Generator().manual_seed(seed + 7))


def equilibration_check(mix, w, ladder, seed):
    """Is the Langevin arm actually equilibrated, or just lucky?

    Doubling the steps per level must not move the answer. If it does, the
    "Langevin reaches q_w because it equilibrates" reading is unearned and the
    arm is measuring its own budget. Returns the distance to ``q_w`` at the
    standard budget and at twice it.
    """
    grid, ref, _ = reference(mix, w, seed)
    out = {}
    for tag, steps in (("standard", LANGEVIN_STEPS), ("doubled", 2 * LANGEVIN_STEPS)):
        x = annealed_langevin(
            guided_score_fn(mix, COMPONENT, w),
            sampling.prior_init(N_SAMPLES, seed),
            ladder, tau=sampling.LANGEVIN_TAU, n_steps_per_level=steps,
            generator=torch.Generator().manual_seed(seed + 3),
        )
        out[tag] = distance(x, ref, seed).distance
    return out


OVERSHOOT_WEIGHTS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def overshoot_law(mix, ladder, seed):
    """The radial displacement against ``w``, over a range the sweep does not cover.

    Cheap -- ancestral only, no reference grid -- so the weight range can be
    wide enough to *test* a law rather than fit one. The displacement is
    anchored at ``w = 1`` and the rest are compared against ``log(1+w)``, which
    is a fit and is labelled as one: the mechanism in
    :func:`diffusion.guidance.guided_stationary_point` predicts an outward
    overshoot growing with w, and integrating that attractor uniformly over the
    ladder gives ``((1+w)/w) log(1+w) - 1`` -- the right leading logarithm but
    over-predicting badly at large w, so the ladder's weighting is not uniform
    and the clean log law below is empirical.
    """
    import math

    rows = []
    offsets = {}
    for w in OVERSHOOT_WEIGHTS:
        x = ddpm_ancestral(
            guided_score_fn(mix, COMPONENT, w),
            sampling.prior_init(N_SAMPLES, seed), ladder,
            generator=torch.Generator().manual_seed(seed + 1),
        )
        offsets[w] = radial_offset(mix, COMPONENT, x)
    anchor = offsets[1.0]
    for w in OVERSHOOT_WEIGHTS:
        pred = anchor * math.log1p(w) / math.log(2.0)
        rows.append({"w": w, "radial_offset": offsets[w], "log_law": pred,
                     "ratio": offsets[w] / pred})
        print(f"  w={w:5.1f}  radial offset {offsets[w]:.4f}   "
              f"log(1+w) law {pred:.4f}   ratio {offsets[w]/pred:.3f}")
    return rows


def run() -> None:
    LOGS.mkdir(exist_ok=True)
    mix = sampling.target()
    ddpm_ladder, lang_ladder = ladders()
    rows = []
    clouds = {}
    baseline = None      # q_0 draws, to measure how far the tilt moves the target
    t0 = time.time()

    for w in WEIGHTS:
        grid, ref, floor = reference(mix, w, SEED)
        if baseline is None:
            baseline = ref
        row = {
            "w": w,
            "floor": floor.distance,
            "floor_stderr": floor.stderr,
            "tail_mass": grid.tail_mass(),
            "purity_target": component_purity(mix, COMPONENT, ref),
            "spread_target": spread_about(mix, COMPONENT, ref),
            "radial_target": radial_offset(mix, COMPONENT, ref),
            # How far the tilt moves the target at all. If this is at the floor
            # then q_w is q_0 to the resolution available, and any arm's drift
            # from q_w is the sampler rather than the tilt -- which is exactly
            # what turns out to be the case on a target this well separated.
            "tilt_shift": distance(ref, baseline, SEED).distance,
        }

        anc = ddpm_ancestral(
            guided_score_fn(mix, COMPONENT, w),
            sampling.prior_init(N_SAMPLES, SEED), ddpm_ladder,
            generator=torch.Generator().manual_seed(SEED + 1),
        )
        lang = annealed_langevin(
            guided_score_fn(mix, COMPONENT, w),
            sampling.prior_init(N_SAMPLES, SEED), lang_ladder,
            tau=sampling.LANGEVIN_TAU, n_steps_per_level=LANGEVIN_STEPS,
            generator=torch.Generator().manual_seed(SEED + 3),
        )
        for tag, x in (("ancestral", anc), ("langevin", lang)):
            d = distance(x, ref, SEED)
            row[f"{tag}_sw"] = d.distance
            row[f"{tag}_sw_stderr"] = d.stderr
            row[f"{tag}_over_floor"] = d.distance / floor.distance
            row[f"{tag}_purity"] = component_purity(mix, COMPONENT, x)
            row[f"{tag}_spread"] = spread_about(mix, COMPONENT, x)
            row[f"{tag}_radial"] = radial_offset(mix, COMPONENT, x)
        rows.append(row)
        if w == max(WEIGHTS):
            clouds = {"reference": ref[:4000], "ancestral": anc[:4000],
                      "langevin": lang[:4000], "w": w, "means": mix.means,
                      "component": COMPONENT}
        print(f"w={w:4.1f}  floor {row['floor']:.4f}   "
              f"ancestral {row['ancestral_sw']:.4f} ({row['ancestral_over_floor']:.2f}x)  "
              f"langevin {row['langevin_sw']:.4f} ({row['langevin_over_floor']:.2f}x)   "
              f"tilt shift {row['tilt_shift']:.4f}   "
              f"radial target {row['radial_target']:+.4f} / "
              f"anc {row['ancestral_radial']:+.4f} / "
              f"lang {row['langevin_radial']:+.4f}   "
              f"spread {row['spread_target']:.4f}/{row['ancestral_spread']:.4f}/"
              f"{row['langevin_spread']:.4f}   "
              f"purity {row['purity_target']:.4f}/{row['ancestral_purity']:.4f}/"
              f"{row['langevin_purity']:.4f}")

    print("\nradial overshoot against w, ancestral only:")
    law = overshoot_law(mix, ddpm_ladder, SEED)

    eq = equilibration_check(mix, max(WEIGHTS), lang_ladder, SEED)
    print(f"\nequilibration control at w={max(WEIGHTS)}: "
          f"{LANGEVIN_STEPS} steps/level -> {eq['standard']:.4f}, "
          f"{2*LANGEVIN_STEPS} -> {eq['doubled']:.4f}")

    with open(LOGS / "guidance.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(LOGS / "guidance_overshoot.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["w", "radial_offset", "log_law", "ratio"])
        writer.writeheader()
        writer.writerows(law)
    with open(LOGS / "guidance.json", "w") as fh:
        json.dump({
            "component": COMPONENT, "n_samples": N_SAMPLES, "seed": SEED,
            "ddpm_levels": DDPM_LEVELS, "langevin_levels": LANGEVIN_LEVELS,
            "langevin_steps_per_level": LANGEVIN_STEPS,
            "langevin_tau": sampling.LANGEVIN_TAU,
            "sigma_min": sampling.SIGMA_MIN, "sigma_max": sampling.SIGMA_MAX,
            "grid_half_width": GRID_HALF_WIDTH, "grid_resolution": GRID_RESOLUTION,
            "n_projections": N_PROJECTIONS,
            "equilibration_control": eq,
            "seconds": time.time() - t0,
        }, fh, indent=2)
    # The scatter panel needs points, and 16k rows x 5 weights is not a CSV;
    # the largest-weight cell's own samples are carried out of the loop rather
    # than re-sampled, so the picture is of the run that produced the numbers.
    torch.save(clouds, LOGS / "guidance_clouds.pt")
    print(f"\nwrote {LOGS/'guidance.csv'}, {LOGS/'guidance.json'} and "
          f"{LOGS/'guidance_clouds.pt'} in {time.time()-t0:.0f}s")
    figures()


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(LOGS / "guidance.csv") as fh:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    clouds = torch.load(LOGS / "guidance_clouds.pt", weights_only=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    ws = [r["w"] for r in rows]

    ax = axes[0]
    ax.plot(ws, [r["ancestral_sw"] for r in rows], "o-", color="#b2182b",
            label="ancestral (one pass)")
    ax.plot(ws, [r["langevin_sw"] for r in rows], "s-", color="#2166ac",
            label="annealed Langevin")
    ax.plot(ws, [r["floor"] for r in rows], "k--", lw=1.1,
            label="floor: two exact draws")
    ax.set_xlabel("guidance weight $w$")
    ax.set_ylabel("sliced $W_2$ to $q_w$")
    ax.set_title("Distance to the density guidance\nis said to target")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(ws, [r["radial_target"] for r in rows], "k--", lw=1.2, label="$q_w$")
    ax.plot(ws, [r["ancestral_radial"] for r in rows], "o-", color="#b2182b",
            label="ancestral")
    ax.plot(ws, [r["langevin_radial"] for r in rows], "s-", color="#2166ac",
            label="annealed Langevin")
    ax.set_xlabel("guidance weight $w$")
    ax.set_ylabel(r"mean radial offset from $\mu_k$")
    ax.set_title("Where the one-pass sampler goes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # The radial *marginal*, not the cloud. A 0.18 displacement inside a cloud
    # 0.40 wide is invisible in a scatter plot -- the two point sets overlap
    # almost completely -- so the panel that is supposed to show the finding
    # would show nothing. Projecting onto the radial direction is the same
    # statistic the middle panel summarises, drawn as a distribution.
    ax = axes[2]
    mu = clouds["means"][int(clouds.get("component", 0))].double()
    u = mu / mu.norm()
    bins = torch.linspace(-1.6, 1.6, 61).numpy()
    for key, color, label in (("reference", "#444444", "$q_w$ (exact)"),
                              ("langevin", "#2166ac", "annealed Langevin"),
                              ("ancestral", "#b2182b", "ancestral")):
        proj = ((clouds[key].double() - mu) @ u).numpy()
        ax.hist(proj, bins=bins, histtype="step", density=True, lw=1.5,
                color=color, label=label)
        ax.axvline(float(proj.mean()), color=color, ls=":", lw=1.1)
    ax.axvline(0.0, color="black", lw=0.9)
    ax.set_xlabel(r"radial coordinate relative to $\mu_k$")
    ax.set_ylabel("density")
    ax.set_title(f"Radial marginal at $w = {clouds['w']:g}$\n"
                 "(dotted = each arm's mean)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "guidance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGURES/'guidance.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", action="store_true",
                    help="replot from the committed logs, no sampling")
    args = ap.parse_args()
    if args.figures:
        figures()
    else:
        run()


if __name__ == "__main__":
    main()
