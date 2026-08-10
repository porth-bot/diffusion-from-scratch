"""Variance-preserving vs variance-exploding: what actually differs, measured.

``diffusion/vp.py`` shows the two processes are one chain in two coordinate
systems -- the VP state is the VE state times ``sqrt(abar)``, pathwise, and the
two ancestral samplers agree to float precision given the same noise draws
(``tests/test_vp.py``). So the process is not a modelling choice, and this
experiment is not a horse race. It measures the one thing the coordinate change
does *not* carry across, which is the prior each convention starts from:

    VE:  ``N(0, sigma_max^2 I)``     wrong by ``1 + v/sigma_max^2``, always
    VP:  ``N(0, I)``                 wrong by ``1 + abar_max (v - 1)``, so
                                     exactly right when the data have unit
                                     variance and not otherwise

for data of per-coordinate variance ``v``. Three measurements:

1. **The prior error, closed form and measured**, across ``sigma_max``, on
   standardized data (``v = 1``) and on this repo's own ring target
   (``v = r^2/2 + s^2 = 8.16``, which is not standardized and was never meant
   to be). The point of including the ring is that it is the distribution every
   other experiment here samples: if VP's advantage is conditional on
   standardization, then on the repo's actual target there is no advantage, and
   that is worth showing rather than assuming.
2. **What the prior difference is worth against the ladder.** On a Gaussian
   target both error sources are exactly computable, so they can be put on one
   axis: the prior contributes ``(prior variance gap) x (chain contraction)``,
   the ladder contributes the plug-in deficit of Sec. 4.4. The ratio is the
   practical answer to "should I switch".
3. **The same sweep with the learned score**, through the network Day 24
   trained, to check that the accounting survives an estimation error it did
   not include. Both arms use the *same* network -- there is only one score
   here, wrapped into VP coordinates by ``vp_score_fn`` -- so any difference is
   the prior and nothing else.

Not run, and named rather than implied: a network *trained* in VP coordinates.
The usual practical argument for VP is that the network sees ``O(1)`` inputs at
every noise level while a VE network sees inputs spanning ``[sigma_min,
sigma_max]``, which is an argument about conditioning and preconditioning, not
about the process. Testing it needs a second training run under a second
parameterization, and nothing below speaks to it.

Run
---
    python experiments/vp_ve.py            # sweep, log, and plot  (~4 min)
    python experiments/vp_ve.py --figures  # replot from committed logs
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

import sampling  # noqa: E402  -- Day 25's ring target, net loader, and metrics

from diffusion.langevin import geometric_sigmas  # noqa: E402
from diffusion.metrics import sliced_wasserstein  # noqa: E402
from diffusion.samplers import ddpm_ancestral, ddpm_gaussian_variance  # noqa: E402
from diffusion.targets import GaussianMixture  # noqa: E402
from diffusion.vp import (  # noqa: E402
    _abar,
    ddpm_ancestral_vp,
    prior_variance_error,
    to_vp,
    vp_score_fn,
)

LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"

SIGMA_MIN = sampling.SIGMA_MIN
SIGMA_MAX_SWEEP = [2.0, 3.0, 5.0, 8.0, 16.0, 32.0]
N_LEVELS = 200                     # the ladder Day 25 samples at
LEVEL_SWEEP = [8, 16, 32, 64, 128, 256]
N_SAMPLES = 8000
N_PROJECTIONS = 512
PRIOR_SEEDS = [12, 23, 34]

# The repo's ring: radius 4, component sd 0.4, so per-coordinate variance is
# r^2 E[cos^2] + s^2 = r^2/2 + s^2. Nothing about it is standardized.
RING_VAR = sampling.RING_RADIUS**2 / 2 + sampling.RING_STD**2


def standardized_ring() -> GaussianMixture:
    """The same ring scaled to unit per-coordinate variance.

    The comparison needs a ``v = 1`` target that is otherwise identical to the
    ``v = 8.16`` one, so that the only thing moving between the two arms is the
    scale the VP prior assumes.
    """
    k = 1.0 / math.sqrt(RING_VAR)
    return GaussianMixture.ring(
        n_components=sampling.N_MODES,
        radius=sampling.RING_RADIUS * k,
        std=sampling.RING_STD * k,
    )


def sample_both(score, sigma_max, n_levels, seed):
    """One VE run and one VP run, each from its own convention's prior."""
    sigmas = geometric_sigmas(sigma_max, SIGMA_MIN, n_levels)
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(N_SAMPLES, 2, generator=g)

    ve = ddpm_ancestral(
        score, sigma_max * z, sigmas,
        generator=torch.Generator().manual_seed(seed + 1000),
    )
    vp = ddpm_ancestral_vp(
        vp_score_fn(score), z, sigmas,
        generator=torch.Generator().manual_seed(seed + 1000),
    )
    return ve, vp


def row_for(tag, target, reference, score, sigma_max, n_levels, seed, learned):
    ve, vp = sample_both(score, sigma_max, n_levels, seed)
    var = float(torch.diagonal(target.cov).mean())
    ve_ratio, vp_ratio = prior_variance_error(sigma_max, var)
    return {
        "target": tag,
        "score": "learned" if learned else "exact",
        "sigma_max": sigma_max,
        "n_levels": n_levels,
        "seed": seed,
        "data_var": var,
        "ve_prior_ratio": ve_ratio,
        "vp_prior_ratio": vp_ratio,
        "ve_sw2": sliced_wasserstein(
            ve, reference, n_projections=N_PROJECTIONS,
            generator=torch.Generator().manual_seed(99),
        ).distance,
        "vp_sw2": sliced_wasserstein(
            vp, reference, n_projections=N_PROJECTIONS,
            generator=torch.Generator().manual_seed(99),
        ).distance,
        "ve_var": float(ve.var(0, unbiased=True).mean()),
        "vp_var": float(vp.var(0, unbiased=True).mean()),
    }


def reference_floor(target, reference, seeds=(1, 2, 3)) -> float:
    """Sliced W2 between two independent exact draws of the same size.

    Without this the table's VE/VP columns have no scale: a gap of 0.001 could
    be a real effect or could be nothing. It is nothing, and this is the number
    that says so. Reported per target, because sliced W2 carries the target's
    own units and the standardized ring is 2.86x smaller than the ring by
    construction -- the two targets' columns are not comparable with each other,
    only each with its own floor.
    """
    out = []
    for seed in seeds:
        draw = target.sample(N_SAMPLES, generator=torch.Generator().manual_seed(seed))
        out.append(sliced_wasserstein(
            draw, reference, n_projections=N_PROJECTIONS,
            generator=torch.Generator().manual_seed(99),
        ).distance)
    return sum(out) / len(out)


def pathwise_gap() -> dict:
    """The equivalence, at the precision float32 allows, as a logged number.

    ``tests/test_vp.py`` asserts this in float64. Repeating it here in the
    experiment's own dtype is what lets the figure caption say the two curves
    are the same curve rather than two curves that happen to overlap.
    """
    p = sampling.target()
    sigmas = geometric_sigmas(8.0, SIGMA_MIN, 64)
    g = torch.Generator().manual_seed(5)
    z = torch.randn(4000, 2, generator=g)
    score = sampling.exact_score(p)
    ve = ddpm_ancestral(score, 8.0 * z, sigmas,
                        generator=torch.Generator().manual_seed(6))
    vp = ddpm_ancestral_vp(vp_score_fn(score), to_vp(8.0 * z, sigmas[0]), sigmas,
                           generator=torch.Generator().manual_seed(6))
    return {
        "max_abs_gap": float((ve - vp).abs().max()),
        "state_scale": float(ve.abs().max()),
    }


def ladder_tradeoff() -> list[dict]:
    """Prior error against discretization error, both in closed form.

    A Gaussian target makes the whole ancestral chain linear, so
    ``ddpm_gaussian_variance`` gives the sampled variance exactly from either
    convention's starting variance. The difference between the two runs is the
    prior's entire contribution; the gap from the target variance is the
    ladder's. No sampling and no seeds -- these are the two error sources with
    the Monte Carlo removed.
    """
    rows = []
    for sigma_max in SIGMA_MAX_SWEEP:
        for n_levels in LEVEL_SWEEP:
            sigmas = geometric_sigmas(sigma_max, SIGMA_MIN, n_levels)
            ve = ddpm_gaussian_variance(1.0, sigmas, init_var=sigma_max**2)
            vp = ddpm_gaussian_variance(1.0, sigmas, init_var=1.0 / _abar(sigma_max))
            rows.append({
                "sigma_max": sigma_max,
                "n_levels": n_levels,
                "ve_var": ve,
                "vp_var": vp,
                "prior_contribution": vp - ve,
                "ladder_deficit": 1.0 - vp,
            })
    return rows


def run() -> None:
    t0 = time.time()
    LOGS.mkdir(exist_ok=True)

    targets = {"ring": sampling.target(), "ring_standardized": standardized_ring()}
    net = sampling.load_net()
    references = {
        tag: p.sample(N_SAMPLES, generator=torch.Generator().manual_seed(4321))
        for tag, p in targets.items()
    }

    rows = []
    for tag, p in targets.items():
        for sigma_max in SIGMA_MAX_SWEEP:
            for seed in PRIOR_SEEDS:
                rows.append(row_for(tag, p, references[tag],
                                    sampling.exact_score(p), sigma_max,
                                    N_LEVELS, seed, learned=False))
        print(f"  {tag}: exact-score sweep done ({time.time() - t0:.0f}s)")

    # The learned arm runs on the ring only: the network was trained on it, and
    # a network evaluated on the standardized ring would be extrapolating in x,
    # which is a different failure and not the one being measured.
    p = targets["ring"]
    for sigma_max in SIGMA_MAX_SWEEP:
        for seed in PRIOR_SEEDS:
            rows.append(row_for("ring", p, references["ring"],
                                sampling.learned_score(net), sigma_max,
                                N_LEVELS, seed, learned=True))
    print(f"  ring: learned-score sweep done ({time.time() - t0:.0f}s)")

    with (LOGS / "vp_ve_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    trade = ladder_tradeoff()
    with (LOGS / "vp_ve_tradeoff.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trade[0].keys()))
        w.writeheader()
        w.writerows(trade)

    floors = {tag: reference_floor(p, references[tag])
              for tag, p in targets.items()}
    meta = {"pathwise": pathwise_gap(), "ring_var": RING_VAR, "floors": floors,
            "n_samples": N_SAMPLES, "n_levels": N_LEVELS,
            "seconds": round(time.time() - t0, 1)}
    with (LOGS / "vp_ve.json").open("w") as f:
        json.dump(meta, f, indent=2)

    report(rows, trade, meta)
    figures()


def report(rows, trade, meta) -> None:
    print(f"\npathwise: the two samplers' outputs differ by at most "
          f"{meta['pathwise']['max_abs_gap']:.2e} on states of size "
          f"{meta['pathwise']['state_scale']:.1f} -- float32 rounding, not a "
          f"difference in the chain.\n")
    print("sliced-W2 floor between two exact draws of this size: "
          + ", ".join(f"{k} {v:.4f}" for k, v in meta["floors"].items())
          + "  (W2 is in each target's own units; the two are not comparable)")
    header = ("target             score   sigma_max | prior ratio VE / VP | "
              "sliced W2 VE / VP | VP - VE")
    print(header)
    print("-" * len(header))
    for tag in ("ring_standardized", "ring"):
        for score in ("exact", "learned"):
            cells = [r for r in rows if r["target"] == tag and r["score"] == score]
            if not cells:
                continue
            for sigma_max in SIGMA_MAX_SWEEP:
                c = [r for r in cells if r["sigma_max"] == sigma_max]
                ve = sum(r["ve_sw2"] for r in c) / len(c)
                vp = sum(r["vp_sw2"] for r in c) / len(c)
                print(f"{tag:18s} {score:7s} {sigma_max:9.0f} | "
                      f"{c[0]['ve_prior_ratio']:.4f} / {c[0]['vp_prior_ratio']:.4f} | "
                      f"{ve:.4f} / {vp:.4f} | {vp - ve:+.4f}")
    print("\nprior vs ladder, on a Gaussian target (closed form):")
    for sigma_max in (3.0, 8.0):
        for n_levels in (8, 64, 256):
            t = next(r for r in trade if r["sigma_max"] == sigma_max
                     and r["n_levels"] == n_levels)
            print(f"  sigma_max {sigma_max:4.0f}, {n_levels:4d} levels: "
                  f"prior {t['prior_contribution']:.5f}, "
                  f"ladder {t['ladder_deficit']:.5f}, "
                  f"ratio {t['prior_contribution'] / t['ladder_deficit']:.4f}")


def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    text = ("target", "score")
    with (LOGS / "vp_ve_sweep.csv").open() as f:
        rows = [{k: (v if k in text else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]
    with (LOGS / "vp_ve_tradeoff.csv").open() as f:
        trade = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    with (LOGS / "vp_ve.json").open() as f:
        meta = json.load(f)

    def curve(tag, score, key):
        out = []
        for sigma_max in SIGMA_MAX_SWEEP:
            c = [r for r in rows if r["target"] == tag and r["score"] == score
                 and r["sigma_max"] == sigma_max]
            out.append(sum(r[key] for r in c) / len(c))
        return out

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
    xs = SIGMA_MAX_SWEEP

    ax = axes[0, 0]
    ax.axhline(1.0, color="k", lw=1.0, ls=":", label="exact prior")
    for tag, colour in (("ring_standardized", "C0"), ("ring", "C3")):
        ax.plot(xs, curve(tag, "exact", "ve_prior_ratio"), "o-", color=colour,
                lw=1.4, ms=4, label=f"VE, {tag.replace('ring_standardized', 'v=1')}"
                .replace("ring", f"v={RING_VAR:.2f}"))
        ax.plot(xs, curve(tag, "exact", "vp_prior_ratio"), "s--", color=colour,
                lw=1.4, ms=4, label=f"VP, {tag.replace('ring_standardized', 'v=1')}"
                .replace("ring", f"v={RING_VAR:.2f}"))
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.minorticks_off()
    ax.set_xlabel(r"$\sigma_{\max}$")
    ax.set_ylabel("assumed prior variance / true")
    ax.set_title("1. VP's prior is exact only on standardized data", loc="left")
    ax.legend(fontsize=6.5)

    ax = axes[0, 1]
    for tag, colour in (("ring_standardized", "C0"), ("ring", "C3")):
        ax.plot(xs, curve(tag, "exact", "ve_sw2"), "o-", color=colour, lw=1.4,
                ms=4, label=f"VE, {'v=1' if 'standard' in tag else f'v={RING_VAR:.2f}'}")
        ax.plot(xs, curve(tag, "exact", "vp_sw2"), "s--", color=colour, lw=1.4,
                ms=4, label=f"VP, {'v=1' if 'standard' in tag else f'v={RING_VAR:.2f}'}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.minorticks_off()
    ax.set_xlabel(r"$\sigma_{\max}$")
    ax.set_ylabel("sliced $W_2$ to the target")
    ax.set_title(f"2. What that buys, at {N_LEVELS} levels (exact score)", loc="left")
    ax.legend(fontsize=6.5)

    ax = axes[1, 0]
    for sigma_max, colour in zip((3.0, 8.0, 32.0), ("C0", "C2", "C4")):
        sel = [t for t in trade if t["sigma_max"] == sigma_max]
        levels = [t["n_levels"] for t in sel]
        ax.plot(levels, [t["ladder_deficit"] for t in sel], "-", color=colour,
                lw=1.6, label=rf"ladder, $\sigma_{{\max}}={sigma_max:g}$")
        ax.plot(levels, [t["prior_contribution"] for t in sel], "--", color=colour,
                lw=1.6, label=rf"prior, $\sigma_{{\max}}={sigma_max:g}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ladder length")
    ax.set_ylabel("variance error on a unit Gaussian")
    ax.set_title("3. The prior gap against the one that dominates it", loc="left")
    ax.legend(fontsize=6.5, ncol=2)

    ax = axes[1, 1]
    ve = curve("ring", "learned", "ve_sw2")
    vp = curve("ring", "learned", "vp_sw2")
    ax.plot(xs, curve("ring", "exact", "ve_sw2"), "o-", color="0.5", lw=1.3,
            ms=4, label="VE, exact score")
    ax.plot(xs, curve("ring", "exact", "vp_sw2"), "s--", color="0.5", lw=1.3,
            ms=4, label="VP, exact score")
    ax.plot(xs, ve, "o-", color="C1", lw=1.5, ms=4, label="VE, learned score")
    ax.plot(xs, vp, "s--", color="C1", lw=1.5, ms=4, label="VP, learned score")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.minorticks_off()
    ax.set_xlabel(r"$\sigma_{\max}$")
    ax.set_ylabel("sliced $W_2$ to the target")
    ax.set_title("4. The same accounting under a learned score", loc="left")
    ax.legend(fontsize=6.5)

    # The floor between two exact draws, per target, in that target's units.
    # Every VE/VP gap in panels 2 and 4 sits below these lines.
    for ax, tags in ((axes[0, 1], (("ring_standardized", "C0"), ("ring", "C3"))),
                     (axes[1, 1], (("ring", "k"),))):
        for tag, colour in tags:
            ax.axhline(meta["floors"][tag], color=colour, lw=0.9, ls=":")
            ax.annotate("two exact draws", xy=(xs[0], meta["floors"][tag]),
                        xytext=(0, 3), textcoords="offset points",
                        fontsize=6, color=colour)

    fig.savefig(FIGURES / "vp_ve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIGURES / 'vp_ve.png'}  "
          f"(pathwise gap {meta['pathwise']['max_abs_gap']:.2e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", action="store_true",
                    help="replot from committed logs only")
    args = ap.parse_args()
    figures() if args.figures else run()
