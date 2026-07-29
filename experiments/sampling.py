"""Sample from the learned score, and measure how far off the samples are.

Everything before this experiment scored the *score*. This one scores the
samples, which is the thing anyone actually wants -- and it is where a target
with closed-form structure earns its keep for the last time, because "the
samples are wrong" has three separable causes and the toys let each be
switched off:

- **prior mismatch** -- the chain starts at ``N(0, sigma_max^2 I)``, not at
  ``p_{sigma_max}``. Switched off by starting from an exact draw from
  ``p_{sigma_max}``, which the mixture provides.
- **discretization** -- the ancestral update replaces the posterior over
  ``x_0`` by its mean. Switched off only in the limit, so it is measured by
  refining the ladder.
- **score estimation** -- ``s_theta`` instead of ``s``. Switched off by
  passing the analytic noised score through the identical sampler.

Run the same sampler with the exact score and the difference between the two
runs is estimation error, full stop. No image model can do this, and it is the
reason the repo is built on a mixture.

Two metrics, because they fail differently: MMD (kernel, dimensionless, comes
with a permutation test that says whether a gap is real) and sliced
Wasserstein (transport, in the data's own units, no bandwidth to choose).
Both are compared against a **reference row** -- two independent exact draws
of the same size -- since neither reads zero at finite sample size and a
number without that floor beside it cannot be interpreted.

MMD is reported at two bandwidths, which is not padding. The median heuristic
lands on 5.7 here, which is the *diameter of the ring*: a kernel that wide
compares the samples' gross shape and is blind to whether the individual modes
have the right width. At a bandwidth equal to the component standard
deviation, 0.4, the same test sees a difference the wide kernel misses. The
median heuristic is a default, not an oracle, and on a target with structure
at two very different scales it picks the wrong one.

Run
---
    python experiments/sampling.py            # sample, measure, log, and plot
    python experiments/sampling.py --figures  # replot from committed logs
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from diffusion.langevin import annealed_langevin, geometric_sigmas, stability_bound  # noqa: E402
from diffusion.metrics import (  # noqa: E402
    median_heuristic_bandwidth,
    mmd_permutation_test,
    sliced_wasserstein,
)
from diffusion.model import ScoreNet  # noqa: E402
from diffusion.samplers import ddpm_ancestral, ddpm_gaussian_variance  # noqa: E402
from diffusion.targets import GaussianMixture  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"

# Same ring and same ladder endpoints the score network was trained on --
# a sampler run outside the levels the score was fitted at would be measuring
# extrapolation, which is a different question.
SIGMA_MIN, SIGMA_MAX = 0.05, 8.0
RING_RADIUS, RING_STD, N_MODES = 4.0, 0.4, 8
# Large, because the sliced-Wasserstein floor between two exact draws falls
# only as 1/sqrt(n) and at n = 4000 it is 0.12 with a 0.07-0.20 spread -- wide
# enough to hide every effect below. Most of that floor is the multinomial
# fluctuation in how many draws land on each of the eight modes, which costs
# transport distance in proportion to how far apart the modes are.
N_SAMPLES = 16_000
# MMD is O(n^2) in memory through the pooled kernel matrix the permutation
# test re-indexes, so it runs on a subset; sliced Wasserstein uses all points.
MMD_POINTS = 2000
N_PERMUTATIONS = 199
N_PROJECTIONS = 512
# The second bandwidth: the component standard deviation, i.e. the scale of
# the structure the median heuristic's ring-diameter kernel steps over.
FINE_BANDWIDTH = RING_STD

DDPM_LEVELS = 200
NFE_SWEEP = [5, 10, 20, 50, 100, 200, 500, 1000]

# tau at the finest level, a fifth of the 4 sigma_min^2 = 0.01 stability bound
# (diffusion.langevin): inside the bound is necessary, not sufficient.
LANGEVIN_TAU = 0.002


def target() -> GaussianMixture:
    return GaussianMixture.ring(n_components=N_MODES, radius=RING_RADIUS, std=RING_STD)


def load_net() -> ScoreNet:
    blob = torch.load(LOGS / "score_net.pt", weights_only=False)
    net = ScoreNet(**blob["config"])
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net


def exact_score(p: GaussianMixture):
    def s(x: torch.Tensor, sigma: float) -> torch.Tensor:
        return p.noised(sigma).score(x)

    return s


def learned_score(net: ScoreNet):
    def s(x: torch.Tensor, sigma: float) -> torch.Tensor:
        with torch.no_grad():
            return net(x, sigma)

    return s


def mode_spread(x: torch.Tensor, n_modes: int = N_MODES) -> float:
    """Within-mode standard deviation per coordinate, against a target of 0.4.

    Assign each sample to the nearest mode by angle and take the pooled sd of
    the offsets from that mode's centre. This is the quantity the ancestral
    sampler is *known* to get short -- replacing the posterior over ``x_0`` by
    its mean removes spread at every level (:mod:`diffusion.samplers`) -- and
    it is what a kernel at the ring's diameter cannot see and a kernel at the
    component width can.
    """
    ang = torch.arange(n_modes, dtype=x.dtype) * (2 * math.pi / n_modes)
    centres = RING_RADIUS * torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
    label = torch.round(torch.atan2(x[:, 1], x[:, 0]) / (2 * math.pi / n_modes)).long() % n_modes
    offsets = x - centres[label]
    return float(offsets.reshape(-1).std())


def mode_weights(x: torch.Tensor, n_modes: int = N_MODES) -> list[float]:
    """Fraction of samples nearest each mode of the ring, by angle.

    The modes are equally spaced on a circle, so the nearest one is a rounding
    of the angle -- no assignment problem to solve. This is the metric that
    catches the specific failure the whole annealing apparatus exists to
    prevent (a sampler that finds the support but weights it wrong), and
    unlike MMD it says *which* mode is short.
    """
    label = torch.round(torch.atan2(x[:, 1], x[:, 0]) / (2 * math.pi / n_modes)).long()
    counts = torch.bincount(label % n_modes, minlength=n_modes).double()
    return (counts / x.shape[0]).tolist()


def weight_tv(weights: list[float]) -> float:
    """Total variation from uniform: ``(1/2) sum_k |w_k - 1/K|``."""
    k = len(weights)
    return 0.5 * sum(abs(w - 1.0 / k) for w in weights)


def cheap_measure(samples: torch.Tensor, reference: torch.Tensor, seed: int) -> dict[str, float]:
    """Sliced Wasserstein and the mode summaries -- everything but MMD.

    Split out because MMD costs an ``(n+m)^2`` kernel and 199 re-indexings of
    it, so the NFE sweep runs on these alone; the arm table pays for the
    permutation test where a p-value is what the claim needs.
    """
    sw = sliced_wasserstein(
        samples, reference, n_projections=N_PROJECTIONS,
        generator=torch.Generator().manual_seed(seed),
    )
    w = mode_weights(samples)
    return {
        "sw2": sw.distance,
        "sw2_stderr": sw.stderr,
        "weight_tv": weight_tv(w),
        "weight_min": min(w),
        "weight_max": max(w),
        "mode_sd": mode_spread(samples),
    }


def measure(
    samples: torch.Tensor, reference: torch.Tensor, bandwidth: float, seed: int
) -> dict[str, float]:
    out = cheap_measure(samples, reference, seed)
    for tag, h in (("wide", bandwidth), ("fine", FINE_BANDWIDTH)):
        mmd = mmd_permutation_test(
            samples[:MMD_POINTS], reference[:MMD_POINTS],
            n_permutations=N_PERMUTATIONS, bandwidth=h,
            generator=torch.Generator().manual_seed(seed),
        )
        out[f"mmd2_{tag}"] = mmd.mmd2
        out[f"mmd_p_{tag}"] = mmd.p_value
        out[f"mmd_null_q95_{tag}"] = mmd.null_q95
    return out


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------
def prior_init(n: int, seed: int) -> torch.Tensor:
    """``N(0, sigma_max^2 I)`` -- what a real sampler has to start from."""
    g = torch.Generator().manual_seed(seed)
    return SIGMA_MAX * torch.randn(n, 2, generator=g)


def run() -> None:
    LOGS.mkdir(exist_ok=True)
    p = target()
    net = load_net()
    s_exact, s_learned = exact_score(p), learned_score(net)
    ladder = geometric_sigmas(SIGMA_MAX, SIGMA_MIN, DDPM_LEVELS)

    ref_gen = torch.Generator().manual_seed(4321)
    reference = p.sample(N_SAMPLES, generator=ref_gen)
    # One bandwidth for every arm, set by the reference against itself, so the
    # rows of the table are comparable. Letting each arm pick its own median
    # would change the kernel along with the samples.
    ref_twin = p.sample(N_SAMPLES, generator=ref_gen)
    bandwidth = median_heuristic_bandwidth(reference, ref_twin)
    print(f"reference bandwidth (median heuristic): {bandwidth:.4f}")

    arms: list[dict] = []

    def add(name: str, score: str, init: str, nfe: int, x: torch.Tensor, seconds: float) -> None:
        row = {"arm": name, "score": score, "init": init, "nfe": nfe, "seconds": seconds}
        row.update(measure(x, reference, bandwidth, seed=7))
        arms.append(row)
        print(
            f"{name:34s} nfe={nfe:5d}  SW2={row['sw2']:.4f}  TV(w)={row['weight_tv']:.4f}  "
            f"mode sd={row['mode_sd']:.4f}  "
            f"MMD2 wide={row['mmd2_wide']:+.1e} (p={row['mmd_p_wide']:.3f})  "
            f"fine={row['mmd2_fine']:+.1e} (p={row['mmd_p_fine']:.3f})"
        )
        return None

    # -- the floor: an independent exact draw, measured like any other arm ---
    add("exact sampler (independent draw)", "-", "-", 0, ref_twin, 0.0)

    # -- ancestral, three controls ------------------------------------------
    top = p.noised(SIGMA_MAX).sample(N_SAMPLES, generator=torch.Generator().manual_seed(11))
    t0 = time.perf_counter()
    x = ddpm_ancestral(s_exact, top, ladder, generator=torch.Generator().manual_seed(1))
    add("ancestral, exact score", "exact", "p_sigma_max", DDPM_LEVELS, x, time.perf_counter() - t0)

    t0 = time.perf_counter()
    x = ddpm_ancestral(
        s_exact, prior_init(N_SAMPLES, 12), ladder, generator=torch.Generator().manual_seed(1)
    )
    add("ancestral, exact score", "exact", "N(0,s_max^2)", DDPM_LEVELS, x, time.perf_counter() - t0)

    t0 = time.perf_counter()
    x_learned = ddpm_ancestral(
        s_learned, prior_init(N_SAMPLES, 12), ladder, generator=torch.Generator().manual_seed(1)
    )
    add(
        "ancestral, learned score", "learned", "N(0,s_max^2)", DDPM_LEVELS,
        x_learned, time.perf_counter() - t0,
    )

    # -- annealed Langevin, at matched and at generous budget ----------------
    langevin_runs: dict[str, torch.Tensor] = {}
    for n_levels, per_level in ((20, 10), (40, 50)):
        sigmas = geometric_sigmas(SIGMA_MAX, SIGMA_MIN, n_levels)
        nfe = n_levels * per_level
        for label, s_fn in (("exact", s_exact), ("learned", s_learned)):
            t0 = time.perf_counter()
            x = annealed_langevin(
                s_fn, prior_init(N_SAMPLES, 12), sigmas, LANGEVIN_TAU, per_level,
                generator=torch.Generator().manual_seed(2),
            )
            add(
                f"annealed Langevin, {label} score", label, "N(0,s_max^2)", nfe,
                x, time.perf_counter() - t0,
            )
            langevin_runs[f"{label}_{nfe}"] = x

    with (LOGS / "sampler_arms.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(arms[0].keys()))
        w.writeheader()
        w.writerows(arms)

    # -- the NFE sweep: where each error source stops mattering --------------
    print("\nNFE sweep (ancestral, N(0,s_max^2) init)")
    print("  nfe    SW2 exact   SW2 learned   mode sd exact   predicted   mode sd learned")
    sweep = []
    for nfe in NFE_SWEEP:
        rung = geometric_sigmas(SIGMA_MAX, SIGMA_MIN, nfe)
        row: dict[str, float] = {"nfe": float(nfe)}
        for label, s_fn in (("exact", s_exact), ("learned", s_learned)):
            x = ddpm_ancestral(
                s_fn, prior_init(N_SAMPLES, 12), rung, generator=torch.Generator().manual_seed(1)
            )
            m = cheap_measure(x, reference, seed=7)
            for k in ("sw2", "weight_tv", "mode_sd"):
                row[f"{k}_{label}"] = m[k]
        # What the single-Gaussian recursion says one *mode* should come out
        # at: variance RING_STD^2, started from the prior's variance. It has
        # no business being right on a mixture -- and once the ladder is fine
        # enough that the fine levels see one mode at a time, it is.
        row["mode_sd_predicted"] = math.sqrt(
            ddpm_gaussian_variance(RING_STD**2, rung, init_var=SIGMA_MAX**2)
        )
        sweep.append(row)
        print(
            f"{nfe:5d}   {row['sw2_exact']:9.4f}   {row['sw2_learned']:11.4f}   "
            f"{row['mode_sd_exact']:13.4f}   {row['mode_sd_predicted']:9.4f}   "
            f"{row['mode_sd_learned']:15.4f}"
        )

    with (LOGS / "sampler_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)

    # The floor both curves are chasing -- and it is a *distribution*, not a
    # constant: most of it is the multinomial fluctuation in mode occupancy,
    # which varies a lot between draws. Eight independent
    # reference-vs-reference pairs, so "the sweep has bottomed out" is a
    # statement about the metric rather than about the sampler.
    floor_gen = torch.Generator().manual_seed(555)
    floors = []
    for _ in range(8):
        a = p.sample(N_SAMPLES, generator=floor_gen)
        b = p.sample(N_SAMPLES, generator=floor_gen)
        floors.append(
            sliced_wasserstein(
                a, b, n_projections=N_PROJECTIONS, generator=torch.Generator().manual_seed(7)
            ).distance
        )
    floor_mean = sum(floors) / len(floors)
    floor_sd = math.sqrt(sum((v - floor_mean) ** 2 for v in floors) / (len(floors) - 1))
    print(
        f"\nreference-vs-reference SW2 floor: {floor_mean:.4f} +- {floor_sd:.4f} "
        f"(8 pairs, range {min(floors):.4f}-{max(floors):.4f})"
    )
    print(f"reference within-mode sd: {mode_spread(reference):.4f} (target {RING_STD})")

    with (LOGS / "sampling.json").open("w") as f:
        json.dump(
            {
                "n_samples": N_SAMPLES,
                "mmd_points": MMD_POINTS,
                "n_permutations": N_PERMUTATIONS,
                "n_projections": N_PROJECTIONS,
                "bandwidth_median_heuristic": bandwidth,
                "bandwidth_fine": FINE_BANDWIDTH,
                "reference_mode_sd": mode_spread(reference),
                "target_mode_sd": RING_STD,
                "sigma_min": SIGMA_MIN,
                "sigma_max": SIGMA_MAX,
                "ddpm_levels": DDPM_LEVELS,
                "langevin_tau": LANGEVIN_TAU,
                "langevin_stability_bound": stability_bound(SIGMA_MIN),
                "sw2_floor_mean": floor_mean,
                "sw2_floor_sd": floor_sd,
                "sw2_floor_values": floors,
            },
            f,
            indent=2,
        )

    torch.save(
        {
            "reference": reference,
            "ancestral_learned": x_learned,
            "annealed_learned": langevin_runs["learned_2000"],
            "ancestral_exact": ddpm_ancestral(
                s_exact, prior_init(N_SAMPLES, 12), ladder,
                generator=torch.Generator().manual_seed(1),
            ),
        },
        LOGS / "samples.pt",
    )
    figures()


# --------------------------------------------------------------------------
# figures (replayable from the committed logs)
# --------------------------------------------------------------------------
def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    p = target()
    blob = torch.load(LOGS / "samples.pt", weights_only=False)
    with (LOGS / "sampling.json").open() as f:
        meta = json.load(f)
    with (LOGS / "sampler_sweep.csv").open() as f:
        sweep = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]

    # -- the samples themselves ---------------------------------------------
    lim = 6.0
    q = torch.linspace(-lim, lim, 200)
    gx, gy = torch.meshgrid(q, q, indexing="xy")
    dens = p.log_prob(torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)).exp().reshape(gx.shape)

    panels = [
        ("reference", "exact sampler"),
        ("ancestral_exact", "ancestral, exact score"),
        ("ancestral_learned", "ancestral, learned score"),
        ("annealed_learned", "annealed Langevin, learned score"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2))
    for ax, (key, title) in zip(axes, panels):
        x = blob[key]
        ax.contour(gx, gy, dens, levels=6, cmap="Blues", linewidths=0.8)
        ax.scatter(x[:, 0], x[:, 1], s=1.5, alpha=0.12, color="0.15")
        w = mode_weights(x)
        ax.set_title(
            f"{title}\nweight TV {weight_tv(w):.3f}   within-mode sd {mode_spread(x):.3f}",
            fontsize=10,
        )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIGURES / "samples.png", dpi=150)
    plt.close(fig)

    # -- accuracy against budget --------------------------------------------
    nfe = [r["nfe"] for r in sweep]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.2))
    ax[0].loglog(nfe, [r["sw2_exact"] for r in sweep], "o-", color="C0", label="exact score")
    ax[0].loglog(nfe, [r["sw2_learned"] for r in sweep], "s-", color="C3", label="learned score")
    ax[0].axhline(meta["sw2_floor_mean"], color="k", ls="--", lw=1)
    ax[0].axhspan(
        min(meta["sw2_floor_values"]), max(meta["sw2_floor_values"]), color="k", alpha=0.12
    )
    ax[0].annotate(
        "two independent exact draws\n(the metric's own floor)",
        xy=(nfe[-1], meta["sw2_floor_mean"]), xytext=(0, 12),
        textcoords="offset points", ha="right", fontsize=8,
    )
    ax[0].set_xlabel("score evaluations (NFE)")
    ax[0].set_ylabel(r"sliced $W_2$ against the exact sampler")
    ax[0].set_title("Both fall, and one of them stops")
    ax[0].legend()
    ax[0].grid(alpha=0.3, which="both")

    ax[1].semilogx(
        nfe, [r["mode_sd_exact"] for r in sweep], "o-", color="C0", label="exact score"
    )
    ax[1].semilogx(
        nfe, [r["mode_sd_learned"] for r in sweep], "s-", color="C3", label="learned score"
    )
    ax[1].semilogx(
        nfe, [r["mode_sd_predicted"] for r in sweep], ":", color="0.4",
        label="single-Gaussian recursion",
    )
    ax[1].axhline(meta["target_mode_sd"], color="k", ls="--", lw=1, label="target (0.4)")
    ax[1].set_xlabel("score evaluations (NFE)")
    ax[1].set_ylabel("within-mode standard deviation")
    ax[1].set_title("Mode width, against arithmetic")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].semilogx(
        nfe, [r["weight_tv_exact"] for r in sweep], "o-", color="C0", label="exact score"
    )
    ax[2].semilogx(
        nfe, [r["weight_tv_learned"] for r in sweep], "s-", color="C3", label="learned score"
    )
    ax[2].set_xlabel("score evaluations (NFE)")
    ax[2].set_ylabel("TV distance of mode weights from uniform")
    ax[2].set_title("Are the eight modes weighted evenly?")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "sampler_accuracy.png", dpi=150)
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
