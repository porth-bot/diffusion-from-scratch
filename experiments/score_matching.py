"""Train a score network by DSM, and score it against the truth at every noise level.

This is the experiment the repo's target choice was made for. Because a
Gaussian mixture stays a Gaussian mixture when noised, ``grad log p_sigma`` is
available in closed form at *every* ``sigma`` -- so a learned score is not
reported as a picture that looks about right, it is reported as an error
against the exact answer, level by level.

Several numbers per level, because a score fails in more than one way:

- **relative MSE**, ``mean||s_theta - s||^2 / mean||s||^2``. Scale-free, so
  it can be compared across levels whose scores differ by two orders of
  magnitude in size.
- **cosine similarity**, which isolates direction. A sampler follows the
  direction of the score; one that points correctly but is 20% short still
  moves states to the right place, only more slowly. Reported both plainly
  and weighted by ``||s||``, because the plain average reads noise wherever
  the exact field passes through zero -- see
  :func:`diffusion.dsm.score_diagnostics`.

And one number for the whole run: the **irreducible loss**, what a perfect
score would score, which is what makes the training curve readable at all.

Run
---
    python experiments/score_matching.py            # train, log, and plot
    python experiments/score_matching.py --figures  # replot from committed logs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from diffusion.dsm import (  # noqa: E402
    eval_dsm_floor,
    fixed_eval_batch,
    irreducible_dsm_loss,
    score_diagnostics,
    train_score_model,
)
from diffusion.langevin import geometric_sigmas  # noqa: E402
from diffusion.model import ScoreNet, count_parameters  # noqa: E402
from diffusion.targets import GaussianMixture  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"

# The 8-mode ring: modes 3.06 apart (~7.7 component sd), so the coarse levels
# genuinely have to merge them and the fine levels genuinely have to separate
# them again. sigma spans from well under a mode's width to twice the ring
# radius, at which point p_sigma is very nearly a single Gaussian.
SIGMA_MIN, SIGMA_MAX = 0.05, 8.0
N_EVAL_LEVELS = 13
FIELD_SIGMAS = [4.0, 1.0, 0.2]

WIDTH, DEPTH = 128, 4
N_STEPS, BATCH, LR, SEED = 60_000, 512, 2e-3, 0


def target() -> GaussianMixture:
    return GaussianMixture.ring(n_components=8, radius=4.0, std=0.4)


def eval_sigmas() -> list[float]:
    """Geometric, low to high, so the plots read left to right in sigma."""
    return sorted(geometric_sigmas(SIGMA_MAX, SIGMA_MIN, N_EVAL_LEVELS))


def ladder_floor(
    p: GaussianMixture, n_levels: int = 33, n_points: int = 400_000
) -> tuple[float, float]:
    r"""The irreducible DSM loss averaged over the training measure on sigma.

    ``E_{log sigma ~ U} L*(sigma)``, by the trapezoid rule in ``log sigma``
    on a dense geometric ladder, returned with its Monte Carlo standard
    error. The floor varies from ``d = 2`` at the fine end to ~0 at the coarse
    end, so the quadrature has to be fine enough that its own error is small
    compared to the gap being measured -- which is a fraction of a percent,
    hence 33 levels (the value is stable to 1e-4 between 33 and 129) and
    400k points per level.
    """
    levels = sorted(geometric_sigmas(SIGMA_MAX, SIGMA_MIN, n_levels))
    est = [
        irreducible_dsm_loss(
            p, s, n_points=n_points,
            generator=torch.Generator().manual_seed(99), return_stderr=True,
        )
        for s in levels
    ]
    # Uniform spacing in log sigma, so the trapezoid weights are 1/2 at the
    # endpoints and 1 inside, normalized by the total weight.
    weights = [0.5] + [1.0] * (n_levels - 2) + [0.5]
    total = sum(weights)
    value = sum(w * v for w, (v, _) in zip(weights, est)) / total
    # Levels use independent draws, so the weighted-sum variance adds.
    var = sum((w / total) ** 2 * se**2 for w, (_, se) in zip(weights, est))
    return value, math.sqrt(var)


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------
def run() -> None:
    LOGS.mkdir(exist_ok=True)
    p = target()
    torch.manual_seed(SEED)
    net = ScoreNet(dim=2, width=WIDTH, depth=DEPTH)
    print(f"score net: {count_parameters(net):,} parameters")

    # A frozen batch, so the loss curve is a deterministic function of the
    # parameters, and its exact optimum is computable on the very same points.
    batch = fixed_eval_batch(p, eval_sigmas(), n_per_level=4000, seed=1234)
    batch_floor = eval_dsm_floor(p, batch)

    hist = train_score_model(
        net,
        p,
        SIGMA_MIN,
        SIGMA_MAX,
        n_steps=N_STEPS,
        batch_size=BATCH,
        lr=LR,
        seed=SEED,
        log_every=100,
        eval_every=500,
        eval_sigmas=[SIGMA_MIN, math.sqrt(SIGMA_MIN * SIGMA_MAX), SIGMA_MAX],
        eval_points=4096,
        eval_batch=batch,
        progress=True,
    )
    print(f"trained in {hist.seconds:.1f} s")

    levels = eval_sigmas()
    diag = score_diagnostics(
        net, p, levels, n_points=20_000,
        generator=torch.Generator().manual_seed(2024),
    )

    # The loss the *perfect* score would attain, level by level, and averaged
    # over the log-uniform measure the training loop actually draws sigma
    # from. Trapezoid in log sigma on a dense ladder, not a plain mean over
    # the 13 reporting levels: the floor runs from ~d at the fine end to ~0 at
    # the coarse end, so a coarse rule with full weight on both endpoints is
    # biased by more than the gap being measured.
    floors = [
        irreducible_dsm_loss(p, s, n_points=100_000, generator=torch.Generator().manual_seed(99))
        for s in levels
    ]
    floor, floor_se = ladder_floor(p)

    # A single minibatch loss is far too noisy to compare against the floor
    # (its sd across batches is ~0.07 here), so windowed means are what get
    # reported -- one early, one late, because the gap between them is the
    # point: the loss barely moves while the actual score error falls several
    # fold, which is only visible because the exact score exists.
    def window(lo: int, hi: int) -> tuple[float, float, int]:
        vals = [v for s, v in zip(hist.steps, hist.loss) if lo <= s <= hi]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        return m, math.sqrt(var / len(vals)), len(vals)

    early, early_se, n_early = window(1_000, 3_000)
    late, late_se, n_late = window(N_STEPS - 6_000, N_STEPS)
    print(
        f"\nirreducible DSM loss over the ladder: {floor:.4f} +- {floor_se:.4f}\n"
        f"  minibatch loss, steps 1k-3k:   {early:.4f} +- {early_se:.4f} (n={n_early})"
        f"   = floor + {100 * (early - floor) / floor:.1f}%\n"
        f"  minibatch loss, last 6k steps: {late:.4f} +- {late_se:.4f} (n={n_late})"
        f"   = floor + {100 * (late - floor) / floor:.1f}%"
    )

    # The same comparison without any Monte Carlo error on either side: a
    # frozen batch and its own exact optimum, on identical points.
    print(f"\nfrozen eval batch ({batch.x0.shape[0]:,} points), exact optimum {batch_floor:.4f}")
    print("  step   eval loss   excess over optimum   relative MSE   cosine")
    for s, ev_loss, rel, cos in zip(
        hist.eval_steps, hist.eval_loss, hist.eval_relative_mse, hist.eval_cosine
    ):
        if s in (1, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000, N_STEPS):
            print(
                f"{s:7d}   {ev_loss:9.4f}   {ev_loss - batch_floor:19.4f}   "
                f"{rel:12.4f}   {cos:6.4f}"
            )

    with (LOGS / "loss_floor.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "irreducible_loss"])
        w.writerows(zip(levels, floors))

    with (LOGS / "training.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        w.writerows(zip(hist.steps, hist.loss))
    with (LOGS / "training_eval.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "relative_mse", "cosine", "eval_loss"])
        w.writerows(
            zip(hist.eval_steps, hist.eval_relative_mse, hist.eval_cosine, hist.eval_loss)
        )
    with (LOGS / "score_error.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sigma", "mse", "relative_mse", "cosine",
                "cosine_weighted", "exact_norm", "near_zero_fraction",
            ],
        )
        w.writeheader()
        w.writerows(diag.as_rows())

    torch.save(
        {
            "state_dict": net.state_dict(),
            "config": {"dim": 2, "width": WIDTH, "depth": DEPTH},
            "sigma_min": SIGMA_MIN,
            "sigma_max": SIGMA_MAX,
            "seed": SEED,
            "n_steps": N_STEPS,
        },
        LOGS / "score_net.pt",
    )

    with (LOGS / "score_matching.json").open("w") as f:
        json.dump(
            {
                "parameters": count_parameters(net),
                "width": WIDTH,
                "depth": DEPTH,
                "n_steps": N_STEPS,
                "batch_size": BATCH,
                "lr": LR,
                "seed": SEED,
                "sigma_min": SIGMA_MIN,
                "sigma_max": SIGMA_MAX,
                "train_seconds": hist.seconds,
                "final_loss": hist.loss[-1],
                "early_training_loss": early,
                "early_training_loss_se": early_se,
                "late_training_loss": late,
                "late_training_loss_se": late_se,
                "irreducible_loss": floor,
                "irreducible_loss_se": floor_se,
                "eval_batch_points": batch.x0.shape[0],
                "eval_batch_floor": batch_floor,
                "eval_loss_final": hist.eval_loss[-1],
            },
            f,
            indent=2,
        )

    print("\nsigma      rel MSE     cosine   cos(wtd)  ||s_exact||   near-zero")
    for row in diag.as_rows():
        print(
            f"{row['sigma']:8.4f}  {row['relative_mse']:9.2e}  "
            f"{row['cosine']:8.4f}  {row['cosine_weighted']:8.4f}  "
            f"{row['exact_norm']:10.3f}  {row['near_zero_fraction']:9.3f}"
        )
    figures()


# --------------------------------------------------------------------------
# figures (replayable from the committed logs + checkpoint)
# --------------------------------------------------------------------------
def _late_minibatch_sd(tr: list[tuple[int, float]], after: int = 10_000) -> float:
    """Batch-to-batch sd of the training loss once it has settled.

    Plotted beside the excess risk to make the point quantitative: a single
    minibatch's noise is several times the entire remaining excess, so the
    improvement over the second half of training is not merely hard to see in
    a loss curve, it is below the resolution of one.
    """
    vals = [v for s, v in tr if s >= after]
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))



def load_net() -> ScoreNet:
    blob = torch.load(LOGS / "score_net.pt", weights_only=False)
    net = ScoreNet(**blob["config"])
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net


def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    with (LOGS / "score_error.csv").open() as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    sig = [r["sigma"] for r in rows]

    # -- error vs noise level ------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].loglog(sig, [r["relative_mse"] for r in rows], "o-", color="C0")
    ax[0].set_xlabel(r"noise level $\sigma$")
    ax[0].set_ylabel(r"relative MSE  $\|s_\theta - s\|^2 / \|s\|^2$")
    ax[0].set_title("Score error against the exact noised score")
    ax[0].grid(alpha=0.3, which="both")
    ax[1].semilogx(sig, [r["cosine"] for r in rows], "o-", color="C1", label="plain mean")
    ax[1].semilogx(
        sig, [r["cosine_weighted"] for r in rows], "s--", color="C2",
        label=r"weighted by $\|s\|$",
    )
    ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel(r"noise level $\sigma$")
    ax[1].set_ylabel("cosine similarity")
    ax[1].set_title("The mid-ladder dip is the metric, not the model")
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "score_error_vs_sigma.png", dpi=150)
    plt.close(fig)

    # -- training curves -----------------------------------------------------
    with (LOGS / "training.csv").open() as f:
        tr = [(int(r["step"]), float(r["loss"])) for r in csv.DictReader(f)]
    with (LOGS / "training_eval.csv").open() as f:
        ev = [
            (int(r["step"]), float(r["relative_mse"]), float(r["cosine"]), float(r["eval_loss"]))
            for r in csv.DictReader(f)
        ]
    with (LOGS / "score_matching.json").open() as f:
        meta = json.load(f)
    batch_floor = meta["eval_batch_floor"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(
        [s for s, _ in tr], [v for _, v in tr], color="0.7", lw=0.8,
        label="minibatch loss (512 pts)",
    )
    ax[0].plot(
        [s for s, _, _, _ in ev], [v for _, _, _, v in ev], color="C0", lw=1.6,
        label="frozen-batch loss (52k pts)",
    )
    ax[0].axhline(
        batch_floor, color="k", ls="--", lw=1,
        label=f"exact optimum on that batch ({batch_floor:.3f})",
    )
    ax[0].set_xlabel("step")
    ax[0].set_ylabel(r"DSM loss  $\|\epsilon_\theta - \epsilon\|^2$")
    ax[0].set_title("The loss looks flat after ~2k steps...")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # The same curve with the constant subtracted: the excess risk, which is
    # the only part of the loss that has anything to do with the model.
    ax[1].semilogy(
        [s for s, _, _, _ in ev], [v - batch_floor for _, _, _, v in ev],
        color="C0", lw=1.6, label="excess over the optimum",
    )
    sd = _late_minibatch_sd(tr)
    ax[1].axhline(
        sd, color="0.5", ls=":", lw=1.2,
        label=f"sd of one minibatch loss ({sd:.3f})",
    )
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("excess risk")
    ax[1].set_title("...but subtract the constant and it is still falling")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")

    (l0,) = ax[2].semilogy(
        [s for s, _, _, _ in ev], [m for _, m, _, _ in ev], "-", color="C0", lw=1.2,
        label="relative MSE (left)",
    )
    ax[2].set_xlabel("step")
    ax[2].set_ylabel("relative MSE (mean over 3 levels)")
    twin = ax[2].twinx()
    (l1,) = twin.plot(
        [s for s, _, _, _ in ev], [c for _, _, c, _ in ev], "--", color="C1", lw=1.2,
        label="cosine (right)",
    )
    twin.set_ylabel("cosine similarity")
    ax[2].set_title("Error against the exact score")
    ax[2].legend(handles=[l0, l1], loc="center right", fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "score_matching_training.png", dpi=150)
    plt.close(fig)

    # -- the fields themselves ----------------------------------------------
    net = load_net()
    p = target()
    lim = 7.0
    q = torch.linspace(-lim, lim, 25)
    gx, gy = torch.meshgrid(q, q, indexing="xy")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    # A finer grid for the density contours the arrows are drawn over.
    qf = torch.linspace(-lim, lim, 200)
    fx, fy = torch.meshgrid(qf, qf, indexing="xy")
    fine = torch.stack([fx.reshape(-1), fy.reshape(-1)], dim=1)

    fig, axes = plt.subplots(len(FIELD_SIGMAS), 3, figsize=(13, 4.2 * len(FIELD_SIGMAS)))
    for row, sigma in enumerate(FIELD_SIGMAS):
        noised = p.noised(sigma)
        exact = noised.score(pts)
        with torch.no_grad():
            learned = net(pts, sigma)
        dens = noised.log_prob(fine).exp().reshape(fx.shape)
        # One arrow scale per level, set by the exact field, so the two quiver
        # panels in a row are directly comparable in length as well as angle.
        scale = float(exact.norm(dim=1).max()) * 14
        for col, (field, name) in enumerate(((exact, "exact"), (learned, "learned"))):
            a = axes[row, col]
            a.contourf(fx, fy, dens, levels=12, cmap="Blues", alpha=0.7)
            a.quiver(
                gx, gy, field[:, 0].reshape(gx.shape), field[:, 1].reshape(gx.shape),
                scale=scale, width=0.004, color="0.15",
            )
            a.set_title(rf"{name} score over $p_\sigma$, $\sigma={sigma}$")
            a.set_xlim(-lim, lim)
            a.set_ylim(-lim, lim)
            a.set_aspect("equal")
        # Error normalized by the LEVEL's mean |s|, not pointwise: a pointwise
        # ratio divides by ~0 at the centre of the ring (where the exact score
        # vanishes by symmetry) and the single resulting 1e5 cell swamps the
        # colour scale. Same trap the weighted cosine exists for.
        a = axes[row, 2]
        ref = float(exact.norm(dim=1).mean())
        err = (learned - exact).norm(dim=1).reshape(gx.shape) / ref
        im = a.pcolormesh(gx, gy, err, cmap="magma", shading="nearest")
        fig.colorbar(im, ax=a)
        a.set_title(rf"$\|s_\theta - s\| \,/\, \overline{{\|s\|}}$, $\sigma={sigma}$")
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIGURES / "score_fields.png", dpi=150)
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
