"""Deterministic sampling and likelihoods, against a target that knows the answer.

Two questions, one machinery (:mod:`diffusion.ode`).

**Is the noise doing any work?** The probability-flow ODE has the same marginals
as the stochastic reverse process, so at infinite resolution the two samplers
agree. At finite resolution they do not, and which one is ahead is an empirical
question with a cost axis attached: the ancestral sampler spends one score
evaluation per level, RK4 spends four per step and buys fourth-order accuracy.
This experiment puts all three (ancestral, PF-ODE with Euler, PF-ODE with RK4)
on the same *number of function evaluations* axis, with the exact score as the
control, so "the ODE is better" cannot hide a factor of four.

**What is the model's likelihood, and is it right?** The flow is invertible, so
the change-of-variables formula turns it into a density -- and because the
noised mixture has a closed-form density, that estimate has a ground truth. The
same estimate run on the exact score measures integration error; run on the
learned score it is a genuine normalized model density, so its average gap to
the truth over exact draws is a Monte Carlo estimate of
``KL(p_{sigma_min} || model)`` and must be positive. Swapping the top-of-ladder
prior in and out splits that KL into the part caused by starting the sampler at
``N(0, sigma_max^2 I)`` instead of ``p_{sigma_max}`` and the part caused by the
score being learned -- a decomposition in nats, of exactly the two error
sources Day 25 could only bound with two-sample metrics.

Run
---
    python experiments/ode.py            # sample, measure likelihoods, log, plot
    python experiments/ode.py --figures  # replot from committed logs
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

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import sampling  # noqa: E402  -- the Day 25 experiment's target, net loader and metrics

from diffusion.langevin import geometric_sigmas  # noqa: E402
from diffusion.model import ScoreNet  # noqa: E402
from diffusion.ode import (  # noqa: E402
    STEP_EVALUATIONS,
    pf_ode_log_prob,
    pf_ode_nfe,
    pf_ode_sample,
)
from diffusion.samplers import ddpm_ancestral  # noqa: E402
from diffusion.targets import GaussianMixture  # noqa: E402

LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"

SIGMA_MIN, SIGMA_MAX = sampling.SIGMA_MIN, sampling.SIGMA_MAX
N_SAMPLES = sampling.N_SAMPLES

# Evaluation budgets. The ancestral sampler and Euler take one evaluation per
# node, RK4 four per step, so a budget of 40 buys 40 ancestral levels or only
# 10 RK4 nodes -- which is the whole point of plotting against NFE.
NFE_BUDGETS = [9, 17, 41, 81, 201, 401]

# Likelihoods: fewer points, because each one carries a full ODE solve with a
# divergence at every stage of every step.
N_LL_POINTS = 2_000
LL_NODES = 41  # RK4 nodes for the reported likelihood numbers
LL_NODE_SWEEP = [3, 5, 9, 17, 41, 81]
LL_PROBE_SWEEP = [1, 2, 4, 8, 16, 32]

TRAJECTORY_POINTS = 240


def learned_score_with_grad(net: ScoreNet):
    """``s_theta(x, sigma)`` that autograd can differentiate.

    :func:`sampling.learned_score` wraps the network in ``no_grad``, which is
    right for the samplers and fatal for the divergence -- so the likelihood
    path takes this one, and :mod:`diffusion.ode` raises rather than silently
    reporting a zero divergence if the wrong one is passed.
    """

    def s(x: torch.Tensor, sigma: float) -> torch.Tensor:
        return net(x, sigma)

    return s


def rk4_nodes_for(nfe: int) -> int:
    """Largest node count whose RK4 cost fits in ``nfe`` evaluations."""
    return max(2, (nfe - 1) // STEP_EVALUATIONS["rk4"] + 1)


# --------------------------------------------------------------------------
# part A: the same budget, three dynamics
# --------------------------------------------------------------------------
def sampler_sweep(p: GaussianMixture, scores: dict, reference: torch.Tensor) -> list[dict]:
    rows: list[dict] = []
    print("SDE vs ODE at matched score evaluations")
    print("  budget  sampler   score     nfe      SW2   TV(w)   mode sd   seconds")
    for budget in NFE_BUDGETS:
        plans = [
            ("ancestral", "sde", dict(levels=budget)),
            ("pf-ode euler", "euler", dict(levels=budget)),
            ("pf-ode rk4", "rk4", dict(levels=rk4_nodes_for(budget))),
        ]
        for name, kind, cfg in plans:
            ladder = geometric_sigmas(SIGMA_MAX, SIGMA_MIN, cfg["levels"])
            for label, s_fn in scores.items():
                x0 = sampling.prior_init(N_SAMPLES, 12)
                t0 = time.perf_counter()
                if kind == "sde":
                    x = ddpm_ancestral(
                        s_fn, x0, ladder, generator=torch.Generator().manual_seed(1)
                    )
                    nfe = len(ladder)
                else:
                    x = pf_ode_sample(s_fn, x0, ladder, method=kind)
                    nfe = pf_ode_nfe(len(ladder), kind)
                seconds = time.perf_counter() - t0
                m = sampling.cheap_measure(x, reference, seed=7)
                row = {
                    "budget": budget, "sampler": name, "score": label,
                    "init": "N(0,s_max^2)", "nodes": cfg["levels"], "nfe": nfe,
                    "seconds": seconds,
                    **{k: m[k] for k in ("sw2", "sw2_stderr", "weight_tv", "mode_sd")},
                }
                rows.append(row)
                print(
                    f"  {budget:6d}  {name:13s} {label:8s} {nfe:5d}  {row['sw2']:7.4f}  "
                    f"{row['weight_tv']:6.4f}  {row['mode_sd']:8.4f}  {seconds:7.1f}"
                )

    # -- the control that names the ODE's residual error ---------------------
    # The deterministic sampler converges to something slightly *narrower* than
    # the target and stays there no matter how fine the ladder gets, which can
    # only be the prior: the flow transports whatever it is handed, and it is
    # handed N(0, sigma_max^2 I) rather than p_{sigma_max}. Feeding it an exact
    # top-of-ladder draw removes that term and nothing else. (The stochastic
    # sampler's version of this row is in Day 25's sampler_arms.csv.)
    print("\ncontrol: the same ODE started from an exact p_sigma_max draw")
    top = p.noised(SIGMA_MAX).sample(N_SAMPLES, generator=torch.Generator().manual_seed(11))
    budget = NFE_BUDGETS[-2]
    ladder = geometric_sigmas(SIGMA_MAX, SIGMA_MIN, rk4_nodes_for(budget))
    for label, s_fn in scores.items():
        t0 = time.perf_counter()
        x = pf_ode_sample(s_fn, top, ladder, method="rk4")
        seconds = time.perf_counter() - t0
        m = sampling.cheap_measure(x, reference, seed=7)
        row = {
            "budget": budget, "sampler": "pf-ode rk4", "score": label,
            "init": "p_sigma_max", "nodes": len(ladder),
            "nfe": pf_ode_nfe(len(ladder), "rk4"), "seconds": seconds,
            **{k: m[k] for k in ("sw2", "sw2_stderr", "weight_tv", "mode_sd")},
        }
        rows.append(row)
        print(
            f"  {budget:6d}  pf-ode rk4    {label:8s} {row['nfe']:5d}  {row['sw2']:7.4f}  "
            f"{row['weight_tv']:6.4f}  {row['mode_sd']:8.4f}  {seconds:7.1f}"
        )
    return rows


# --------------------------------------------------------------------------
# part B: likelihoods, against the closed form
# --------------------------------------------------------------------------
def likelihood_study(p: GaussianMixture, net: ScoreNet) -> tuple[list[dict], dict]:
    bottom, top = p.noised(SIGMA_MIN), p.noised(SIGMA_MAX)
    pts = bottom.sample(N_LL_POINTS, generator=torch.Generator().manual_seed(99))
    truth = bottom.log_prob(pts)

    s_exact = sampling.exact_score(p)
    s_learned = learned_score_with_grad(net)
    exact_top = (lambda z: top.log_prob(z), "p_sigma_max")
    gauss_top = (
        lambda z: -0.5 * (z * z).sum(dim=1) / SIGMA_MAX**2
        - math.log(2.0 * math.pi * SIGMA_MAX**2),
        "N(0,s_max^2)",
    )

    rows: list[dict] = []

    def record(kind: str, score: str, prior: str, nodes: int, probes: int, est) -> dict:
        err = est.log_prob - truth
        row = {
            "kind": kind, "score": score, "prior": prior, "nodes": nodes,
            "probes": probes, "nfe": est.nfe,
            "mean_abs_err": float(err.abs().mean()),
            "max_abs_err": float(err.abs().max()),
            "mean_gap": float(err.mean()),            # -KL when the flow is a model
            "gap_stderr": float(err.std() / math.sqrt(err.shape[0])),
            "mean_log_prob": float(est.log_prob.mean()),
            "probe_stderr": float(est.stderr.mean()),
        }
        rows.append(row)
        return row

    # -- integration error, isolated by the exact divergence -----------------
    print("\nIntegration error (exact score, exact divergence, truth known)")
    print("  nodes   nfe   mean |err|    max |err|")
    for nodes in LL_NODE_SWEEP:
        ladder = geometric_sigmas(SIGMA_MAX, SIGMA_MIN, nodes)
        est = pf_ode_log_prob(s_exact, pts, ladder, exact_top[0], exact=True)
        row = record("integration", "exact", exact_top[1], nodes, 0, est)
        print(f"  {nodes:5d} {est.nfe:5d}   {row['mean_abs_err']:.3e}   {row['max_abs_err']:.3e}")

    # -- estimator variance, at a fixed (fine) ladder -------------------------
    ladder = geometric_sigmas(SIGMA_MAX, SIGMA_MIN, LL_NODES)
    print(f"\nHutchinson error at {LL_NODES} nodes (Rademacher probes)")
    print("  probes   mean |err|   reported s.e.")
    for k in LL_PROBE_SWEEP:
        est = pf_ode_log_prob(
            s_exact, pts, ladder, exact_top[0], n_probes=k,
            generator=torch.Generator().manual_seed(5),
        )
        row = record("hutchinson", "exact", exact_top[1], LL_NODES, k, est)
        print(f"  {k:6d}   {row['mean_abs_err']:.4f}      {row['probe_stderr']:.4f}")

    # -- the KL decomposition -----------------------------------------------
    # Each of these three flows is a normalized density, so mean(log p - log q)
    # over exact draws estimates KL(p || q) >= 0. Comparing them isolates what
    # each departure from the truth costs, in nats.
    print(f"\nKL decomposition ({N_LL_POINTS} exact draws, exact divergence)")
    kl = {}
    for tag, s_fn, (top_fn, top_name) in (
        ("exact score, exact prior", s_exact, exact_top),
        ("exact score, Gaussian prior", s_exact, gauss_top),
        ("learned score, Gaussian prior", s_learned, gauss_top),
    ):
        est = pf_ode_log_prob(s_fn, pts, ladder, top_fn, exact=True)
        row = record("kl", "exact" if s_fn is s_exact else "learned", top_name, LL_NODES, 0, est)
        kl[tag] = -row["mean_gap"]
        print(
            f"  {tag:32s} KL ~ {-row['mean_gap']:+.4f} +- {row['gap_stderr']:.4f} nats   "
            f"E[log p] = {row['mean_log_prob']:.4f}"
        )

    meta = {
        "n_points": N_LL_POINTS,
        "ll_nodes": LL_NODES,
        "sigma_min": SIGMA_MIN,
        "sigma_max": SIGMA_MAX,
        "entropy_reference": float(truth.mean()),
        "kl_nats": kl,
    }
    return rows, meta


# --------------------------------------------------------------------------
def run() -> None:
    LOGS.mkdir(exist_ok=True)
    p = sampling.target()
    net = sampling.load_net()
    scores = {"exact": sampling.exact_score(p), "learned": sampling.learned_score(net)}

    reference = p.sample(N_SAMPLES, generator=torch.Generator().manual_seed(4321))
    rows = sampler_sweep(p, scores, reference)
    with (LOGS / "ode_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ll_rows, meta = likelihood_study(p, net)
    with (LOGS / "ode_likelihood.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ll_rows[0].keys()))
        w.writeheader()
        w.writerows(ll_rows)

    # -- what the figures replay from ---------------------------------------
    ladder = geometric_sigmas(SIGMA_MAX, SIGMA_MIN, 51)
    x0 = sampling.prior_init(TRAJECTORY_POINTS, 12)
    _, traj = pf_ode_sample(scores["exact"], x0, ladder, trajectory=True)
    _, traj_learned = pf_ode_sample(scores["learned"], x0, ladder, trajectory=True)

    bottom = p.noised(SIGMA_MIN)
    pts = bottom.sample(N_LL_POINTS, generator=torch.Generator().manual_seed(99))
    top = p.noised(SIGMA_MAX)
    scatter = {
        "truth": bottom.log_prob(pts),
        "exact_div": pf_ode_log_prob(
            scores["exact"], pts, geometric_sigmas(SIGMA_MAX, SIGMA_MIN, LL_NODES),
            lambda z: top.log_prob(z), exact=True,
        ).log_prob,
        "one_probe": pf_ode_log_prob(
            scores["exact"], pts, geometric_sigmas(SIGMA_MAX, SIGMA_MIN, LL_NODES),
            lambda z: top.log_prob(z), n_probes=1,
            generator=torch.Generator().manual_seed(5),
        ).log_prob,
        "learned": pf_ode_log_prob(
            learned_score_with_grad(net), pts,
            geometric_sigmas(SIGMA_MAX, SIGMA_MIN, LL_NODES),
            lambda z: -0.5 * (z * z).sum(dim=1) / SIGMA_MAX**2
            - math.log(2.0 * math.pi * SIGMA_MAX**2),
            exact=True,
        ).log_prob,
    }
    torch.save(
        {"trajectory_exact": traj, "trajectory_learned": traj_learned, "scatter": scatter},
        LOGS / "ode_figures.pt",
    )
    with (LOGS / "ode.json").open("w") as f:
        json.dump(meta, f, indent=2)
    figures()


# --------------------------------------------------------------------------
def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    p = sampling.target()
    with (LOGS / "ode_sweep.csv").open() as f:
        sweep = [
            {k: (v if k in ("sampler", "score", "init") else float(v)) for k, v in r.items()}
            for r in csv.DictReader(f)
        ]
    # The control row is a table entry, not a point on a budget curve.
    control = [r for r in sweep if r["init"] != "N(0,s_max^2)"]
    sweep = [r for r in sweep if r["init"] == "N(0,s_max^2)"]
    with (LOGS / "ode_likelihood.csv").open() as f:
        ll = [
            {k: (v if k in ("kind", "score", "prior") else float(v)) for k, v in r.items()}
            for r in csv.DictReader(f)
        ]
    with (LOGS / "sampling.json").open() as f:
        day25 = json.load(f)
    with (LOGS / "ode.json").open() as f:
        meta = json.load(f)
    blob = torch.load(LOGS / "ode_figures.pt", weights_only=False)

    styles = {
        ("ancestral", "exact"): ("o-", "C0"), ("ancestral", "learned"): ("o--", "C0"),
        ("pf-ode euler", "exact"): ("^-", "C2"), ("pf-ode euler", "learned"): ("^--", "C2"),
        ("pf-ode rk4", "exact"): ("s-", "C3"), ("pf-ode rk4", "learned"): ("s--", "C3"),
    }

    def series(sampler, score, key):
        rows = sorted(
            (r for r in sweep if r["sampler"] == sampler and r["score"] == score),
            key=lambda r: r["nfe"],
        )
        return [r["nfe"] for r in rows], [r[key] for r in rows]

    # -- figure 1: the tradeoff ---------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4))
    for (sampler, score), (fmt, colour) in styles.items():
        xs, ys = series(sampler, score, "sw2")
        ax[0].loglog(xs, ys, fmt, color=colour, ms=5, lw=1.4,
                     label=f"{sampler}, {score}")
    ax[0].axhline(day25["sw2_floor_mean"], color="k", ls=":", lw=1)
    ax[0].axhspan(
        min(day25["sw2_floor_values"]), max(day25["sw2_floor_values"]), color="k", alpha=0.10
    )
    ax[0].annotate(
        "two independent exact draws", xy=(NFE_BUDGETS[-1], day25["sw2_floor_mean"]),
        xytext=(0, 8), textcoords="offset points", ha="right", fontsize=8,
    )
    ax[0].set_xlabel("score evaluations (NFE)")
    ax[0].set_ylabel(r"sliced $W_2$ against the exact sampler")
    ax[0].set_title("Same budget, three dynamics")
    ax[0].legend(fontsize=7.5)
    ax[0].grid(alpha=0.3, which="both")

    for (sampler, score), (fmt, colour) in styles.items():
        xs, ys = series(sampler, score, "mode_sd")
        ax[1].semilogx(xs, ys, fmt, color=colour, ms=5, lw=1.4)
    ax[1].axhline(sampling.RING_STD, color="k", ls="--", lw=1, label="target (0.4)")
    for r in control:
        if r["score"] == "exact":
            ax[1].plot(
                r["nfe"], r["mode_sd"], "*", color="C3", ms=13,
                label=r"rk4, exact score, exact $p_{\sigma_{max}}$ start",
            )
    ax[1].set_xlabel("score evaluations (NFE)")
    ax[1].set_ylabel("within-mode standard deviation")
    ax[1].set_title("Mode width: opposite biases, and a prior")
    # The 3-node RK4 points sit at 0.81 / 0.76 and would flatten everything
    # else; the interesting range is the last few per cent.
    ax[1].set_ylim(0.37, 0.50)
    ax[1].annotate(
        "rk4 at 3 nodes: 0.81 (off scale)", xy=(9, 0.494), fontsize=7, color="C3",
    )
    ax[1].legend(fontsize=7.5, loc="center right")
    ax[1].grid(alpha=0.3)

    lim = 9.0
    q = torch.linspace(-6.0, 6.0, 200)
    gx, gy = torch.meshgrid(q, q, indexing="xy")
    dens = p.log_prob(torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)).exp().reshape(gx.shape)
    traj = blob["trajectory_exact"]
    ax[2].contour(gx, gy, dens, levels=6, cmap="Blues", linewidths=0.7)
    for i in range(traj.shape[1]):
        ax[2].plot(traj[:, i, 0], traj[:, i, 1], "-", color="0.3", lw=0.4, alpha=0.5)
    ax[2].scatter(traj[0, :, 0], traj[0, :, 1], s=6, color="C1", label=r"prior draw ($\sigma_{max}$)")
    ax[2].scatter(traj[-1, :, 0], traj[-1, :, 1], s=6, color="C3", label="sample")
    ax[2].set_xlim(-lim, lim)
    ax[2].set_ylim(-lim, lim)
    ax[2].set_aspect("equal")
    ax[2].set_title("Deterministic transport, exact score")
    ax[2].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "ode_vs_sde.png", dpi=150)
    plt.close(fig)

    # -- figure 2: likelihoods ----------------------------------------------
    sc = blob["scatter"]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4))
    lo = float(min(sc["truth"].min(), sc["learned"].min()))
    hi = float(max(sc["truth"].max(), sc["learned"].max()))
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="exact")
    ax[0].scatter(sc["truth"], sc["one_probe"], s=4, alpha=0.25, color="C2",
                  label="exact score, 1 probe")
    ax[0].scatter(sc["truth"], sc["learned"], s=4, alpha=0.25, color="C3",
                  label="learned score, exact div.")
    ax[0].scatter(sc["truth"], sc["exact_div"], s=4, alpha=0.35, color="C0",
                  label="exact score, exact div.")
    ax[0].set_xlabel(r"exact $\log p_{\sigma_{min}}(x)$")
    ax[0].set_ylabel("transported estimate")
    ax[0].set_title("Change of variables against the closed form")
    ax[0].legend(fontsize=7.5, loc="upper left")
    ax[0].grid(alpha=0.3)

    integ = sorted((r for r in ll if r["kind"] == "integration"), key=lambda r: r["nodes"])
    ax[1].loglog([r["nodes"] for r in integ], [r["mean_abs_err"] for r in integ],
                 "o-", color="C0", label="measured")
    ref_n = [r["nodes"] for r in integ]
    anchor = integ[1]["mean_abs_err"] * integ[1]["nodes"] ** 4
    ax[1].loglog(ref_n, [anchor / n**4 for n in ref_n], ":", color="0.4",
                 label=r"$h^4$ reference")
    ax[1].set_xticks(ref_n)
    ax[1].set_xticklabels([str(int(n)) for n in ref_n])
    ax[1].minorticks_off()
    ax[1].set_xlabel("RK4 nodes")
    ax[1].set_ylabel(r"mean $|\hat{\log p} - \log p|$ (nats)")
    ax[1].set_title("Integration error, exact divergence")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")

    hut = sorted((r for r in ll if r["kind"] == "hutchinson"), key=lambda r: r["probes"])
    ks = [r["probes"] for r in hut]
    ax[2].loglog(ks, [r["mean_abs_err"] for r in hut], "s-", color="C2", label="measured")
    # k = 1 has no standard error to report (one draw), so the series starts at 2.
    ax[2].loglog(
        [r["probes"] for r in hut if r["probes"] > 1],
        [r["probe_stderr"] for r in hut if r["probes"] > 1],
        "^-", color="C1", label="reported standard error",
    )
    anchor = hut[0]["mean_abs_err"]
    ax[2].loglog(ks, [anchor / math.sqrt(k) for k in ks], ":", color="0.4",
                 label=r"$1/\sqrt{k}$ reference")
    floor = min(r["mean_abs_err"] for r in integ if r["nodes"] == meta["ll_nodes"])
    ax[2].axhline(floor, color="k", ls="--", lw=1, label="integration error at this ladder")
    ax[2].set_xlabel("Hutchinson probes $k$")
    ax[2].set_ylabel(r"mean $|\hat{\log p} - \log p|$ (nats)")
    ax[2].set_title("Estimator variance, and the floor under it")
    ax[2].legend(fontsize=7.5)
    ax[2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "ode_likelihood.png", dpi=150)
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
