#!/usr/bin/env bash
#
# Regenerate every figure in figures/ from the committed logs and weights.
#
#     ./reproduce.sh              # tests, then all 12 figures: well under a minute
#     PYTHON=/path/to/python ./reproduce.sh
#
# NO TRAINING AND NO SAMPLING happens here, and that is the point. Running this
# repo end to end is roughly an hour of CPU -- the sampler sweep alone draws
# 16,000 samples per arm across eight arms and six ladder refinements, and the
# ODE sweep solves six budgets times three integrators. So the artifacts ship
# with the repo -- CSV sweeps and JSON summaries in logs/, the trained score
# network in logs/score_net.pt, the sample tensors in logs/samples.pt and
# logs/ode_figures.pt -- and each experiment's --figures flag turns them back
# into its figures without recomputing anything.
#
# Why tensors and not just CSVs: a figure of a *number* (the accuracy-vs-budget
# panels, the schedule curves) can come from a sweep log, but a figure of a
# *cloud* (figures/samples.png) or of a *field* (figures/score_fields.png) is a
# picture of actual draws and of the network itself, and only the tensors and
# the weights reproduce those.
#
# To actually recompute, run the experiments without --figures. Each prints its
# own cost; the README's Reproduce section lists them:
#     python experiments/score_matching.py   # ~2 min  (trains the score net)
#     python experiments/sampling.py         # ~20 min (8 arms + ladder sweep)
#     python experiments/ode.py              # ~25 min (NFE sweep + likelihoods)
#     python experiments/schedules.py        # ~15 min (3 ladders x 5 budgets x 3 seeds)
#     python experiments/vp_ve.py            # ~2 min  (VP vs VE prior sweep)
#     python experiments/vp_coords.py --train  # ~35 min (40 paired training runs)
#     python experiments/guidance.py         # ~19 min (classifier-free guidance sweep)
#     python experiments/overlap.py          # ~3 min  (guidance vs component overlap)
#
# Determinism: the replay is pure post-processing of committed files, so it is
# exact -- rerunning it leaves the working tree clean. Recomputation is seeded
# and replays on the same torch build and CPU, but torch guarantees no bitwise
# determinism across versions or hardware, which is why the artifacts are
# committed rather than regenerated.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "${PY}" ]; then
    if [ -x .venv/bin/python ]; then PY="$PWD/.venv/bin/python"; else PY="python3"; fi
fi

echo "=================================================================="
echo "diffusion-from-scratch: figure reproduction (no training, no sampling)"
echo "python:     $("$PY" -V 2>&1)  ($PY)"
"$PY" - <<'EOF'
import importlib
for name in ("torch", "numpy", "matplotlib"):
    try:
        m = importlib.import_module(name)
        print(f"{name+':':11s} {m.__version__}")
    except ImportError:
        print(f"{name+':':11s} MISSING")
EOF
echo "=================================================================="

started=$SECONDS

step() {  # step <label> <script> [args...]
    local label="$1"; shift
    echo
    echo "------------------------------------------------------------------"
    echo ">>> $label"
    echo "------------------------------------------------------------------"
    local t0=$SECONDS
    "$PY" "$@"
    echo "    [${label}: $((SECONDS - t0))s]"
}

# The suite includes the checks that every experiment's --figures path reads
# only committed artifacts and that the committed weights still load into the
# model the code builds today -- the failures this script exists to catch.
step "test suite" -m pytest -q
step "score fields, score error vs sigma, training curve" experiments/score_matching.py --figures
step "samples, sampler accuracy vs budget" experiments/sampling.py --figures
step "SDE vs ODE, likelihoods" experiments/ode.py --figures
step "noise schedules" experiments/schedules.py --figures
step "EMA-of-weights ablation" experiments/ema.py --figures
step "VP vs VE: one chain, two coordinate systems" experiments/vp_ve.py --figures
step "classifier-free guidance: what it samples from" experiments/guidance.py --figures
step "training in VP coordinates" experiments/vp_coords.py --figures
step "guidance against component overlap" experiments/overlap.py --figures

echo
echo "=================================================================="
echo "done in $((SECONDS - started))s. figures/:"
ls -1 figures/
echo "=================================================================="
