"""The repo claims a reader can rebuild every figure from a clean clone.

``reproduce.sh`` runs each experiment's ``--figures`` path, which recomputes
nothing: it reads committed logs, sample tensors, and the trained score
network. That claim has two ways to rot silently, and neither shows up in any
other test:

1. An artifact stops being committed -- added to ``.gitignore``, or written
   during a run and never staged. The figures still build *here*, because the
   file is sitting in the working tree, and fail for everyone else.
2. ``ScoreNet``'s constructor or state-dict keys change. The weights in
   ``logs/score_net.pt`` were saved by an older definition, so every figure of
   the learned score -- and the whole learned-score half of the README --
   becomes unreproducible, while the training code still runs fine.

So: assert every artifact the replay reads is tracked by git, and load the
committed weights into the model the code builds *today*.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import sampling as sampling_experiment  # noqa: E402

# Every file read by an experiment's figures() path, by experiment. Kept
# explicit rather than globbed from logs/ so that adding an artifact without
# committing it is a failure and not a silently smaller test.
REPLAY_ARTIFACTS = {
    "score_matching.py": [
        "logs/score_error.csv",
        "logs/training.csv",
        "logs/training_eval.csv",
        "logs/score_matching.json",
        "logs/score_net.pt",
    ],
    "sampling.py": [
        "logs/samples.pt",
        "logs/sampling.json",
        "logs/sampler_sweep.csv",
    ],
    "ode.py": [
        "logs/ode_sweep.csv",
        "logs/ode_likelihood.csv",
        "logs/ode.json",
        "logs/ode_figures.pt",
        "logs/sampling.json",
    ],
    "schedules.py": [
        "logs/schedule_sweep.csv",
        "logs/schedules.json",
        "logs/sampling.json",
    ],
}

FIGURES = [
    "figures/score_fields.png",
    "figures/score_error_vs_sigma.png",
    "figures/score_matching_training.png",
    "figures/samples.png",
    "figures/sampler_accuracy.png",
    "figures/ode_vs_sde.png",
    "figures/ode_likelihood.png",
    "figures/schedules.png",
]


def tracked_files() -> set[str] | None:
    """Paths git knows about, or None when this is not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return set(out.split())


@pytest.mark.parametrize("experiment", sorted(REPLAY_ARTIFACTS))
def test_every_artifact_the_figure_replay_reads_is_committed(experiment):
    paths = REPLAY_ARTIFACTS[experiment]
    missing = [p for p in paths if not (ROOT / p).exists()]
    assert not missing, f"{experiment} --figures reads absent files: {missing}"

    tracked = tracked_files()
    if tracked is None:
        pytest.skip("not a git checkout")
    untracked = [p for p in paths if p not in tracked]
    assert not untracked, (
        f"{experiment} --figures reads uncommitted files: {untracked} -- they "
        "exist here and would be missing from a clean clone"
    )


def test_every_figure_the_readme_shows_is_committed():
    tracked = tracked_files()
    if tracked is None:
        pytest.skip("not a git checkout")
    assert not [f for f in FIGURES if f not in tracked]


def test_the_committed_weights_load_into_todays_model():
    """The failure mode a training run would never catch."""
    net = sampling_experiment.load_net()
    x = torch.tensor([[0.0, 0.0], [4.0, 0.0], [-3.0, 3.0]])
    s = net(x, 1.0)
    assert s.shape == x.shape
    assert torch.isfinite(s).all()


def test_the_loaded_network_is_the_one_the_readme_reports():
    """87,298 parameters, quoted in the README as "87k"."""
    net = sampling_experiment.load_net()
    assert sum(p.numel() for p in net.parameters()) == 87298


def test_the_network_points_roughly_at_the_ring_where_the_score_is_large():
    """A weight file can load, be the right size, and still be scrambled.

    Well outside the ring the score points inward at every noise level, so a
    single dot product against the exact field separates "loaded" from
    "loaded and meaningful" without re-measuring what the score-error tables
    already measure.
    """
    net = sampling_experiment.load_net()
    p = sampling_experiment.target()
    x = torch.tensor([[7.0, 0.0], [0.0, 7.0], [-5.0, -5.0]])
    for sigma in (0.2, 1.0, 4.0):
        exact = p.noised(sigma).score(x)
        with torch.no_grad():
            learned = net(x, sigma)
        cos = torch.nn.functional.cosine_similarity(learned, exact, dim=1)
        assert (cos > 0.9).all(), f"sigma={sigma}: {cos}"
