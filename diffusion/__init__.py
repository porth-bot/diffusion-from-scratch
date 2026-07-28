"""diffusion-from-scratch: score-based generative modeling on targets whose
score is known exactly.

A diffusion model is an estimate of ``grad log p``. This repo builds one from
that sentence outward -- Langevin dynamics that need only the score
(:mod:`diffusion.langevin`), denoising score matching that learns it, and the
reverse-time samplers that turn it back into draws -- on 2D targets chosen so
that the truth is available in closed form (:mod:`diffusion.targets`).

The choice of targets is the design of the repo. A Gaussian mixture stays a
Gaussian mixture when noised, so its score is analytic at *every* noise level:
a learned score can be reported as an error against truth, and a sampler can
be run with the exact score as a control to separate estimation error from
discretization error. Neither check exists on an image dataset, which is why
the toys are the point rather than a shortcut.
"""

from diffusion import dsm, langevin, model, targets  # noqa: F401
from diffusion.dsm import (  # noqa: F401
    ScoreDiagnostics,
    TrainHistory,
    dsm_loss,
    explicit_score_matching_loss,
    sample_log_uniform_sigmas,
    score_diagnostics,
    train_score_model,
)
from diffusion.langevin import (  # noqa: F401
    annealed_langevin,
    geometric_sigmas,
    stability_bound,
    ula_gaussian_variance,
    unadjusted_langevin,
)
from diffusion.model import ScoreNet, SinusoidalEmbedding, count_parameters  # noqa: F401
from diffusion.targets import (  # noqa: F401
    Crescent,
    Gaussian,
    GaussianMixture,
    Mixture,
    Target,
    TwoMoons,
    finite_difference_score,
)

__all__ = [
    "Target",
    "Gaussian",
    "Mixture",
    "GaussianMixture",
    "Crescent",
    "TwoMoons",
    "finite_difference_score",
    "unadjusted_langevin",
    "annealed_langevin",
    "geometric_sigmas",
    "stability_bound",
    "ula_gaussian_variance",
    "SinusoidalEmbedding",
    "ScoreNet",
    "count_parameters",
    "sample_log_uniform_sigmas",
    "dsm_loss",
    "explicit_score_matching_loss",
    "score_diagnostics",
    "ScoreDiagnostics",
    "TrainHistory",
    "train_score_model",
    "dsm",
    "langevin",
    "model",
    "targets",
]
