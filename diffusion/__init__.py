"""diffusion-from-scratch: score-based generative modeling on targets whose
score is known exactly.

A diffusion model is an estimate of ``grad log p``. This repo builds one from
that sentence outward -- Langevin dynamics that need only the score
(:mod:`diffusion.langevin`), denoising score matching that learns it, the
reverse-time samplers that turn it back into draws, and the deterministic
probability-flow ODE that also hands back likelihoods (:mod:`diffusion.ode`)
-- on 2D targets chosen so that the truth is available in closed form
(:mod:`diffusion.targets`).

The choice of targets is the design of the repo. A Gaussian mixture stays a
Gaussian mixture when noised, so its score is analytic at *every* noise level:
a learned score can be reported as an error against truth, and a sampler can
be run with the exact score as a control to separate estimation error from
discretization error. Neither check exists on an image dataset, which is why
the toys are the point rather than a shortcut.
"""

from diffusion import dsm, langevin, metrics, model, ode, samplers, targets  # noqa: F401
from diffusion.dsm import (  # noqa: F401
    EvalBatch,
    ScoreDiagnostics,
    TrainHistory,
    dsm_loss,
    eval_dsm_floor,
    eval_dsm_loss,
    explicit_score_matching_loss,
    fixed_eval_batch,
    irreducible_dsm_loss,
    sample_log_uniform_sigmas,
    score_diagnostics,
    train_score_model,
)
from diffusion.langevin import (  # noqa: F401
    SIGMA_SCHEDULES,
    annealed_langevin,
    cosine_sigmas,
    geometric_sigmas,
    linear_sigmas,
    sigma_ladder,
    stability_bound,
    ula_gaussian_variance,
    unadjusted_langevin,
)
from diffusion.metrics import (  # noqa: F401
    MMDTest,
    SlicedWasserstein,
    gaussian_rbf_mmd2,
    gaussian_sliced_wasserstein2,
    median_heuristic_bandwidth,
    mmd_permutation_test,
    rbf_kernel_matrix,
    rbf_mmd2,
    sliced_wasserstein,
)
from diffusion.model import ScoreNet, SinusoidalEmbedding, count_parameters  # noqa: F401
from diffusion.ode import (  # noqa: F401
    LogProbResult,
    VectorField,
    euler_step,
    exact_divergence,
    gaussian_pf_ode_scale,
    gaussian_probes,
    hutchinson_divergence,
    integrate,
    pf_ode_field,
    pf_ode_log_prob,
    pf_ode_nfe,
    pf_ode_sample,
    rademacher_probes,
    rk4_step,
)
from diffusion.samplers import ddpm_ancestral, ddpm_gaussian_variance  # noqa: F401
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
    "linear_sigmas",
    "cosine_sigmas",
    "SIGMA_SCHEDULES",
    "sigma_ladder",
    "stability_bound",
    "ula_gaussian_variance",
    "SinusoidalEmbedding",
    "ScoreNet",
    "count_parameters",
    "sample_log_uniform_sigmas",
    "dsm_loss",
    "explicit_score_matching_loss",
    "irreducible_dsm_loss",
    "EvalBatch",
    "fixed_eval_batch",
    "eval_dsm_loss",
    "eval_dsm_floor",
    "score_diagnostics",
    "ScoreDiagnostics",
    "TrainHistory",
    "train_score_model",
    "ddpm_ancestral",
    "ddpm_gaussian_variance",
    "VectorField",
    "euler_step",
    "rk4_step",
    "integrate",
    "pf_ode_field",
    "pf_ode_sample",
    "pf_ode_nfe",
    "gaussian_pf_ode_scale",
    "rademacher_probes",
    "gaussian_probes",
    "exact_divergence",
    "hutchinson_divergence",
    "LogProbResult",
    "pf_ode_log_prob",
    "median_heuristic_bandwidth",
    "rbf_kernel_matrix",
    "rbf_mmd2",
    "MMDTest",
    "mmd_permutation_test",
    "SlicedWasserstein",
    "sliced_wasserstein",
    "gaussian_rbf_mmd2",
    "gaussian_sliced_wasserstein2",
    "dsm",
    "langevin",
    "metrics",
    "model",
    "ode",
    "samplers",
    "targets",
]
