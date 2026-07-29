# diffusion-from-scratch

**Work in progress.** Scaffold and core; the full README (measured results,
figures, limitations) lands when the repo is finished.

Score-based generative modeling built from the score outward, in PyTorch, on
2D targets whose scores are known in **closed form**.

A diffusion model is an estimate of one object:

$$s(x) = \nabla_x \log p(x).$$

Everything else — Langevin sampling, denoising score matching, the reverse-time
SDE, the probability-flow ODE — is written in terms of it. So the design
decision this repo makes first is: *use targets where the truth is available*.
A 2D Gaussian mixture has an analytic score, exact moments, and an exact
sampler, and — the property everything here rests on — it **stays a Gaussian
mixture when noised**, so the score of the noised marginal is closed-form at
every noise level. A learned score can therefore be reported as an *error
against truth* rather than as a picture that looks about right, and a sampler
can be run with the exact score as a control, which separates score-estimation
error from discretization error. Neither check exists on an image dataset.
That is why the toys are the point and not a shortcut.

## What is here so far

| Module | Contents |
| --- | --- |
| [`diffusion/targets.py`](diffusion/targets.py) | `GaussianMixture` (responsibility-weighted analytic score, exact moments and sampler, closed under noising), `TwoMoons` built as an actual normalized density in polar coordinates rather than a sampling recipe, and the finite-difference oracle every score is checked against |
| [`diffusion/langevin.py`](diffusion/langevin.py) | Unadjusted Langevin, and Song & Ermon's annealed Langevin with the $\tau_i \propto \sigma_i^2$ step-size schedule and the stability bound that schedule implies |
| [`diffusion/model.py`](diffusion/model.py) | The score network: an MLP conditioned on $\log\sigma$ through a written-out sinusoidal embedding, with the score formed as $-\epsilon_\theta/\sigma$ so the learned object is unit-scale at every noise level |
| [`diffusion/dsm.py`](diffusion/dsm.py) | Denoising score matching (Vincent 2011) with its equivalence proof, the $\sigma^2$ weighting that collapses it to $\\|\epsilon_\theta - \epsilon\\|^2$, per-level diagnostics against the exact score, and the loss's irreducible floor |
| [`diffusion/samplers.py`](diffusion/samplers.py) | Ancestral (DDPM) sampling down the ladder, derived for this repo's variance-exploding chain, with the closed-form variance it returns on a Gaussian target — including the deficit a coarse ladder is *supposed* to have |
| [`diffusion/metrics.py`](diffusion/metrics.py) | Two-sample metrics written out: RBF-kernel MMD with a permutation test, sliced Wasserstein, and the exact value each takes on Gaussians |

Every sampler takes the score as a callable, so the same code runs on the
*exact* score and on the network. That is what makes the sampler claims
testable at all: run both and the difference is estimation error, full stop.

### What a learned score is actually worth, measured

Because the mixture is closed under noising, a trained network can be scored
against the truth at every level rather than eyeballed
([`experiments/score_matching.py`](experiments/score_matching.py), 8-mode
ring, 87k parameters, 60k steps, ~2 min on CPU):

| $\sigma$ | 0.05 | 0.18 | 0.63 | 2.25 | 8.0 |
| --- | --- | --- | --- | --- | --- |
| relative MSE | 1.1e-1 | 2.8e-2 | 2.1e-2 | 4.2e-3 | 4.6e-3 |
| cosine | 0.896 | 0.970 | 0.976 | 0.955 | 0.996 |
| cosine, weighted by $\\|s\\|$ | 0.946 | 0.989 | 0.990 | 0.997 | 0.999 |

The fine end is the worst, which the $-\epsilon_\theta/\sigma$
parameterization predicts: any error in $\epsilon_\theta$ is amplified by
$1/\sigma$ precisely where the loss looks best.

### The training loss cannot tell you when to stop

The DSM loss does not go to zero. Its floor is
$\mathbb{E}\\|\epsilon - \mathbb{E}[\epsilon \mid x]\\|^2$, which runs from
$d = 2$ as $\sigma \to 0$ (the noisy point reveals $x_0$ and says nothing
about the noise) to $0$ at large $\sigma$ (the point is almost pure noise, so
the noise is nearly determined) — the opposite of the intuition that fine
levels are the easy ones. Averaged over the training measure it is
$1.203$ here, computed exactly.

So a well-trained network reports $1.212$ and a barely-trained one $1.228$,
and one minibatch's loss has a standard deviation of $0.073$ — eight times
the entire remaining excess. Subtract the floor and the same curve is
informative: excess risk falls from $0.025$ at step 2k to $0.0093$ at 60k
while the true relative score error falls 4.5×. This is the concrete payoff
of a target with a known score, and it is why the repo reports error against
truth rather than a loss curve.

### One measurement trap, recorded

The plain mean cosine dips to 0.955 mid-ladder. That is not the model: at
that level the ring has collapsed into a blob and 9% of the draws land in the
field's near-zero interior, where direction is undefined and a cosine
measures noise. Weighted by $\\|s\\|$ the same level reads 0.997. Both
statistics are reported, with the near-zero fraction beside them, so the dip
can be attributed rather than guessed at.

### The step size is bounded, and the bound is enforced

Noising bounds curvature, so it bounds the step size. The score of a
$\sigma$-noised density is a posterior mean, $s_\sigma(x) = (\mathbb{E}[x_0
\mid x] - x)/\sigma^2$, whose Jacobian is $(\mathrm{Cov}[x_0 \mid x]/\sigma^2 -
I)/\sigma^2 \succeq -I/\sigma^2$ for **any** base density. Linearizing the ULA
update, contraction needs $\tau|\lambda| < 4$, so $\tau_i < 4\sigma_i^2$ — one
condition, $\tau < 4\sigma_L^2$, covering every level of the ladder.

`annealed_langevin` refuses a $\tau$ above it, because this repo's own first
draft was above it and nearly got away with it. An unstable ladder does not
blow up visibly: the coarse levels drive the state to $10^{80}$ and the
strongly contractive fine levels squash it back, returning samples with the
right support and the *wrong mode weights* — invisible on a symmetric target.
A 25/75 mixture read 0.53 instead of 0.75, which is why an asymmetric mixture
is now a permanent test. The bound is necessary, not sufficient: at half the
bound the same test still reads 0.70.

## Sampling: three errors, taken apart

![samples](figures/samples.png)

A sample can be wrong for three reasons, and on a target with a known score
each one has an off switch: **prior mismatch** (start from an exact draw from
$p_{\sigma_{\max}}$ instead of from $N(0,\sigma_{\max}^2 I)$), **discretization**
(refine the ladder), and **score estimation** (pass the analytic noised score
through the identical sampler). [`experiments/sampling.py`](experiments/sampling.py)
runs all three, 16,000 samples per arm, measured against an independent exact
draw — which is itself a row, because neither metric reads zero at finite
sample size:

| sampler | score | init | NFE | weight TV | mode sd | sliced $W_2$ | MMD $p$, $h=5.71$ | MMD $p$, $h=0.4$ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact sampler | — | — | 0 | 0.005 | 0.401 | 0.043 | 0.26 | 0.48 |
| ancestral | exact | $p_{\sigma_{\max}}$ | 200 | 0.005 | 0.394 | 0.076 | 0.81 | 0.99 |
| ancestral | exact | $N(0,\sigma_{\max}^2)$ | 200 | 0.007 | 0.394 | 0.050 | 0.73 | 0.24 |
| ancestral | **learned** | $N(0,\sigma_{\max}^2)$ | 200 | 0.036 | 0.406 | 0.150 | 0.37 | **0.010** |
| annealed Langevin | exact | $N(0,\sigma_{\max}^2)$ | 200 | 0.012 | 0.419 | 0.047 | 0.72 | 0.46 |
| annealed Langevin | **learned** | $N(0,\sigma_{\max}^2)$ | 200 | 0.066 | 0.428 | 0.274 | 0.015 | **0.005** |
| annealed Langevin | exact | $N(0,\sigma_{\max}^2)$ | 2000 | 0.008 | 0.405 | 0.064 | 0.24 | 0.25 |
| annealed Langevin | **learned** | $N(0,\sigma_{\max}^2)$ | 2000 | 0.150 | 0.432 | **0.807** | 0.005 | 0.005 |

Sliced $W_2$ between two independent exact draws is $0.066 \pm 0.020$ (eight
pairs, range 0.036–0.092), so that band is the floor, not zero; $p = 0.005$ is
the resolution floor of a 199-permutation test, not a measured 0.005. Target
mode width is 0.4 and an exact draw reads 0.401.

With the exact score, ancestral sampling at 200 levels is indistinguishable
from an exact draw on every statistic here, and the prior mismatch at
$\sigma_{\max} = 8$ costs nothing measurable. Swapping in the learned score is
what moves the numbers.

The two samplers are wrong in opposite directions, which is worth noticing:
ancestral comes back a little *narrow* (0.394 against 0.401) because it
discards the posterior spread over $x_0$ at every level, while annealed
Langevin at 200 evaluations comes back *wide* (0.419) because ten steps per
level is not enough to contract the state down from the previous, broader
level. Giving it ten times the budget fixes that particular defect — 0.405 —
and, with the learned score, breaks something larger.

### More compute makes the Langevin sampler *worse*

The last three rows are the day's real finding, and it took the exact-score
control to make it a finding rather than a guess. Annealed Langevin with the
learned score goes from sliced $W_2$ 0.274 at 200 evaluations to **0.807 at
2000** — ten times the budget, three times the error, mode weights off by
0.150 in TV. The same sampler with the *exact* score does not degrade
(0.047 → 0.064, both at the floor), so this is not the ladder, the step size,
or the stability bound. It is the score.

The reason is what an equilibrium sampler is for. Langevin converges to the
stationary distribution of the score it is handed, so a biased score has a
biased stationary law and more steps converge to it *harder*. Ancestral
sampling never equilibrates — it visits each level exactly once — so the same
score error stays a perturbation instead of becoming a limit. That inverts the
usual reading of a compute knob: here the sampler that cannot be run longer is
the one protected from being run longer.

### Discretization error closes; estimation error does not

![sampler accuracy](figures/sampler_accuracy.png)

Refining the ancestral ladder drives the exact-score arm into the reference
floor by about 20 evaluations and it stays there. The learned-score arm falls
alongside it and then stops, at sliced $W_2 \approx 0.14$ and mode-weight TV
$\approx 0.03$, a plateau no amount of further refinement touches — because
what remains is not a step-size error.

The middle panel is a check nobody ordered. `ddpm_gaussian_variance` predicts
the variance the sampler returns on a *single* Gaussian; applied to one
component of the mixture it predicts the *within-mode* width, and once the
ladder is fine enough to resolve a mode it is right to under 1%:

| NFE | 20 | 50 | 100 | 200 | 1000 |
| --- | --- | --- | --- | --- | --- |
| predicted mode sd | 0.350 | 0.378 | 0.387 | 0.392 | 0.396 |
| measured (exact score) | 0.378 | 0.387 | 0.390 | 0.394 | 0.396 |

The gap at NFE 20 is real and expected: the coarse levels of that ladder do
not see one mode at a time, so a single-Gaussian recursion has no reason to
apply there, and it does not.

### The median heuristic picks the wrong scale

Look along the learned-ancestral row: MMD reports $p = 0.37$ at the
median-heuristic bandwidth and $p = 0.010$ at $h = 0.4$. Both are the same
test on the same samples.

The median pairwise distance on this target is 5.71, which is the *diameter of
the ring* — a kernel that wide compares gross shape and steps straight over
whether the individual modes have the right width. Set $h$ to the component
standard deviation, 0.4, and the test resolves the structure that actually
differs. The median heuristic is a sensible default and it is not an oracle;
on a target with structure at two scales an order of magnitude apart, it finds
the one you were not asking about. Sliced Wasserstein, which has no bandwidth,
flags the same row without being told where to look.

## Relation to mcmc-from-scratch

The ULA update

$$x \leftarrow x + \tfrac{\tau}{2}\, s(x) + \sqrt{\tau}\, z$$

is exactly the MALA proposal in
[mcmc-from-scratch](https://github.com/porth-bot/mcmc-from-scratch) with the
Metropolis accept step deleted, and the deletion has a price this repo states
rather than hides: MALA is exact at any step size, ULA is biased by $O(\tau)$.
On a standard Gaussian that bias is closed-form — ULA's stationary variance is
$1/(1-\tau/4)$, 6.7% too wide at $\tau = 0.25$ — and
[`tests/test_langevin.py`](tests/test_langevin.py) measures it. The reason to
accept the bias is not speed: a learned score gives no $\log p$ to put in a
Metropolis ratio, so the accept step is unavailable in principle.

## Reproduce

```bash
python -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/pytest -q
```

## Provenance

A from-scratch study resource for learning how score-based generative models
work. Every derivation is written out (in the module docstrings now, in
`theory/derivations.md` when it lands) and every non-obvious claim is tested
against a closed-form or independently computed ground truth rather than
taken on faith.

## License

MIT — see [LICENSE](LICENSE).
