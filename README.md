# diffusion-from-scratch

![ci](https://github.com/porth-bot/diffusion-from-scratch/actions/workflows/ci.yml/badge.svg)

Score-based generative modeling built from the score outward, in PyTorch, on
2D targets whose scores are known in **closed form**.

![score fields](figures/score_fields.png)

*The object a diffusion model estimates, and the estimate, side by side. Rows
are noise levels — $\sigma = 4$, $1$, $0.2$ — of an 8-mode ring mixture. Left
is the **exact** score of the noised density $p_\sigma$, which is available
here because a Gaussian mixture stays a Gaussian mixture when noised; middle
is what an 87k-parameter network learned by denoising score matching; right is
the error, normalized by that level's mean $\\|s\\|$. The learned field is not
a picture that looks about right — it is wrong by a measurable amount, and the
amount is worst at the **fine** level, exactly where the training loss looks
best. Everything below is that observation, taken seriously.*

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

## Core modules

| Module | Contents |
| --- | --- |
| [`diffusion/targets.py`](diffusion/targets.py) | `GaussianMixture` (responsibility-weighted analytic score, exact moments and sampler, closed under noising), `TwoMoons` built as an actual normalized density in polar coordinates rather than a sampling recipe, and the finite-difference oracle every score is checked against |
| [`diffusion/langevin.py`](diffusion/langevin.py) | Unadjusted Langevin, and Song & Ermon's annealed Langevin with the $\tau_i \propto \sigma_i^2$ step-size schedule and the stability bound that schedule implies; plus the three noise ladders (geometric, linear, and the cosine schedule rederived for a variance-exploding process) |
| [`diffusion/model.py`](diffusion/model.py) | The score network: an MLP conditioned on $\log\sigma$ through a written-out sinusoidal embedding, with the score formed as $-\epsilon_\theta/\sigma$ so the learned object is unit-scale at every noise level |
| [`diffusion/dsm.py`](diffusion/dsm.py) | Denoising score matching (Vincent 2011) with its equivalence proof, the $\sigma^2$ weighting that collapses it to $\\|\epsilon_\theta - \epsilon\\|^2$, per-level diagnostics against the exact score, and the loss's irreducible floor |
| [`diffusion/samplers.py`](diffusion/samplers.py) | Ancestral (DDPM) sampling down the ladder, derived for this repo's variance-exploding chain, with the closed-form variance it returns on a Gaussian target — including the deficit a coarse ladder is *supposed* to have |
| [`diffusion/metrics.py`](diffusion/metrics.py) | Two-sample metrics written out: RBF-kernel MMD with a permutation test, sliced Wasserstein, and the exact value each takes on Gaussians |
| [`diffusion/ode.py`](diffusion/ode.py) | The probability-flow ODE with $\sigma$ as the clock, a written-out RK4 (measured fourth-order), and likelihoods by instantaneous change of variables with a Hutchinson trace estimator — plus the exact flow on a Gaussian, which is what the solver is tested against |

[`theory/derivations.md`](theory/derivations.md) carries the maths behind all
of it, section-numbered to match the modules: the mixture's score and its
closure under noising, Tweedie's formula, Hyvärinen's identity and Vincent's
denoising form of it, Anderson's reverse-time SDE and the ancestral sampler as
its discretization, the probability-flow ODE and its likelihoods, and the
Langevin bridge back to
[mcmc-from-scratch](https://github.com/porth-bot/mcmc-from-scratch).

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

![score error vs sigma](figures/score_error_vs_sigma.png)

*The table above, at every level of the ladder rather than five of them. Left:
relative MSE against the exact noised score, climbing an order of magnitude as
$\sigma$ falls — the error is a property of where on the ladder you look, not a
single number. Right: the two cosines. They agree at the coarse end and split
below $\sigma \approx 1$, and the plain mean's dip near $\sigma = 2$ is the
measurement trap two sections down, not a defect of the model.*

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

![training curves](figures/score_matching_training.png)

*The same run, read three ways. Left: the DSM loss, flat against its exact
optimum (dashed) from ~2k steps on, with one minibatch's noise swamping the
remaining excess. Middle: subtract the optimum and the excess risk is still
falling at 60k steps — under the dotted line marking the sd of a single
minibatch loss, i.e. entirely inside what the left panel could never have
resolved. Right: the quantity that actually matters, error against the exact
score, falling over the same stretch the loss called finished.*

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

## Take the noise out: the probability-flow ODE

![ODE vs SDE](figures/ode_vs_sde.png)

The forward process has no drift, so its density obeys the heat equation
$\partial_t p = \tfrac12 g^2 \Delta p$. Write $\Delta p = \nabla\cdot(p \nabla \log p)$
and that *is* a continuity equation — which says $p$ is transported by a
velocity field. Taking $\sigma$ as the clock instead of $t$, the schedule
cancels out of it entirely:

$$\frac{dx}{d\sigma} = -\sigma\, s(x, \sigma).$$

So sampling becomes an initial value problem, and two things fall out of the
same equation. One Euler step of it is **half** the ancestral sampler's mean
update — the missing half plus the injected noise are a Langevin step, which
leaves $p_\sigma$ invariant, so the stochastic sampler is this ODE plus a term
that does nothing in exact arithmetic. And on a Gaussian target the ODE
separates: every point moves along its own ray by
$\sqrt{(v+\sigma_1^2)/(v+\sigma_0^2)}$, an exact solution the solver is tested
against *pointwise*, which is how `tests/test_ode.py` measures RK4's order as
4.0 and Euler's as 1.0 rather than trusting the label.

### Same budget, three dynamics

[`experiments/ode.py`](experiments/ode.py) puts the ancestral sampler and the
PF-ODE (Euler and RK4) on one axis: score evaluations. RK4 spends four per
step, so at a budget of 41 it gets 11 nodes against the ancestral sampler's 41
levels. Sliced $W_2$ against an exact draw, 16,000 samples per cell:

| sliced $W_2$ | NFE 9 | 17 | 41 | 81 | 201 | 401 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ancestral, exact | 0.131 | 0.090 | 0.062 | 0.055 | 0.051 | 0.080 |
| PF-ODE Euler, exact | 0.180 | 0.113 | 0.082 | 0.074 | 0.070 | 0.069 |
| PF-ODE RK4, exact | 0.687 | **0.059** | 0.067 | 0.068 | 0.068 | 0.068 |
| ancestral, **learned** | 0.236 | 0.172 | 0.147 | 0.146 | 0.151 | 0.125 |
| PF-ODE Euler, **learned** | 0.205 | 0.150 | 0.123 | 0.116 | 0.112 | 0.111 |
| PF-ODE RK4, **learned** | 0.666 | 0.113 | 0.111 | 0.109 | 0.110 | 0.110 |

Read the exact-score rows against the floor, not against zero: two independent
exact draws differ by $0.066 \pm 0.020$ (range 0.036–0.092 over eight pairs), so
from 41 evaluations on **all three are inside that band and none of them is
distinguishable from an exact draw, or from each other**. What can still be said
is who gets there first: at 17 evaluations RK4 reads 0.059, at the floor's
centre, while the ancestral sampler is at its top edge (0.090) and Euler is
outside it (0.113). And at 9 evaluations, where RK4 has 2 steps to work with, it
is by far the worst of the three (0.687) — fourth-order accuracy buys nothing
until there are steps to be accurate over.

The learned-score rows are outside the band, and they separate: the ODE sits
below the ancestral sampler at every matched budget from 17 up, with steadier
mode weights (TV 0.019 against 0.031). Each individual gap is about twice the
floor's own spread; what makes the ordering credible is that five budgets and
two statistics all point the same way. The honest summary is that *which
dynamics is better here is a property of the score, not of the samplers* — with
the truth in hand the noise costs nothing, and with a biased score the extra
randomness does not help.

Middle panel, the coarse-ladder biases, which point in opposite directions:
ancestral approaches the target mode width **from below** (0.378 at 17
evaluations, 0.396 at 401) because it replaces the posterior over $x_0$ by its
mean and loses spread; Euler approaches it **from above** (0.433 → 0.395)
because an under-resolved flow has not finished contracting.

### A deterministic sampler inherits its prior, permanently

One number in that sweep does not move at all: the ODE's mode-weight TV with
the exact score is **0.0121 at every budget from 9 to 401**, to four decimals.
That is not a coincidence, it is what determinism means — the map from prior
draw to sample is a fixed diffeomorphism, so which mode each prior point lands
in is decided by the prior sample, and refining the solver cannot re-roll it.
The stochastic sampler's TV wanders between 0.004 and 0.012 across the same
budgets because it re-randomizes at every level.

That also explains the floor under the ODE's mode width (0.3932 against a
target of 0.4, flat from 81 evaluations on). Feed the identical run an exact
draw from $p_{\sigma_{\max}}$ instead of $N(0, \sigma_{\max}^2 I)$:

| PF-ODE RK4, 201 NFE | weight TV | mode sd | sliced $W_2$ |
| --- | ---: | ---: | ---: |
| exact score, $N(0,\sigma_{\max}^2)$ start | 0.0121 | 0.3932 | 0.068 |
| exact score, exact $p_{\sigma_{\max}}$ start | 0.0055 | 0.3980 | 0.065 |

Prior mismatch, and nothing else, was most of what remained.

### The noise schedule cancels — for one of the two samplers

![schedules](figures/schedules.png)

If the schedule really does drop out of $dx/d\sigma = -\sigma s$, then two
ladders with the same endpoints describe the same trajectory and differ only in
where they put nodes. That is a statement about the continuous limit, so
[`experiments/schedules.py`](experiments/schedules.py) asks what it is worth at
a finite budget. Three ladders, each uniform spacing in a different warp of
$\sigma$:

- **geometric** — uniform in $\log\sigma$, the repo's default.
- **linear** — uniform in $\sigma$, the naive choice.
- **cosine** — uniform in $\arctan\sigma$. That *is* Nichol & Dhariwal's
  schedule: their $\bar\alpha = \cos^2(\pi u/2)$ is variance-preserving, and a
  VP state is $\sqrt{\bar\alpha}\,(x_0 + \sigma\varepsilon)$ with
  $\sigma^2 = (1-\bar\alpha)/\bar\alpha$, so $\sigma(u) = \tan(\pi u / 2)$.

Sliced $W_2$ against an exact draw, exact score, 16,000 samples per cell,
averaged over three prior draws (seed-to-seed sd averages 0.008):

| exact score | NFE 9 | 17 | 41 | 101 | 201 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ancestral, geometric | 0.109 | 0.079 | 0.067 | 0.052 | 0.054 |
| ancestral, cosine | 0.096 | 0.079 | 0.064 | 0.060 | 0.051 |
| ancestral, linear | **0.233** | 0.125 | 0.086 | 0.066 | 0.063 |
| PF-ODE RK4, geometric | **0.689** | 0.064 | 0.0713 | 0.0715 | 0.0715 |
| PF-ODE RK4, cosine | 0.302 | 0.064 | 0.0708 | 0.0715 | 0.0715 |
| PF-ODE RK4, linear | 0.237 | 0.121 | 0.0711 | 0.0715 | 0.0715 |

**The ODE forgets the schedule, on schedule.** Its three arms disagree by 0.451
at 9 evaluations — 57× the seed-to-seed spread — then by 0.058 at 17, 0.0005 at
41, and 0.0001 from 101 on, which is below what the statistic can resolve. Three
different ladders, the same answer to four decimals: the cancellation is not a
rewriting trick, it is something the sampler actually does.

**The ancestral sampler does not.** Its spread never closes: 0.021 at 41, 0.015
at 101, 0.016 at 201, always with the linear ladder worst. That is the right
outcome rather than a discrepancy — the derivation covers the flow, not the
chain, and the chain's per-level plug-in error (§ ancestral, above) depends on
the *gap* between adjacent levels. At 201 levels the linear ladder's finest
ratio is still 1.79 where the geometric one is 1.03.

**And at 9 evaluations the two samplers rank the schedules in opposite orders.**
Linear is the worst ancestral ladder (0.233) and the best ODE one (0.237 against
geometric's 0.689), because RK4 at that budget has 3 nodes and the geometric
ladder spends them at the fine end where this target's flow has already stopped
moving, while the ancestral sampler has 9 levels and is hurt by linear's final
ratio of 21. A study that ran one sampler would have come away with a
schedule recommendation that is wrong for the other one.

The practical conclusion is the least dramatic reading. At any budget past 41,
swapping the schedule moves the learned-score ODE by less than 0.0015 while the
score being learned rather than exact costs 0.037 — more than twenty-five
times as much.
On this target the schedule is a knob worth setting sensibly once and then
leaving alone.

## Likelihoods, checked against a closed form

![ODE likelihood](figures/ode_likelihood.png)

The flow is invertible, so it carries densities as well as points. Along a
trajectory $d/ds \log p_s = -\nabla\cdot v$ (the continuity equation again), and
integrating *up* the ladder gives

$$\log p_{\sigma_{\min}}(x) = \log p_{\sigma_{\max}}(z) + \int_{\sigma_{\min}}^{\sigma_{\max}} \nabla\cdot v \, ds,$$

with the divergence estimated by $\mathbb{E}[\varepsilon^\top J \varepsilon]$ at
one vector-Jacobian product per probe (FFJORD). Probes are drawn **once per
solve**: resampling them inside the integrator makes the augmented field
discontinuous in $s$ and silently costs the solver its order. Rademacher probes
are the default because on a diagonal Jacobian they return the trace exactly,
where Gaussian probes still carry variance — and at points where one mixture
component dominates, the noised score's Jacobian *is* exactly isotropic, so that
is not a hypothetical.

Because the noised mixture's density is closed-form, every number here has a
truth to be wrong against. Two errors, separated:

| RK4 nodes | 3 | 5 | 9 | 17 | 41 | 81 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean $\lvert$error$\rvert$, exact divergence (nats) | 1.4e+0 | 1.2e-1 | 1.4e-2 | 5.5e-4 | 1.1e-5 | 1.3e-6 |

| Hutchinson probes $k$ | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean $\lvert$error$\rvert$ at 41 nodes (nats) | 0.78 | 0.38 | 0.30 | 0.22 | 0.150 | 0.111 |

Integration error falls at roughly $h^4$ until it reaches the float32 noise
floor around $10^{-6}$ (the 81-node point is that floor, not the method's), and
estimator variance falls as $1/\sqrt{k}$ — but at 41 nodes it is still *four
orders of magnitude above* the integration error (0.111 nats at 32 probes
against 1.1e-5). At $d = 2$ the exact trace costs two
backward passes and the estimator costs accuracy, so the honest conclusion is
that Hutchinson is machinery for dimensions this repo does not have, included
because that is where the method is actually used and because its cost is worth
measuring rather than assuming.

### What the model's likelihood is worth, in nats

Each of these three flows is a normalized density, so averaging
$\log p - \log q$ over exact draws estimates $\mathrm{KL}(p_{\sigma_{\min}} \| q)$,
which cannot be negative. 2,000 draws, 41 nodes, exact divergence:

| flow | $\mathbb{E}[\log q]$ | estimated KL (nats) |
| --- | ---: | ---: |
| exact score, exact $p_{\sigma_{\max}}$ prior | −3.1211 | $-9\times10^{-6} \pm 0.0000$ |
| exact score, $N(0,\sigma_{\max}^2)$ prior | −3.1300 | $+0.0089 \pm 0.0029$ |
| **learned** score, $N(0,\sigma_{\max}^2)$ prior | −3.1885 | $+0.0674 \pm 0.0082$ |

The first row is the method's own zero: that flow reproduces the target
exactly, so whatever it reports is the error bar on the other two. Against it,
starting the sampler from a Gaussian instead of $p_{\sigma_{\max}}$ costs
0.0089 nats and the score being learned costs a further 0.0585 — the estimation
error is about seven times the prior gap. Both were visible in the sample
metrics only as "somewhere inside the floor band"; the likelihood puts a number
on each, which is the whole reason to compute one.

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

Every figure, from a clean clone, without training or sampling anything:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e .
./reproduce.sh                  # 207 tests, then all 8 figures: under a minute
```

Running this repo end to end is about an hour of CPU — the sampler sweep alone
draws 16,000 samples per arm across eight arms and six ladder refinements — so
the artifacts ship with it: CSV sweeps and JSON summaries in `logs/`, the
trained score network in `logs/score_net.pt`, and the sample tensors the cloud
figures are pictures of. The split matters: a figure of a *number* rebuilds
from a sweep log, but a figure of a *cloud* or of a *field* is a picture of
actual draws and of the network itself, and only the tensors and the weights
reproduce those. Each experiment takes `--figures` to replay its own.

`tests/test_reproduction.py` is what keeps that claim true rather than
aspirational: every artifact the replay reads must be **tracked by git**
(sitting in the working tree is not the same thing), and the committed weights
must still load into the `ScoreNet` the code builds today — a break that
retraining would silently paper over, since training rewrites the file it
would have failed on.

`requirements.txt` pins the exact versions the committed artifacts were
produced with (Python 3.12.13, torch 2.13.0); `pyproject.toml` keeps lower
bounds, which is what CI installs on 3.10 and 3.12. The replay is pure
post-processing and is therefore exact — rerunning it leaves the working tree
clean. Recomputation is seeded and replays on the same torch build, but torch
guarantees no bitwise determinism across versions or hardware, which is why
the artifacts are committed rather than regenerated on demand.

To actually recompute:

```bash
python experiments/score_matching.py   # ~2 min   trains the score network
python experiments/sampling.py         # ~20 min  8 sampler arms + ladder sweep
python experiments/ode.py              # ~25 min  NFE sweep + likelihoods
python experiments/schedules.py        # ~15 min  3 ladders x 5 budgets x 3 seeds
```

## Limitations / next

- **2D toys, and that is the design.** Every number here exists because the
  target's score is closed-form at every noise level. Nothing in this repo
  transfers as a *result* to images or text: no claim is made about sample
  quality on real data, and none of these networks (87k parameters, four
  layers) would be a reasonable model of anything larger. What transfers is
  the machinery and the way of checking it.
- **One trained network.** The score-estimation numbers — the error tables,
  the learned-score rows in every sampler comparison, the 0.0585-nat
  likelihood gap — come from a single training run (seed 0, 60k steps). The
  sampler and schedule comparisons vary the sampler and the ladder against
  that fixed network, which is the right control for *those* questions, but
  none of the learned-score numbers carries a training-seed error bar. The
  exact-score arms have no such caveat.
- **`TwoMoons` is a tested density, not an experimental target.** It is a
  genuinely normalized density in polar coordinates with an analytic score,
  and it is checked against the finite-difference oracle — but it is not
  closed under noising, so it cannot supply ground truth at intermediate
  levels, which is what every experiment here needs. The experiments all run
  on the ring mixture for that reason.
- **No conditioning and no guidance.** Classifier-free guidance, and
  conditional generation on the mixture component, are the obvious next thing
  a target with labeled components makes measurable.
- **Fast samplers stop at RK4.** The ODE side compares Euler and RK4 at
  matched score evaluations; Heun/EDM-style solvers, DDIM, and distillation
  are not here. On this target the flow is resolved by ~40 evaluations
  anyway, so the study that would justify them is one on a harder target.
- **Hutchinson is measured, not needed.** At $d = 2$ the exact divergence
  costs two backward passes and beats the estimator by four orders of
  magnitude at every probe count tried. It is included because that is what
  the method actually uses in high dimension, and because its cost is worth
  measuring rather than assuming.
- **Float32 throughout.** The likelihood integration error reaches its noise
  floor around $10^{-6}$ nats at 81 RK4 nodes; the 81-node column measures
  the arithmetic, not the method.
- **Variance-exploding only.** The forward process here has no drift.
  Variance-preserving is a change of variables away (the cosine schedule is
  derived through exactly that change in
  [`diffusion/langevin.py`](diffusion/langevin.py)), but a VP-vs-VE
  comparison is not run.

## References

Hyvärinen (2005) (score matching); Vincent (2011) (the denoising form, and
the equivalence proof §3.3 rests on); Robbins (1956) / Efron (2011)
(Tweedie's formula, and the curvature bound behind the step-size ceiling);
Song & Ermon (2019, 2020) (annealed Langevin, the $\sigma^2$ weighting,
ladder spacing); Ho, Jain & Abbeel (2020) (the ancestral sampler,
$\epsilon$-prediction); Song et al. (2021) (the SDE framing, the
probability-flow ODE, likelihoods); Anderson (1982) (the reverse-time SDE);
Nichol & Dhariwal (2021) (the cosine schedule, rederived here for a
variance-exploding process); Karras et al. (2022) ($\sigma$ as the clock);
Chen et al. (2018) and Grathwohl et al. (2019) (instantaneous change of
variables, FFJORD); Hutchinson (1989) (the trace estimator); Roberts &
Tweedie (1996) (ULA's bias, MALA's correction). Full list with roles in
[`theory/derivations.md`](theory/derivations.md).

## Part of a from-scratch series

Same bar in each: the core written out by hand, every non-obvious claim
checked against a closed form or an independent oracle, limitations stated
rather than buried.

| Repo | Built from scratch |
| --- | --- |
| **diffusion-from-scratch** *(this repo)* | Score matching, reverse-time samplers, and the probability-flow ODE — against exact scores at every noise level |
| [mcmc-from-scratch](https://github.com/porth-bot/mcmc-from-scratch) | Metropolis-Hastings, Gibbs, HMC, MALA, NUTS, parallel tempering — validated against exact posteriors |
| [gp-from-scratch](https://github.com/porth-bot/gp-from-scratch) | GP regression, kernels with hand-derived gradients, ML-II, and the NTK/NNGP wide-network correspondence |
| [pinn-from-scratch](https://github.com/porth-bot/pinn-from-scratch) | Physics-informed networks: exact autograd PDE residuals against closed-form solutions |
| [grokking-transformer](https://github.com/porth-bot/grokking-transformer) | A transformer that groks modular arithmetic, and the Fourier circuit it learns |

The load-bearing link is to mcmc-from-scratch, and it is a shared sampler
rather than a shared theme: the Langevin update here *is* that repo's MALA
proposal with the accept step deleted, the deletion is forced (a learned score
gives no density to put in a Metropolis ratio), and both repos measure the
same closed-form ULA bias. The annealing that makes the biased sampler work is
the same idea as its parallel tempering, applied to noise levels instead of
temperatures — and the failure it fixes, a chain trapped in one mode, is
measured on both sides.

## Provenance

A from-scratch study resource for learning how score-based generative models
work. Every derivation is written out — in the module docstrings and, at
length, in [`theory/derivations.md`](theory/derivations.md) — and every
non-obvious claim is tested against a closed-form or independently computed
ground truth rather than taken on faith.

*Suggested GitHub topics:* `diffusion-models` `score-matching`
`generative-models` `sde` `langevin-dynamics` `pytorch` `from-scratch`

## License

MIT — see [LICENSE](LICENSE).
