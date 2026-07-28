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

The samplers are driven by the *exact* score, which is what makes the sampler
claims testable at all: any failure is the sampler's, not an estimation error.

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
