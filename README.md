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

Both are driven by the *exact* score today, which is what makes the sampler
claims testable at all: any failure is the sampler's, not an estimation error.

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
