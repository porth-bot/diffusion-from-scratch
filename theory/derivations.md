# Derivations

Hand-derived math behind the code, section-numbered so the module docstrings
can point here. Sections map onto modules: Sec. 1-2 → `diffusion/targets.py`,
Sec. 3 → `diffusion/dsm.py` and `diffusion/model.py`, Sec. 4 →
`diffusion/samplers.py`, Sec. 5 → `diffusion/ode.py`, Sec. 6 →
`diffusion/langevin.py`.

Notation follows the source. $p_0$ is the data density, $p_\sigma$ the density
of $x_0 + \sigma\varepsilon$ with $\varepsilon \sim N(0, I)$, $s_\sigma(x) =
\nabla_x \log p_\sigma(x)$ its score, and $s_\theta(x, \sigma)$ the network's
estimate of it. Points are $d$-dimensional ($d = 2$ everywhere in this repo),
$\tau$ is a Langevin step size, and a *ladder* is a decreasing sequence
$\sigma_L > \dots > \sigma_1 > 0$ — coarse first, because that is the order the
samplers walk it.

Everything here that can be checked is checked numerically somewhere in
`tests/`: every analytic score against finite differences of its own
`log_prob` (`tests/test_targets.py`), the ESM/DSM equivalence of Sec. 3.3 as a
measured constant (`tests/test_dsm.py`), the reverse-process variance of
Sec. 4.4 against the recursion it predicts (`tests/test_samplers.py`), and the
flow and likelihood of Sec. 5 against the closed-form Gaussian flow and the
exact noised-mixture density (`tests/test_ode.py`).

Contents:

1. The score, and what it is for
2. The Gaussian mixture: its score, its noised marginals, and Tweedie's formula
3. Learning the score: Hyvärinen's identity and Vincent's denoising form
4. The forward SDE, Anderson's reverse SDE, and DDPM as its discretization
5. The probability-flow ODE and likelihoods
6. The Langevin bridge back to mcmc-from-scratch

---

## 1. The score, and what it is for

Generative modeling wants a density. Densities are hard because they must
normalize: any parametric $\tilde p_\theta$ defines
$p_\theta = \tilde p_\theta / Z(\theta)$ with $Z(\theta) = \int \tilde
p_\theta$, and $Z$ is an intractable integral that depends on the parameters,
so maximum likelihood cannot even be written down without it.

The **score** sidesteps this entirely:

$$s(x) := \nabla_x \log p(x) = \nabla_x \log \tilde p(x) - \underbrace{\nabla_x \log Z}_{=\,0}.$$

The normalizer is a constant in $x$, so it has zero gradient, and the score of
an unnormalized model is the score of the normalized one. Three facts make
that observation into a method rather than a curiosity, and they are the three
halves of this repo (Sec. 6, Sec. 4, Sec. 5 respectively):

1. **You can sample from a density given only its score.** The overdamped
   Langevin SDE $dX = s(X)\,dt + \sqrt2\,dW$ has $p$ as its stationary
   distribution and never mentions $p$ itself.
2. **You can learn a score from samples alone**, without ever evaluating the
   density it belongs to (Sec. 3).
3. **You can transport one density onto another along a score-defined flow**,
   which turns sampling into an initial value problem and hands back
   likelihoods on the way (Sec. 5).

So a diffusion model is, in the end, an estimate of one vector field. Which
raises the question the whole repo is organized around: *how do you know the
estimate is any good?* On images you cannot know — there is no ground-truth
score to compare against, so the field is judged by looking at its samples.
On a 2D Gaussian mixture you can, at every noise level, in closed form. That
is what Sec. 2 sets up and what every measured claim in the README leans on.

### 1.1 Why the *noised* score, and not just the score

Nothing above needs noise. The reason every method here is indexed by a noise
level $\sigma$ is that the score of the *data* density is badly behaved in
exactly the places a sampler has to work:

- **Off the manifold it is undefined or enormous.** Data concentrates near a
  low-dimensional set; a sampler initialized far from it needs a score there,
  and $p_0$ supplies either nothing or a number dominated by its tails.
- **Between modes it is useless.** In a region where $p_0 \approx 0$ the score
  points somewhere, but no finite-sample estimator has any information there,
  and it is precisely the region a chain must cross to move between modes
  (Sec. 6.2).

Convolving with $N(0, \sigma^2 I)$ fixes both, at the cost of answering a
different question. $p_\sigma$ has full support, its score is bounded in a way
that is uniform over *all* base densities (Sec. 2.3), and as $\sigma$ grows it
smears the modes into each other until the target is effectively unimodal. The
whole design of a diffusion model is the bookkeeping needed to exploit that
without forgetting that $p_\sigma \neq p_0$: learn the score at *every* level
(Sec. 3.4), and walk the levels from coarse to fine so that the easy problem's
answer initializes the hard one (Sec. 4, Sec. 6.2).

---

## 2. The Gaussian mixture: its score, its noised marginals, and Tweedie's formula

`diffusion/targets.py` implements two families. The mixture is the one that
carries the repo's claims, for reasons this section derives.

### 2.1 The score of a mixture is a responsibility-weighted average

Let $p(x) = \sum_k w_k\, p_k(x)$ with $w_k > 0$, $\sum_k w_k = 1$. Define the
**responsibility**

$$r_k(x) := \frac{w_k\, p_k(x)}{p(x)}, \qquad \sum_k r_k(x) = 1 .$$

Then

$$\nabla \log p = \frac{\nabla p}{p} = \frac{\sum_k w_k \nabla p_k}{p} = \sum_k \frac{w_k p_k}{p}\cdot\frac{\nabla p_k}{p_k} = \sum_k r_k(x)\, \nabla \log p_k(x).$$

So the mixture's score is a *convex* combination of the component scores. Two
consequences the implementation depends on:

- $r_k$ is a **softmax** of $\log w_k + \log p_k(x)$, so the shared normalizer
  of $p$ cancels before anything is exponentiated. A hand-rolled
  $\nabla \log \sum_k \exp(\cdot)$ underflows where this does not.
- Because the weights are convex and $r_k \to 1$ for the dominant component in
  a far tail, the mixture score **reduces exactly to the nearest component's**
  out there. That is the behaviour a sampler placed in a distant mode needs,
  and `tests/test_targets.py` asserts it rather than assuming it.

For Gaussian components, $\log N(x; \mu, \Sigma) = -\tfrac12[d\log 2\pi +
\log\det\Sigma + (x-\mu)^\top \Sigma^{-1}(x-\mu)]$, and differentiating the
quadratic form gives $\nabla \log N = -\Sigma^{-1}(x - \mu)$ — a linear pull
toward the mean along the precision-weighted direction. Hence

$$s(x) = -\sum_k r_k(x)\, \Sigma_k^{-1}(x - \mu_k).$$

Everything is computed from one Cholesky factor per component: $\log\det\Sigma
= 2\sum_i \log L_{ii}$ and the quadratic form is $\|L^{-1}(x-\mu)\|^2$ by a
triangular solve, which is cheaper and better conditioned than forming
$\Sigma^{-1}$.

### 2.2 Closure under noising — the property everything rests on

Let $p_\sigma := p_0 * N(0, \sigma^2 I)$, the density of $x_0 + \sigma
\varepsilon$. For a single Gaussian, the sum of independent Gaussians is
Gaussian and covariances add:

$$N(\mu, \Sigma) * N(0, \sigma^2 I) = N(\mu, \Sigma + \sigma^2 I).$$

Convolution is linear, so it distributes over the sum:

$$p_\sigma = \Big(\sum_k w_k N(\mu_k, \Sigma_k)\Big) * N(0,\sigma^2 I) = \sum_k w_k\, N(\mu_k,\ \Sigma_k + \sigma^2 I).$$

**A noised Gaussian mixture is a Gaussian mixture with the same means and
weights and inflated covariances.** So $\log p_\sigma$, $s_\sigma$, the exact
moments and the exact sampler are all available at *every* noise level, from
the same code, by one substitution $\Sigma_k \mapsto \Sigma_k + \sigma^2 I$
(`GaussianMixture.noised`). By Sec. 2.1 the noised score is

$$s_\sigma(x) = -\sum_k r_k^\sigma(x)\,(\Sigma_k + \sigma^2 I)^{-1}(x - \mu_k),$$

with responsibilities recomputed from the inflated components — note that the
weights *shift* with $\sigma$ even though the mixing proportions do not, which
is exactly the smearing-together of the modes that annealing exploits.

This is the single fact that makes the repo's measurements possible. A learned
$s_\theta(\cdot,\sigma)$ has an analytic target at every level, so score error
is *measured* rather than eyeballed; a sampler can be run with the exact score
as a control, so estimation error is *separated* from discretization error; and
the flow of Sec. 5 has a closed-form density to be wrong against. None of the
three is available on an image dataset, which is why the toys are the point
and not a shortcut.

The honest counterpart is `TwoMoons`, built as a normalized density in polar
coordinates rather than as a sampling recipe. Its score is analytic at
$\sigma = 0$ but it is **not** closed under noising — the $\sigma$-noised
crescent is a convolution with no elementary form. The division of labour is
deliberate: the mixture is where methods are validated against truth, the
moons are where they are shown to work on something whose noised marginals
nobody can write down.

### 2.3 Tweedie's formula, and the curvature bound it implies

Write $p_\sigma(x) = \int p_0(x_0)\, N(x; x_0, \sigma^2 I)\, dx_0$ and
differentiate under the integral sign (legitimate: the Gaussian kernel is
smooth and dominates):

$$\nabla_x p_\sigma(x) = \int p_0(x_0)\, N(x; x_0, \sigma^2 I) \cdot \Big(\!-\frac{x - x_0}{\sigma^2}\Big) dx_0 .$$

Dividing by $p_\sigma(x)$ turns the integrand's weight into the posterior
$p(x_0 \mid x) = p_0(x_0) N(x; x_0,\sigma^2 I)/p_\sigma(x)$, so

$$\boxed{\;s_\sigma(x) = \frac{\mathbb{E}[x_0 \mid x] - x}{\sigma^2}\;}\qquad\text{equivalently}\qquad \mathbb{E}[x_0\mid x] = x + \sigma^2 s_\sigma(x).$$

**The score is a denoiser.** It points from the noisy point toward the
posterior mean of the clean point that produced it, scaled by $1/\sigma^2$.
This identity is used four separate times below (Sec. 3.1, Sec. 3.5, Sec. 4.4,
Sec. 6.3), which is a fair measure of how central it is.

Differentiate once more. Writing $m(x) := \mathbb{E}[x_0\mid x]$,

$$\nabla_x m(x) = \frac{\nabla_x \int x_0\, p_0 N \,dx_0}{p_\sigma} - \frac{\big(\int x_0 p_0 N\, dx_0\big) \nabla_x p_\sigma^{\top}}{p_\sigma^2} = \frac{\mathbb{E}[x_0 x_0^\top\mid x] - m m^\top}{\sigma^2} = \frac{\mathrm{Cov}[x_0 \mid x]}{\sigma^2},$$

using $\nabla_x N(x;x_0,\sigma^2 I) = -N\,(x-x_0)/\sigma^2$ in both terms.
Therefore

$$\nabla s_\sigma(x) = \frac{\nabla m(x) - I}{\sigma^2} = \frac{1}{\sigma^2}\Big(\frac{\mathrm{Cov}[x_0\mid x]}{\sigma^2} - I\Big) \;\succeq\; -\frac{I}{\sigma^2},$$

since a covariance matrix is positive semidefinite. **For any base density
whatsoever**, the Jacobian of the $\sigma$-noised score is bounded below by
$-I/\sigma^2$. Noising bounds curvature, uniformly, with no assumption on the
data — and Sec. 6.3 turns that into a step-size bound the code enforces.

---

## 3. Learning the score: Hyvärinen's identity and Vincent's denoising form

### 3.1 The objective that cannot be written down

The thing we want is small:

$$J_{\mathrm{ESM}}(\theta) = \tfrac12\,\mathbb{E}_{p_\sigma(x)}\big\|s_\theta(x) - \nabla \log p_\sigma(x)\big\|^2 . \tag{ESM}$$

It names $\nabla \log p_\sigma$, which is exactly the unknown. (This repo can
*evaluate* it for the mixture — Sec. 2.2, and `dsm.explicit_score_matching_loss`
does — but a method that needs it is not a method. It exists here only to check
the claim of Sec. 3.3.)

### 3.2 Hyvärinen's identity: integrate the cross term by parts

Expand the square. The $\|\nabla\log p_\sigma\|^2$ term has no $\theta$ in it,
so it is a constant $C$; only the cross term is a problem:

$$J_{\mathrm{ESM}} = \mathbb{E}_{p_\sigma}\Big[\tfrac12\|s_\theta\|^2\Big] - \mathbb{E}_{p_\sigma}\big[\langle s_\theta, \nabla \log p_\sigma\rangle\big] + C .$$

For the cross term, the density cancels against the $1/p_\sigma$ hidden in the
score, and then integration by parts moves the derivative onto $s_\theta$:

$$\mathbb{E}_{p_\sigma}\big[\langle s_\theta,\nabla\log p_\sigma\rangle\big] = \int \langle s_\theta(x), \nabla p_\sigma(x)\rangle\, dx = -\int p_\sigma(x)\, \nabla\!\cdot\! s_\theta(x)\, dx,$$

with the boundary term vanishing provided $p_\sigma(x)\,s_\theta(x) \to 0$ as
$\|x\|\to\infty$ (true for a bounded network against a Gaussian-tailed
$p_\sigma$). Hence **Hyvärinen's identity**:

$$J_{\mathrm{ESM}}(\theta) = \mathbb{E}_{p_\sigma}\Big[\ \nabla\!\cdot\! s_\theta(x) + \tfrac12 \|s_\theta(x)\|^2\ \Big] + C .$$

This is exact and mentions no $p_\sigma$ — only expectations under it, which
samples supply. It is also expensive: $\nabla\!\cdot\!s_\theta =
\mathrm{tr}(\nabla s_\theta)$ costs $d$ extra backward passes per training
step (the same trace that reappears in Sec. 5.4, where a stochastic estimator
becomes worth its variance). At $d = 2$ that would be affordable; at image
scale it is not, and this repo takes the route that is used in practice.

### 3.3 Denoising score matching, and the equivalence proof

Vincent (2011) removes the same term a second way, at no cost at all. Build
the noised variable *explicitly*: draw $x_0 \sim p_0$, then $x = x_0 + \sigma
\varepsilon$ with $\varepsilon \sim N(0,I)$, so that $q_\sigma(x \mid x_0) =
N(x; x_0, \sigma^2 I)$ and $p_\sigma(x) = \int p_0(x_0) q_\sigma(x\mid x_0)
dx_0$ *by construction*. Define

$$J_{\mathrm{DSM}}(\theta) = \tfrac12\,\mathbb{E}_{p_0(x_0)\,q_\sigma(x\mid x_0)}\big\|s_\theta(x) - \nabla_x \log q_\sigma(x \mid x_0)\big\|^2 . \tag{DSM}$$

**Claim.** $J_{\mathrm{ESM}}(\theta) = J_{\mathrm{DSM}}(\theta) + C'$ with
$C'$ independent of $\theta$. Same minimizer, same gradient.

*Proof.* Expand both squares. The $\|s_\theta(x)\|^2$ terms agree, because $x$
is a draw from $p_\sigma$ under either expectation — that is precisely what
the construction above guarantees. The remaining squared terms contain no
$\theta$. So only the cross terms need matching. For ESM, as in Sec. 3.2,

$$\mathbb{E}_{p_\sigma}\big[\langle s_\theta, \nabla\log p_\sigma\rangle\big] = \int \langle s_\theta(x), \nabla_x p_\sigma(x)\rangle\, dx .$$

The density cancelled — that is the one step that matters. Now push the
gradient through the definition of $p_\sigma$:

$$\nabla_x p_\sigma(x) = \int p_0(x_0)\, \nabla_x q_\sigma(x\mid x_0)\, dx_0 ,$$

so the cross term becomes

$$\iint p_0(x_0)\,\big\langle s_\theta(x),\ \nabla_x q_\sigma(x\mid x_0)\big\rangle\, dx_0\, dx = \iint p_0(x_0)\, q_\sigma(x\mid x_0)\,\big\langle s_\theta(x),\ \nabla_x \log q_\sigma(x\mid x_0)\big\rangle\, dx_0\, dx,$$

multiplying and dividing by $q_\sigma$. That is the DSM cross term exactly.
The two objectives differ by a constant. $\square$

In one sentence: *multiplying and dividing by the density turns the score of a
marginal we cannot write down into the score of a conditional we chose
ourselves.* And that conditional is a Gaussian we picked, so its score is
free:

$$\nabla_x \log q_\sigma(x\mid x_0) = -\frac{x - x_0}{\sigma^2} = -\frac{\varepsilon}{\sigma}.$$

The regression target is the **negated normalized noise** — known exactly, at
no cost, because we drew it. DSM is literally "predict the noise you added".

The name is the interpretation. The minimizer of a squared error is a
conditional mean, so the optimal $s_\theta(x)$ is the average over all $x_0$
that could have produced $x$ of the direction pointing back to $x_0$:

$$s^\star(x) = \mathbb{E}\Big[-\frac{x - x_0}{\sigma^2}\ \Big|\ x\Big] = \frac{\mathbb{E}[x_0\mid x] - x}{\sigma^2},$$

which is Tweedie's formula (Sec. 2.3) arrived at from the other side. The
optimum of DSM *is* the true noised score, and `tests/test_dsm.py` checks both
halves numerically: that ESM and DSM differ by a constant independent of the
score, and that no score beats the exact noised one on the DSM objective.

### 3.4 Weighting across levels, and where the two divisions by $\sigma$ go

Training happens at all noise levels at once, so the per-level objectives must
be combined:

$$L(\theta) = \mathbb{E}_{\sigma \sim q(\sigma)}\ \lambda(\sigma)\, J_{\mathrm{DSM}}(\theta; \sigma).$$

The target scales as $1/\sigma$, so its square scales as $1/\sigma^2$ and an
unweighted sum is dominated by the finest level by a factor
$(\sigma_{\max}/\sigma_{\min})^2$ — four orders of magnitude on this repo's
ladder ($0.05$ to $8$). The coarse levels, the ones carrying the global
structure that annealing depends on, would contribute nothing. The standard
fix (Song & Ermon 2019) is $\lambda(\sigma) = \sigma^2$, which makes every
level contribute equally:

$$\sigma^2\Big\|s_\theta + \frac{\varepsilon}{\sigma}\Big\|^2 = \big\|\sigma s_\theta + \varepsilon\big\|^2,$$

a unit-scale quantity at every $\sigma$.

Separately, the *network* is parameterized as
$s_\theta(x,\sigma) = -\epsilon_\theta(x,\sigma)/\sigma$
(`diffusion/model.py`), so that its output regresses a standard normal at every
level instead of having to learn the $1/\sigma$ envelope with its weights.
Substituting, the whole objective collapses to

$$L(\theta) = \mathbb{E}\,\big\|\epsilon_\theta(x_0 + \sigma\varepsilon,\ \sigma) - \varepsilon\big\|^2,$$

which is the form implemented in `dsm.dsm_loss` and the form DDPM trains.

The two divisions by $\sigma$ look redundant and are not: the one in the
architecture fixes the *output* scale, the one in the loss weight fixes the
*gradient* scale across levels, and dropping either reintroduces the
imbalance. The price of the architectural one is stated where it is paid: the
division is exact arithmetic, so any error in $\epsilon_\theta$ is amplified by
$1/\sigma$, and the fine end of the ladder is where a learned score is worst
*even though its own loss looks best* there. The measured table in the README
shows exactly that.

$q(\sigma)$ is log-uniform over the ladder, matching its geometric spacing
(Sec. 6.2): the levels the sampler will visit are geometric, so uniform-in-log
is uniform over the levels that matter.

### 3.5 The loss has a floor, and the floor runs the wrong way

The DSM loss does **not** go to zero, and reading a loss curve without knowing
where its floor is says nothing. The residual is irreducible because
$\varepsilon$ is not a function of $x$: the best possible prediction is the
conditional mean, so

$$L^\star(\sigma) = \mathbb{E}\big\|\varepsilon - \mathbb{E}[\varepsilon\mid x]\big\|^2, \qquad \mathbb{E}[\varepsilon \mid x] = \frac{x - \mathbb{E}[x_0\mid x]}{\sigma} = -\sigma\, s_\sigma(x)$$

by Tweedie again — so on this repo's targets the floor is computable exactly
(`dsm.irreducible_dsm_loss`, `dsm.eval_dsm_floor`). Its two limits are
counterintuitive and worth stating:

- $\sigma \to 0$: $x$ reveals $x_0$ and therefore says nothing about
  $\varepsilon$, so $L^\star \to \mathbb{E}\|\varepsilon\|^2 = d$. The **fine**
  levels are where the loss is largest.
- $\sigma \to \infty$: $x \approx \sigma\varepsilon$, so $\varepsilon$ is
  nearly determined by $x$ and $L^\star \to 0$.

So a falling loss curve can mean the network is learning, or merely that the
batch happened to draw coarse levels. Averaged over the training measure the
floor here is $1.203$, a well-trained network
reports $1.212$ and a barely-trained one $1.228$ — while one minibatch's loss
has a standard deviation of $0.073$, eight times the entire remaining excess.
That is why the repo reports error against the exact score per level, and why
the training curve is plotted against this floor rather than against zero.

---

## 4. The forward SDE, Anderson's reverse SDE, and DDPM as its discretization

### 4.1 The forward process

The forward process takes data and destroys it into noise. This repo uses the
**variance-exploding** (VE) form: no drift, and a variance that grows,

$$dx = g(t)\, dw, \qquad g(t)^2 = \frac{d\,[\sigma(t)^2]}{dt},$$

whose solution from $x(0) = x_0$ is $x(t) = x_0 + \sigma(t)\varepsilon$ with
$\varepsilon \sim N(0,I)$ — so the marginal at time $t$ is exactly the
$p_{\sigma(t)}$ of Sec. 2.2, and the whole apparatus of noised marginals is
the SDE's time-marginal family. (The alternative, DDPM's
variance-*preserving* form, adds a contracting drift so that the variance
stays $O(1)$; the two differ by a time-dependent rescaling of $x$, and Sec. 5.2
shows why the choice matters less than it looks.)

Discretely, on a ladder $\sigma_L > \dots > \sigma_1$, the same process is the
Markov chain

$$x_i = x_0 + \sigma_i \varepsilon, \qquad x_i = x_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2}\; z,\quad z\sim N(0,I),$$

which is the entire content of "variance exploding": the noise added between
two levels is independent of everything earlier, and the variances add.

The forward density obeys the Fokker–Planck equation, which for a driftless
diffusion is the heat equation:

$$\partial_t p_t = \tfrac12 g(t)^2\, \Delta p_t . \tag{FP}$$

### 4.2 Anderson's reverse-time SDE

**Statement** (Anderson 1982). The time reversal of $dx = f(x,t)dt + g(t)dw$
is again a diffusion, and it satisfies

$$dx = \big[f(x,t) - g(t)^2\, \nabla_x \log p_t(x)\big]\,dt + g(t)\,d\bar w,$$

where $dt$ is now a *negative* time increment and $\bar w$ is a Wiener process
running backwards. For our driftless forward process this is

$$dx = -g(t)^2\, s_t(x)\, dt + g(t)\, d\bar w .$$

Run from $t = T$ down to $t = 0$ starting at a draw from $p_T$, its marginals
are the forward marginals in reverse, so it ends at a draw from $p_0$.
**Sampling is running the noising process backwards, and the score is the only
thing you need in order to do it.**

*Where it comes from.* A rigorous proof is Anderson's; the following
Fokker–Planck matching gets the drift and shows why the score appears, which
is what the code needs. Let $s := T - t$ run backwards and write
$\bar p(x,s) := p_{T-s}(x)$. If the reverse process is a diffusion
$dx = \bar f\,ds + \bar g\,d\bar w$, its density obeys
$\partial_s \bar p = -\nabla\!\cdot\!(\bar f \bar p) + \tfrac12 \bar g^2
\Delta \bar p$. But $\partial_s \bar p = -\partial_t p$, and by (FP)
$\partial_t p = \tfrac12 g^2 \Delta p$. Taking $\bar g = g$ (the reversal is a
diffusion of the same intensity) and equating:

$$-\nabla\!\cdot\!(\bar f\, p) + \tfrac12 g^2 \Delta p = -\tfrac12 g^2 \Delta p \quad\Longrightarrow\quad \nabla\!\cdot\!(\bar f\, p) = g^2\, \Delta p .$$

Now the identity that makes the score appear — the same one Sec. 5.1 uses:

$$\Delta p = \nabla\!\cdot\!(\nabla p) = \nabla\!\cdot\!\big(p\, \nabla \log p\big).$$

So $\nabla\!\cdot\!(\bar f p) = \nabla\!\cdot\!(p\, g^2 \nabla \log p)$, which
is satisfied by

$$\bar f = g^2\, \nabla\log p = g^2 s .$$

Translating $\bar f\,ds$ back into $-\bar f\,dt$ gives Anderson's drift.

Two honest caveats about that argument. It fixes $\bar f$ only up to a field
whose divergence against $p$ vanishes; matching the *fields* rather than their
divergences is the choice that gives the reversal (Anderson's theorem is what
says this choice is the right one). And it assumes the reversal is a diffusion
at all, which is the substantive part of the theorem, not of this calculation.

*Intuition.* The forward process spreads probability mass out; run backwards,
something has to push it back together, and $\nabla \log p$ is the field that
points uphill in density. The factor $g^2$ says the push must be exactly as
strong as the spreading it is undoing. The remaining $g\,d\bar w$ is the same
noise as forward — reversal does not remove randomness, it adds a drift that
overcomes it.

### 4.3 The reverse conditional, exactly, on the discrete ladder

The sampler this repo implements is not a discretization of the SDE written
directly; it is the discrete chain's own reverse conditional, which is exact
before one approximation is made. Working discretely makes that approximation
explicit and gives a closed form to test against.

Condition on the clean point. Both $x_i$ and $x_{i-1}$ are $x_0$ plus
independent Gaussian noise, so $q(x_{i-1}\mid x_i, x_0)$ is Gaussian, and it is
the standard "combine two Gaussian observations, weighted by precisions"
answer. Write $u := x_{i-1} - x_0$ and $v := x_i - x_{i-1}$: independent, with
variances $\sigma_{i-1}^2$ and $\sigma_i^2 - \sigma_{i-1}^2$. Conditioning a
pair of independent Gaussians on their sum $u + v = x_i - x_0$ gives a Gaussian
with mean $\frac{\sigma_{i-1}^2}{\sigma_i^2}(x_i - x_0)$ and variance
$\frac{\sigma_{i-1}^2(\sigma_i^2 - \sigma_{i-1}^2)}{\sigma_i^2}$. Adding $x_0$
back and writing $a := \sigma_{i-1}^2/\sigma_i^2$:

$$q(x_{i-1}\mid x_i, x_0) = N\Big(a\,x_i + (1-a)\,x_0,\ \ a\,(\sigma_i^2 - \sigma_{i-1}^2)\, I\Big).$$

### 4.4 DDPM ancestral sampling, and the variance it is supposed to get wrong

$x_0$ is of course not available at sampling time. Replace it by its posterior
mean, which Tweedie's formula (Sec. 2.3) hands over in terms of the score and
nothing else: $\mathbb{E}[x_0\mid x_i] = x_i + \sigma_i^2 s(x_i,\sigma_i)$.
Substituting collapses the two mean terms into one:

$$a\,x_i + (1-a)\big(x_i + \sigma_i^2 s\big) = x_i + (\sigma_i^2 - \sigma_{i-1}^2)\, s,$$

and the sampler is

$$\boxed{\;x_{i-1} = x_i + (\sigma_i^2 - \sigma_{i-1}^2)\, s(x_i, \sigma_i) + \sqrt{\frac{\sigma_{i-1}^2(\sigma_i^2 - \sigma_{i-1}^2)}{\sigma_i^2}}\; z\;}$$

This is DDPM's ancestral sampler (Ho et al. 2020, Eq. 11) written for the
variance-exploding chain rather than the variance-preserving one; Song et al.
(2021, App. F) give the same update as the ancestral sampler of the SMLD
model. Setting $\sigma_{i-1} = 0$ makes the noise term vanish and the mean
equal $\mathbb{E}[x_0\mid x_i]$ exactly — so the final denoising step is not an
extra heuristic, it is the last rung of the ladder.

It is also the Euler discretization of Anderson's SDE in the $\sigma$ clock,
which is worth checking because it ties Sec. 4.2 to the formula above. With
$g^2 dt = d(\sigma^2)$, the reverse drift $-g^2 s\,dt$ over a step from
$\sigma_i$ to $\sigma_{i-1}$ contributes $(\sigma_i^2 - \sigma_{i-1}^2)s$ —
the mean update exactly — and the diffusion term contributes variance
$\sigma_i^2 - \sigma_{i-1}^2$, which the exact conditional above discounts by
the factor $a \le 1$. The two agree to first order in the gap and the exact
version is the one implemented.

**What the plug-in costs.** Only one step above is inexact: $p(x_0\mid x_i)$
was replaced by a point mass at its mean. The true reverse conditional is the
mixture

$$q(x_{i-1}\mid x_i) = \int q(x_{i-1}\mid x_i, x_0)\, p(x_0\mid x_i)\, dx_0,$$

whose mean agrees (the plug-in enters linearly) but whose covariance is larger
by the posterior spread propagated through the same linear map:

$$\mathrm{Cov}[x_{i-1}\mid x_i] = a\,(\sigma_i^2 - \sigma_{i-1}^2)\,I + (1-a)^2\,\mathrm{Cov}[x_0\mid x_i].$$

So **ancestral sampling under-disperses**, by an amount that is second order
in the gap between adjacent levels ($1 - a \to 0$ as the ladder is refined) and
that vanishes wherever denoising is nearly deterministic. On a Gaussian target
the deficit is available in closed form, and `samplers.ddpm_gaussian_variance`
returns it — which is how the tests assert what the sampler *does* rather than
what it should do in the limit: a coarse ladder is expected to come back
narrow, and it does, by the predicted factor. Applied to one component of the
mixture it predicts the *within-mode* width to under 1% once the ladder is fine
enough to resolve a mode (README table), which is a check nobody ordered and
the sharpest evidence that the derivation above describes the code.

### 4.5 Three sources of error, and how to take them apart

A sample can be wrong for three reasons, and on a target with a known score
each has an off switch — which is the reason `experiments/sampling.py` exists
in the form it does:

1. **Prior mismatch.** The chain starts from $N(0,\sigma_{\max}^2 I)$, but the
   true top-of-ladder distribution is $p_{\sigma_{\max}}$, which is that
   convolved with the data. They agree only as $\sigma_{\max}\to\infty$.
   Switched off by starting from an exact draw from $p_{\sigma_{\max}}$.
2. **Discretization.** The plug-in of Sec. 4.4, at finitely many levels.
   Removed only in the limit, so it is measured by refining the ladder.
3. **Score estimation.** $s_\theta$ instead of $s$. Switched off by passing the
   analytic noised score through the identical sampler.

---

## 5. The probability-flow ODE and likelihoods

### 5.1 A deterministic flow with the same marginals

Return to (FP) and read it as a transport equation. Using
$\Delta p = \nabla\!\cdot\!(p\,\nabla\log p)$ again,

$$\partial_t p = \tfrac12 g^2\, \nabla\!\cdot\!\big(p\, \nabla \log p\big) = -\nabla\!\cdot\!\big(p\, v\big), \qquad v(x,t) := -\tfrac12 g(t)^2\, \nabla\log p_t(x).$$

A **continuity equation** is exactly the statement that $p$ is transported by
the velocity field $v$. So the *deterministic* ODE $dx/dt = v(x,t)$ has the
same time marginals as the SDE (Song et al. 2021, Eq. 13, with $f = 0$).
Nothing was approximated: the two processes differ in their **paths**, not
their **marginals**.

More is true, and it explains the relationship between the two samplers. For
any $\lambda \ge 0$ the process

$$dx = -\tfrac{1+\lambda^2}{2}\,g^2\, s\,dt + \lambda\, g\, d\bar w$$

has these same marginals: $\lambda = 1$ is Anderson's reverse SDE (Sec. 4.2)
and $\lambda = 0$ is the probability-flow ODE, and the difference between them
is a Langevin term at the current noise level, which leaves $p_\sigma$
invariant by Sec. 6.1. That is the precise version of "the stochastic sampler
is the deterministic one plus noise that does nothing" — in *exact*
arithmetic. In inexact arithmetic it does two opposite things: it
re-randomizes, so errors made at coarse levels can be forgotten (which is why
the stochastic sampler is often more forgiving of a bad score), and it adds
variance. The repo measured both sides of that coin, and the more interesting
one is in Sec. 6.4.

### 5.2 $\sigma$ as the clock: the schedule cancels

$\sigma(t)$ is strictly increasing, so it can serve as the independent variable
instead of $t$. With $g^2 = d(\sigma^2)/dt = 2\sigma\,\dot\sigma$,

$$\frac{dx}{d\sigma} = \frac{dx/dt}{d\sigma/dt} = \frac{-\tfrac12 \big(2\sigma \dot\sigma\big) s}{\dot\sigma} \quad\Longrightarrow\quad \boxed{\ \frac{dx}{d\sigma} = -\sigma\, s(x,\sigma).\ }$$

**The schedule has vanished.** Every VE noise schedule is a reparameterization
of this one trajectory, so the only thing a schedule can change is *where the
integration nodes are placed* — which is a property of the solver, not of the
model (Karras et al. 2022 make the same point). Every function in
`diffusion/ode.py` therefore takes a ladder of $\sigma$ values and treats it as
a set of nodes.

Two consequences worth stating because they are checkable:

- **The exact flow on a Gaussian.** For $N(\mu, v I)$ the noised score is
  $-(x-\mu)/(v+\sigma^2)$, so $dx/d\sigma = \sigma(x-\mu)/(v+\sigma^2)$, which
  separates: $d\log\|x - \mu\| = \sigma\,d\sigma/(v+\sigma^2) = \tfrac12
  d\log(v+\sigma^2)$, and integrating,

  $$x(\sigma_1) = \mu + \big(x(\sigma_0) - \mu\big)\sqrt{\frac{v + \sigma_1^2}{v + \sigma_0^2}} .$$

  The flow is a pure radial scaling. `ode.gaussian_pf_ode_scale` returns that
  factor and the tests use it to measure RK4's order as $4.0$ and Euler's as
  $1.0$ *pointwise*, rather than trusting the label on the method.
- **One Euler step of the ODE is half the ancestral mean update.** From
  $\sigma_i$ to $\sigma_{i-1}$ it is $\sigma_i(\sigma_i - \sigma_{i-1})s$
  against $(\sigma_i^2 - \sigma_{i-1}^2)s = (\sigma_i - \sigma_{i-1})(\sigma_i
  + \sigma_{i-1})s$ — a factor of about 2 for adjacent levels. That is the
  $\lambda$ family of Sec. 5.1 seen one step at a time, and
  `tests/test_ode.py` pins it.

The final stub of the trajectory obeys the same rule: integrating to
$\sigma = 0$ is unavailable (the network is conditioned on $\log\sigma$), and
the residual Euler step $x \mathrel{+}= (\sigma_{\min}^2/2)\, s$ is exactly
**half** the ancestral sampler's Tweedie denoise $x + \sigma_{\min}^2 s$. Not
a discrepancy but the difference between the objects: Tweedie jumps to
$\mathbb{E}[x_0\mid x]$, the ODE only travels the arc it has left.

### 5.3 Likelihoods by instantaneous change of variables

Because the flow is a diffeomorphism, it carries densities as well as points.
Differentiate $\log p_s$ along a trajectory $x(s)$ of $dx/ds = v(x,s)$:

$$\frac{d}{ds}\log p_s(x(s)) = \partial_s \log p_s + \langle v, \nabla \log p_s\rangle = \frac{-\nabla\!\cdot\!(p_s v)}{p_s} + \langle v, \nabla\log p_s\rangle = -\nabla\!\cdot\! v(x(s), s),$$

expanding $\nabla\!\cdot\!(p v) = p\,\nabla\!\cdot\!v + \langle v, \nabla
p\rangle$ and using the continuity equation of Sec. 5.1 (Chen et al. 2018,
Eq. 2). Integrating *up* the ladder from $\sigma_{\min}$ to $\sigma_{\max}$
and rearranging,

$$\log p_{\sigma_{\min}}(x) = \log p_{\sigma_{\max}}(z) + \int_{\sigma_{\min}}^{\sigma_{\max}} \nabla\!\cdot\! v\,\ ds,$$

with $z$ the endpoint of the trajectory started at $x$. Both terms are
computable: the endpoint density is the known prior at the top of the ladder,
and with $v = -\sigma s$ the divergence is $-\sigma\,\mathrm{tr}(\nabla s)$.

### 5.4 The trace, and Hutchinson's estimator

The trace is the expensive part — $d$ backward passes. Hutchinson's (1989)
estimator replaces it with

$$\mathrm{tr}(A) = \mathbb{E}_\varepsilon\big[\varepsilon^\top A\, \varepsilon\big] \quad\text{for any } \varepsilon \text{ with } \mathbb{E}[\varepsilon\varepsilon^\top] = I,$$

which follows from $\mathbb{E}[\varepsilon^\top A \varepsilon] =
\mathbb{E}\,\mathrm{tr}(A\varepsilon\varepsilon^\top) =
\mathrm{tr}(A\,\mathbb{E}[\varepsilon\varepsilon^\top]) = \mathrm{tr}(A)$, and
costs one vector-Jacobian product per probe (Grathwohl et al. 2019 in this
role). Two details that matter in practice and are both enforced in the code:

1. **Probes are drawn once per solve, not per evaluation.** Resampling inside
   the integrator makes the vector field discontinuous in $s$ and silently
   destroys the solver's order; with fixed probes the augmented system is an
   ordinary ODE that RK4 integrates at fourth order.
2. **The estimator is unbiased for $\log p$, hence biased for $p$.** Reporting
   a mean log-likelihood is fine; exponentiating a single estimate is not.
   Rademacher probes are the default: for a diagonal Jacobian every
   $\varepsilon_i^2 = 1$ so the trace comes back *exactly*, where Gaussian
   probes still carry variance from the diagonal — and at points where one
   mixture component dominates, the noised score's Jacobian **is** isotropic,
   so this is not hypothetical.

Because the noised mixture's density is closed-form, the whole pipeline has a
ground truth. Run on the exact score, `ode.pf_ode_log_prob` must reproduce
$\log p_{\sigma_{\min}}$ to integration error. Run on the *learned* score it is
a genuine normalized model density $q$, so averaging $\log p - \log q$ over
exact draws is a Monte Carlo estimate of $\mathrm{KL}(p_{\sigma_{\min}} \| q)
\ge 0$ — a number in nats for an error the two-sample metrics could only place
"somewhere inside the floor band".

---

## 6. The Langevin bridge back to mcmc-from-scratch

### 6.1 Langevin dynamics, and the accept step that is missing

The overdamped Langevin SDE

$$dX_t = \nabla \log p(X_t)\, dt + \sqrt2\, dW_t$$

has $p$ as its stationary distribution — check it against (FP): with $f =
\nabla\log p$ and $g = \sqrt2$, $\partial_t p = -\nabla\!\cdot\!(p\nabla\log
p) + \Delta p = -\Delta p + \Delta p = 0$, using the same identity as Sec. 4.2.
Nothing in that statement mentions $p$ itself, only its score, which is fact
(1) of Sec. 1.

Euler–Maruyama at step $\tau$ gives the **unadjusted Langevin algorithm**:

$$x \leftarrow x + \frac{\tau}{2}\, s(x) + \sqrt{\tau}\, z, \qquad z \sim N(0,I).$$

That update is *exactly* the MALA proposal in the sibling repo
[mcmc-from-scratch](https://github.com/porth-bot/mcmc-from-scratch)
(`mcmc/mala.py`, written there with $\tau = \varepsilon^2$; theory Sec. 7,
Exercise 1) **with the Metropolis accept step deleted**. The deletion is not
free:

- **MALA is exact.** The Hastings correction repairs the discretization error
  at any step size; the chain's stationary law is $p$ on the nose.
- **ULA is biased**, with the bias $O(\tau)$ and not removable by running
  longer. On $N(0,1)$ the update is linear, $x' = (1 - \tau/2)x +
  \sqrt\tau z$, so a stationary variance $V$ must satisfy $V = V(1-\tau/2)^2 +
  \tau$, i.e. $V(\tau - \tau^2/4) = \tau$:

  $$V = \frac{1}{1 - \tau/4}.$$

  Over-dispersed by 6.7% at $\tau = 0.25$ and 19% at $\tau = 0.64$, and
  transient at $\tau \ge 4$. `langevin.ula_gaussian_variance` returns it and
  `tests/test_langevin.py` measures it from this side, as mcmc's Exercise 1(c)
  does from the other.

So why accept the bias? Because in a diffusion model **the density is not
available**. A learned network gives $s_\theta(x,\sigma)$ and no $\log p$, so
there is nothing to put in a Metropolis ratio: the accept step is not skipped
for speed, it is unavailable in principle. That is the honest framing of the
trade — score-based sampling gives up MALA's exactness to be able to run at
all on a target known only through its score.

### 6.2 Annealing, and the step-size schedule it forces

ULA is also a *local* method. Crossing a barrier of depth $\Delta$ nats is an
$e^{-\Delta}$ event per attempt, and this repo's bimodal target has $\Delta =
(\text{sep}/2/\text{std})^2/2 = 36$, so a single chain never leaves its mode.
Song & Ermon's (2019) fix is to run ULA down the ladder of *noised* targets
$p_{\sigma_L}, \dots, p_{\sigma_1}$, carrying the state forward. At large
$\sigma$ the noised density is nearly unimodal — Sec. 2.2's inflated
covariances have smeared the modes into each other — so the chain moves
globally and settles the coarse structure; as $\sigma$ shrinks the target
sharpens back toward $p_0$ and the chain refines locally. **The mixing problem
is solved at the scale where it is easy, and the answer is carried down.**

The ladder is geometric, not linear, because what must stay bounded between
consecutive levels is the *ratio* $\sigma_i/\sigma_{i-1}$: the chain arrives at
level $i-1$ holding draws from $p_{\sigma_i}$, and that is a useful
initialization only if the two densities overlap substantially (Song & Ermon
2020). Linear spacing would take enormous steps in ratio at the low end,
exactly where the density is sharpest.

The step size must shrink with the level. Song & Ermon set

$$\tau_i = \tau\,\frac{\sigma_i^2}{\sigma_1^2},$$

which keeps the ratio of signal to noise constant across levels: for a
component of scale $\sigma_i$ the score is $O(1/\sigma_i)$, so the drift
$(\tau_i/2)s$ is $O(\tau \sigma_i/\sigma_1^2)$, matching the injected noise
$\sqrt{\tau_i} = \sqrt\tau\,\sigma_i/\sigma_1$ up to a level-independent
constant. Without it a single step size is either too small to move at the
coarse levels or too large to be stable at the fine ones — the same resonance
problem a unit mass matrix has in HMC (mcmc's theory Sec. 4.8), reappearing
along the noise axis instead of across coordinates.

### 6.3 A step-size bound that holds for every target

Sec. 2.3 established $\nabla s_\sigma \succeq -I/\sigma^2$ for **any** base
density. Linearize the ULA update around a point: the deterministic part
multiplies deviations by $I + (\tau/2)\nabla s$, with eigenvalues $1 +
(\tau/2)\lambda$. Contraction needs $|1 + (\tau/2)\lambda| < 1$, i.e.
$\tau|\lambda| < 4$, and since $\lambda \ge -1/\sigma^2$ that holds whenever

$$\tau_i < 4\sigma_i^2, \quad\text{which under the schedule above is}\quad \tau < 4\sigma_1^2$$

— one condition, on the finest level, that makes every level stable. It is the
same $4$ that appears in $V = 1/(1-\tau/4)$ above, which is this statement for
$N(0,1)$ where $\lambda = -1$ exactly.

`langevin.annealed_langevin` **refuses** a $\tau$ that violates it, because the
failure is otherwise nearly invisible. An unstable ladder does not hand back
obvious garbage: the coarse levels blow the state up geometrically (to $10^{80}$
in this repo's own bimodal test) and the fine levels, which are strongly
contractive, squash it back into range. What comes out looks like plausible
samples with roughly the right support — but the coarse exploration that was
supposed to decide *which mode each chain belongs to* has been destroyed, so
the mode weights come back wrong. On a symmetric target that is undetectable;
an asymmetric mixture is a permanent test here because it reads $0.53$ instead
of $0.75$. And the bound is necessary, not sufficient: at half the bound the
same test still reads $0.70$.

### 6.4 Where the bridge stops being an analogy

The two repos share an update, but they are pointed at different questions, and
one measured result makes the difference concrete. Annealed Langevin with the
*learned* score gets **worse** with more compute — sliced $W_2$ of $0.274$ at
200 score evaluations against $0.807$ at 2000 — while the same sampler with the
*exact* score does not degrade at all.

This is not a bug and it is not the step size. Langevin converges to the
stationary distribution *of the score it is handed*, so a biased score has a
biased stationary law and more steps converge to it harder. In MCMC terms: the
chain is doing its job perfectly, on the wrong target, and its asymptotic
guarantee is the mechanism of the failure rather than a defence against it.
Ancestral sampling (Sec. 4.4) never equilibrates — it visits each level exactly
once — so the same score error stays a perturbation instead of becoming a
limit.

That inverts the usual reading of a compute knob, and it is the sharpest
statement this repo can make about what changes when the target is learned
rather than known: **an exact-target sampler's virtues become liabilities when
the target is an estimate.** Only the exact-score control makes it a finding
instead of a guess.

---

## References

- Anderson (1982), "Reverse-time diffusion equation models", *Stoch. Process. Appl.* 12 (the reverse-time SDE, Sec. 4.2).
- Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", *JMLR* 6 (Sec. 3.2).
- Vincent (2011), "A connection between score matching and denoising autoencoders", *Neural Comput.* 23 (Sec. 3.3).
- Robbins (1956) / Efron (2011), "Tweedie's formula and selection bias", *JASA* 106 (Sec. 2.3).
- Song & Ermon (2019), "Generative modeling by estimating gradients of the data distribution", *NeurIPS* (annealed Langevin, the $\sigma^2$ weighting).
- Song & Ermon (2020), "Improved techniques for training score-based generative models", *NeurIPS* (ladder spacing, Sec. 6.2).
- Ho, Jain & Abbeel (2020), "Denoising diffusion probabilistic models", *NeurIPS* (the ancestral sampler, Eq. 11; $\epsilon$-prediction).
- Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole (2021), "Score-based generative modeling through stochastic differential equations", *ICLR* (the SDE framing; probability-flow ODE, Eq. 13; App. D likelihoods, App. F the VE ancestral sampler).
- Karras, Aittala, Aila & Laine (2022), "Elucidating the design space of diffusion-based generative models", *NeurIPS* ($\sigma$ as the clock; node placement as a solver choice).
- Chen, Rubanova, Bettencourt & Duvenaud (2018), "Neural ordinary differential equations", *NeurIPS* (instantaneous change of variables, Sec. 5.3).
- Hutchinson (1989), *Commun. Statist. Simul. Comput.* 18 (the trace estimator); Grathwohl, Chen, Bettencourt, Sutskever & Duvenaud (2019), "FFJORD", *ICLR* (its use in a continuous flow).
- Roberts & Tweedie (1996), *Bernoulli* 2 (ULA's bias and MALA's correction — the mcmc-from-scratch side of Sec. 6.1).
