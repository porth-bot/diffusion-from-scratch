"""The score network: an MLP conditioned on the noise level.

Days 1-2 of this repo sampled with the *exact* score. This module is the
first thing that has to be learned, and it has to be learned as a function of
two arguments, not one:

    ``s_theta(x, sigma)  ~=  grad_x log p_sigma(x)``,

because :func:`diffusion.langevin.annealed_langevin` walks down a whole ladder
of noise levels and needs the score of each. One network for the entire ladder
is not a convenience, it is the method (Song & Ermon 2019): the levels share
almost all of their structure, and a separate network per level would both
waste that sharing and leave the score undefined between levels.

Two design decisions carry the module, and both are about *scale*.

Output parameterization: predict the noise, divide by sigma
-----------------------------------------------------------
The regression target for the score at noise level ``sigma`` is
``-eps/sigma`` (see :mod:`diffusion.dsm`), whose magnitude is ``O(1/sigma)``.
Across the ladder used here that is a range of ~1e2 in the target and ~1e4 in
its square. A network asked to output that directly must learn the ``1/sigma``
envelope with its weights before it can learn anything about the density, and
the loss at the fine levels swamps the loss at the coarse ones.

So the network's *output* is the normalized noise and the score is formed by
dividing:

    ``s_theta(x, sigma) = -eps_theta(x, sigma) / sigma``.

Now ``eps_theta`` regresses a standard normal at every level -- unit scale,
level-independent -- and the ``sigma^2``-weighted DSM objective collapses to
plain ``|| eps_theta - eps ||^2`` (:func:`diffusion.dsm.dsm_loss`). This is
NCSN's ``s_theta = s(x)/sigma`` and DDPM's epsilon-prediction, which are the
same choice written two ways.

The division is exact arithmetic, not a fitted approximation, so the *shape*
of the score is all that is learned. One consequence is worth stating because
it is a limitation and not a feature: at very small ``sigma`` any error in
``eps_theta`` is amplified by ``1/sigma``, so the fine end of the ladder is
where a learned score is worst even though its own loss looks best. The
experiment measures exactly that.

Input conditioning: a sinusoidal embedding of log sigma
--------------------------------------------------------
``sigma`` spans orders of magnitude, so the network conditions on
``log sigma``, and it does so through a sinusoidal embedding rather than by
concatenating the scalar. The embedding (Vaswani et al. 2017, as used for
diffusion timesteps by Ho et al. 2020) is written out in
:class:`SinusoidalEmbedding`:

    ``gamma(t) = [ sin(w_1 t), cos(w_1 t), ..., sin(w_m t), cos(w_m t) ]``

with geometrically spaced frequencies. A raw scalar entering a linear layer
can only be used affinely by the first layer, so distinguishing nearby noise
levels costs depth; a sinusoidal embedding makes ``t`` linearly separable at
high resolution immediately, exactly the argument for Fourier features in
:mod:`pinn.features` in the sibling repo pinn-from-scratch. The difference
from that use is instructive: there the features are *random* because the
target's frequency content is unknown, here they are *deterministic and
geometric* because the axis being embedded is one known, bounded, log-scaled
coordinate.

``theory/derivations.md`` Sec. 3.4 puts the output parameterization next to the
loss weighting it interacts with, which is the argument for why both divisions
by ``sigma`` are needed.
"""

from __future__ import annotations

import torch
from torch import nn

Tensor = torch.Tensor

__all__ = [
    "SinusoidalEmbedding",
    "ScoreNet",
    "count_parameters",
]


def count_parameters(module: nn.Module) -> int:
    """Number of trainable scalars -- reported in the experiment tables."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class SinusoidalEmbedding(nn.Module):
    r"""Map a scalar to ``[sin(w_i t), cos(w_i t)]`` at geometric frequencies.

    For an even embedding dimension ``dim = 2m`` the frequencies are

        ``w_i = w_min * (w_max / w_min)^{i / (m - 1)}``,   ``i = 0..m-1``,

    i.e. geometrically spaced over ``[w_min, w_max]``, and the output is the
    concatenation of ``sin(w_i t)`` and ``cos(w_i t)``.

    **Why sin and cos together.** The pair at frequency ``w`` is a point on
    the unit circle at angle ``w t``, so a linear layer reading it can form
    ``cos(w(t - t_0))`` for *any* phase ``t_0`` by taking a linear combination
    -- a band-pass detector centred wherever the network wants it. With only
    ``sin`` the phase would be fixed at 0 and the representation would be
    degenerate at ``t = 0``.

    **Why geometric spacing.** The coordinate embedded here is ``log sigma``
    over a range of a few units, and the resolution a frequency ``w``
    distinguishes is ``~1/w``. Geometric spacing gives constant *relative*
    resolution across the range, matching the fact that the interesting
    structure in ``sigma`` is itself multiplicative (the ladder in
    :func:`diffusion.langevin.geometric_sigmas` is geometric for the same
    reason). Linear spacing would put nearly all the frequencies at the coarse
    end.

    The defaults ``w_min = 1``, ``w_max = 100`` cover a ``log sigma`` range of
    a few units: the slowest component varies over the whole range (it is what
    tells the network roughly where it is), the fastest resolves ~0.06 in
    ``log sigma``, well finer than the spacing of any ladder used here.
    """

    #: Declared so mypy sees the registered buffer as a tensor (the same
    #: annotation grokking-transformer's attention needs for its causal mask).
    freqs: Tensor

    def __init__(self, dim: int = 32, w_min: float = 1.0, w_max: float = 100.0):
        super().__init__()
        if dim < 2 or dim % 2 != 0:
            raise ValueError(f"embedding dim must be even and >= 2, got {dim}")
        if not 0 < w_min <= w_max:
            raise ValueError(f"need 0 < w_min <= w_max, got ({w_min}, {w_max})")
        half = dim // 2
        if half == 1:
            freqs = torch.tensor([w_min], dtype=torch.float64)
        else:
            ratio = (w_max / w_min) ** (1.0 / (half - 1))
            freqs = w_min * ratio ** torch.arange(half, dtype=torch.float64)
        self.dim = dim
        # A buffer, not a parameter: these frequencies are fixed by design,
        # exactly like the frozen B matrix of a random Fourier feature map.
        self.register_buffer("freqs", freqs.to(torch.float32))

    def forward(self, t: Tensor) -> Tensor:
        """``t`` of shape ``(n,)`` -> ``(n, dim)``."""
        t = torch.as_tensor(t)
        if t.ndim != 1:
            raise ValueError(f"expected a 1-D batch of scalars, got shape {tuple(t.shape)}")
        ang = t.unsqueeze(1) * self.freqs.to(t.dtype)  # (n, dim/2)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class ScoreNet(nn.Module):
    """``s_theta(x, sigma)``: a noise-conditioned MLP, score by division.

    Architecture, deliberately plain because the point of the repo is the
    objective and the validation, not the network:

    1. embed ``log sigma`` sinusoidally (:class:`SinusoidalEmbedding`) and
       pass it through a two-layer MLP, giving a conditioning vector;
    2. concatenate that with ``x`` and run a SiLU MLP of the requested width
       and depth;
    3. a linear head produces ``eps_theta(x, sigma)`` of the same shape as
       ``x``;
    4. :meth:`forward` returns the *score* ``-eps_theta / sigma``.

    SiLU rather than ReLU because the score field is smooth and a ReLU network
    has a piecewise-constant Jacobian -- the same reason
    :mod:`pinn.model` uses tanh for PDE residuals, where second derivatives
    are needed. Here only first derivatives are taken (of the *loss*, not the
    output), so the requirement is milder, but a kinked score field shows up
    directly in the ODE sampler of Day 26.

    ``sigma`` may be a scalar or a per-point tensor of shape ``(n,)``: the
    training loop draws a different noise level for every point in the batch,
    while the samplers evaluate a whole batch at one level.
    """

    def __init__(
        self,
        dim: int = 2,
        width: int = 128,
        depth: int = 4,
        embed_dim: int = 32,
        cond_width: int | None = None,
        w_max: float = 100.0,
    ):
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        cond = width if cond_width is None else cond_width
        self.dim = dim
        self.embedding = SinusoidalEmbedding(embed_dim, w_max=w_max)
        self.cond = nn.Sequential(
            nn.Linear(embed_dim, cond), nn.SiLU(), nn.Linear(cond, cond)
        )
        layers: list[nn.Module] = [nn.Linear(dim + cond, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers.append(nn.Linear(width, dim))
        self.net = nn.Sequential(*layers)

    def _broadcast_sigma(self, x: Tensor, sigma: float | Tensor) -> Tensor:
        s = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        if s.ndim == 0:
            s = s.expand(x.shape[0])
        elif s.ndim != 1 or s.shape[0] != x.shape[0]:
            raise ValueError(
                f"sigma must be a scalar or shape ({x.shape[0]},), got {tuple(s.shape)}"
            )
        if bool((s <= 0).any()):
            raise ValueError("sigma must be positive")
        return s

    def epsilon(self, x: Tensor, sigma: float | Tensor) -> Tensor:
        """The raw network output ``eps_theta(x, sigma)`` -- unit scale at every level."""
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected points of shape (n, {self.dim}), got {tuple(x.shape)}")
        s = self._broadcast_sigma(x, sigma)
        h = self.cond(self.embedding(torch.log(s)))
        return self.net(torch.cat([x, h], dim=1))

    def forward(self, x: Tensor, sigma: float | Tensor) -> Tensor:
        """``s_theta(x, sigma) = -eps_theta(x, sigma) / sigma`` -> ``(n, d)``."""
        s = self._broadcast_sigma(x, sigma)
        return -self.epsilon(x, s) / s.unsqueeze(1)

    def score_fn(self, sigma: float):
        """The single-argument ``s(x)`` that :mod:`diffusion.langevin` wants.

        Returned under ``torch.no_grad`` semantics is *not* imposed here --
        the caller decides, since Day 26's ODE sampler differentiates through
        the score while the Langevin samplers do not.
        """

        def s(x: Tensor) -> Tensor:
            return self.forward(x, sigma)

        return s

    def noised_score_fn(self):
        """The two-argument ``s(x, sigma)`` that annealed Langevin wants."""

        def s(x: Tensor, sigma: float) -> Tensor:
            return self.forward(x, sigma)

        return s
