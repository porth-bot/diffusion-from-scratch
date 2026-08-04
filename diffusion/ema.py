"""Exponential moving average of weights: the one training trick every
diffusion paper uses and none of them ablates.

Adam on the DSM objective never converges to a point. Each step draws fresh
``x0`` and fresh ``sigma``, so the gradient is a noisy estimate of the true
one and the iterate ends up executing a random walk in a neighbourhood of the
optimum whose radius is set by the learning rate and the gradient noise, not
by how much longer you train. EMA averages that walk away:

    theta_bar_t = d * theta_bar_{t-1} + (1 - d) * theta_t.                 (1)

What (1) buys, exactly. Write ``theta_t = theta* + xi_t`` with ``xi_t`` i.i.d.,
mean zero, per-coordinate variance ``v`` -- the stationary picture, once the
iterate is jittering rather than travelling. Unrolling (1) from far in the
past gives ``theta_bar = theta* + (1-d) sum_{k>=0} d^k xi_{t-k}``, so

    Var[theta_bar] = v (1-d)^2 sum d^{2k} = v (1 - d) / (1 + d).           (2)

At ``d = 0.999`` that is a 2000x variance reduction, i.e. a ~45x smaller
parameter-space wobble. ``tests/test_ema.py`` measures (2) directly rather
than trusting it, which matters because the i.i.d. assumption behind it is
false in a real run -- consecutive Adam iterates are correlated, so (2) is
the optimistic end of what EMA can do, and the *measured* benefit on a real
training run is the number worth reporting.

What (1) is supposed to cost. The average is over the last ``~1/(1-d)``
steps, so it should lag: while the weights are still genuinely travelling
toward the optimum, an average of them is an average of worse parameters,
and the EMA should lose until the iterate settles. That is the standard
caution, and ``experiments/ema.py`` was written to measure the crossover it
predicts.

It did not find one. Over 20k steps on the ring target, five seeds per decay,
the EMA arm leads from the *second* evaluation (step 500) at every decay
tried, up to ``d = 0.9999`` whose nominal window is half the whole run. The
final score error drops 4-6x (relative MSE 0.091 raw against 0.014 at
``d = 0.999``) and the seed-to-seed spread drops 17x. The lag is real in
principle -- at step 1 the shadow is still mostly initialization and the EMA
is genuinely worse -- but on this problem it is paid off within a few hundred
steps, not a few thousand. See the README for the caveat that limits how far
that generalizes: the warmup correction below caps the effective decay, so
the three largest nominal decays were running nearly the same schedule.

The warmup correction ``d_t = min(d, (1 + t) / (10 + t))`` (the form used in
most diffusion codebases) exists at the other end: with a constant
``d = 0.999`` the shadow is still 37% initialization after 1000 steps,
because nothing ever removes it. The correction starts the effective decay
near 0.1 and raises it, so the initialization is flushed out in a few steps.
It is on by default here, and ``warmup=False`` shows what it fixes.

The correction has a consequence that is easy to miss and that the ablation
ran into: ``(1 + t)/(10 + t)`` reaches ``d`` only at ``t = (10d - 1)/(1 - d)``
-- step 8,990 for ``d = 0.999`` and step 17,990 for ``d = 0.9995``, and never
inside 20k steps for ``d = 0.9999``. In a short run the *nominal* decay is
therefore not the decay in force for most of it, which is why those three
arms land on top of each other.

Buffers are not averaged. ``SinusoidalEmbedding.freqs`` is fixed by design,
not learned, so there is nothing to average and copying it would be a no-op
at best. Only ``named_parameters`` are shadowed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import Tensor, nn


class EMA:
    """Shadow copy of a module's parameters, updated by (1) after each step.

    Parameters
    ----------
    module : nn.Module
        Shadowed at construction; the shadow starts equal to the current
        parameters (not at zero -- a zero-initialized EMA with bias
        correction is the Adam convention, not this one).
    decay : float
        ``d`` in (1), in [0, 1). ``0`` makes the EMA track the raw weights
        exactly, which is the useful degenerate case for tests.
    warmup : bool
        Apply ``d_t = min(d, (1 + t)/(10 + t))``, which flushes the
        initialization out of the average in the first few dozen steps.

    Examples
    --------
    An EMA of a parameter held at a constant value converges to that value,
    and the geometric series says exactly how fast: after ``t`` updates the
    error is ``d^t`` times the initial gap.

    >>> import torch
    >>> from torch import nn
    >>> layer = nn.Linear(1, 1, bias=False)
    >>> with torch.no_grad():
    ...     _ = layer.weight.fill_(0.0)
    >>> ema = EMA(layer, decay=0.9, warmup=False)      # shadow starts at 0
    >>> with torch.no_grad():
    ...     _ = layer.weight.fill_(1.0)                # the "optimum"
    >>> for _ in range(3):
    ...     ema.update(layer)
    >>> round(float(ema.shadow["weight"]), 6)          # 1 - 0.9^3
    0.271

    The shadow can be swapped in for evaluation and is guaranteed to be
    swapped back out, exception or not:

    >>> with ema.swapped(layer):
    ...     round(float(layer.weight.detach()), 6)
    0.271
    >>> float(layer.weight.detach())
    1.0
    """

    def __init__(self, module: nn.Module, decay: float = 0.999, warmup: bool = True):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.n_updates = 0
        self.shadow: dict[str, Tensor] = {
            name: p.detach().clone() for name, p in module.named_parameters()
        }

    def current_decay(self) -> float:
        """The decay actually applied at the next update.

        Without warmup this is just ``decay``. With it, the effective decay
        ramps from 1/10 upward, so the shadow is dominated by recent weights
        while the average is still short.
        """
        if not self.warmup:
            return self.decay
        t = self.n_updates
        return min(self.decay, (1.0 + t) / (10.0 + t))

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        """One application of (1), in place."""
        d = self.current_decay()
        for name, p in module.named_parameters():
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)
        self.n_updates += 1

    @torch.no_grad()
    def copy_to(self, module: nn.Module) -> None:
        """Overwrite the module's parameters with the averaged ones."""
        for name, p in module.named_parameters():
            p.copy_(self.shadow[name])

    @contextmanager
    def swapped(self, module: nn.Module) -> Iterator[nn.Module]:
        """Evaluate under the averaged weights, then restore the raw ones.

        The restore runs in a ``finally``: an exception inside the block must
        not leave training continuing from the *averaged* parameters, which
        would be a silent corruption of the run rather than a crash.
        """
        backup = {name: p.detach().clone() for name, p in module.named_parameters()}
        self.copy_to(module)
        try:
            yield module
        finally:
            with torch.no_grad():
                for name, p in module.named_parameters():
                    p.copy_(backup[name])

    def stationary_variance_ratio(self) -> float:
        """``(1 - d) / (1 + d)``: the factor in (2), for the configured decay.

        The variance of the averaged parameters relative to the raw ones,
        under the i.i.d.-jitter model. Real iterates are positively
        correlated, so treat this as a bound on the benefit, not a prediction
        of it.
        """
        return (1.0 - self.decay) / (1.0 + self.decay)
