"""The advertised surface and the exposed surface must not drift apart.

pinn-from-scratch shipped with `features` listed as a core module in its
README while the package `__init__` never exported it, so `import pinn;
pinn.features` raised `AttributeError` on a fresh clone. This is that check,
written before the same thing can happen here.
"""

from __future__ import annotations

import diffusion


def test_everything_in_dunder_all_actually_resolves():
    for name in diffusion.__all__:
        assert hasattr(diffusion, name), f"diffusion.__all__ advertises {name!r}, which is missing"


def test_submodules_are_reachable_off_the_package():
    assert diffusion.targets.GaussianMixture is diffusion.GaussianMixture
    assert diffusion.langevin.annealed_langevin is diffusion.annealed_langevin
