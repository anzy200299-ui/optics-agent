"""One-wavelength 2D FDFD solves, counted on the main thread.

The library simulate() fans work out to a thread pool. Autograd does not
follow that pool, so adjoint steps go through ceviche.fdfd_ez directly and
reuse the library's ports and scattering-amplitude code.
"""

from __future__ import annotations

from typing import Sequence

import autograd.numpy as npa
import ceviche
import ceviche.primitives as ceviche_primitives
import ceviche.solvers as ceviche_solvers
import numpy as np
from ceviche_challenges.model_base import _wavelengths_nm_to_omegas
from ceviche_challenges.scattering import calculate_amplitudes

from optics_bench.cost import CostMeter

class _MeterHolder:
    """Workers rebind this. The autograd primitive stays patched once."""

    def __init__(self):
        self.meter: CostMeter | None = None

    def note_solve(self) -> None:
        if self.meter is None:
            raise RuntimeError("solve counter has no meter")
        self.meter.note_solve()


_HOLDER = _MeterHolder()
_PATCHED = False


def install_solve_counter(meter: CostMeter) -> None:
    """Count every sparse Maxwell solve, including the adjoint vjp.

    fdfd.solve is not called again on the reverse pass. The reverse pass
    calls sp_solve, and sp_solve calls solve_linear. Counting solve_linear
    keeps the autograd primitive intact.
    """
    global _PATCHED
    _HOLDER.meter = meter
    if _PATCHED:
        return
    # pyMKL imports even when the Intel library is absent, then crashes
    # inside pardiso. Scipy is the solver this machine actually runs.
    ceviche_solvers.HAS_MKL = False
    original = ceviche_solvers.solve_linear

    def counted_solve(A, b, iterative_method=False):
        _HOLDER.note_solve()
        return original(A, b, iterative_method=iterative_method)

    # primitives binds the function with `from .solvers import solve_linear`.
    ceviche_solvers.solve_linear = counted_solve
    ceviche_primitives.solve_linear = counted_solve
    _PATCHED = True


def simulate_s(
    model,
    design,
    wavelengths_nm: Sequence[float],
    excite_port_idxs: Sequence[int] = (0,),
):
    """Complex S columns, shape (n_wavelength, n_excited, n_ports)."""
    eps = model.epsilon_r(design)
    eps_bg = model.epsilon_r_bg()
    dl = model.dl
    pml = model.pml_width
    omegas = _wavelengths_nm_to_omegas(np.asarray(wavelengths_nm, dtype=float))
    rows = []
    for omega in omegas:
        excited = []
        for port_idx in excite_port_idxs:
            sim = ceviche.fdfd_ez(omega, dl, eps_bg, [pml, pml])
            sim.eps_r = eps
            source = model.ports[port_idx].source_fdfd(omega, dl, eps_bg)
            hx, hy, ez = sim.solve(source)
            outgoing = []
            incoming = None
            for j, port in enumerate(model.ports):
                forward, backward = calculate_amplitudes(
                    omega, dl, port, ez, hy, hx, eps_bg
                )
                if j == port_idx:
                    incoming = forward
                outgoing.append(backward)
            excited.append([value / incoming for value in outgoing])
        rows.append(excited)
    return npa.stack([npa.stack(excited) for excited in rows])
