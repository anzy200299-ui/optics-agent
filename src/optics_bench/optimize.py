"""Fixed-schedule adjoint steps. One call is one tool action, not a full budget."""

from __future__ import annotations

from typing import Any

import autograd.numpy as npa
import numpy as np
from autograd import make_vjp

from optics_bench.cost import CostMeter
from optics_bench.judge import hinge_loss
from optics_bench.parameterize import density_from_latent
from optics_bench.protocol import wavelengths_for
from optics_bench.simulate import simulate_s


def zeros_latent(model) -> np.ndarray:
    return np.zeros(model.design_variable_shape, dtype=float)


def fresh_adam(latent: np.ndarray) -> dict[str, Any]:
    return {
        "m": np.zeros_like(latent, dtype=float),
        "v": np.zeros_like(latent, dtype=float),
        "t": 0,
    }


def _detach(value):
    value = getattr(value, "_value", value)
    return np.asarray(value)


def adjoint_step(
    device: str,
    model,
    latent: np.ndarray,
    protocol: dict[str, Any],
    meter: CostMeter,
    fidelity: str,
    radius_px: int | None = None,
    adam: dict[str, Any] | None = None,
    wavelength_offset_nm: float = 0.0,
):
    """Evaluate `latent`, then return a closure that applies one Adam update.

    The closure runs the adjoint solve. Callers that already meet the spec
    skip it, so a successful forward is not charged an unused adjoint.
    """
    para = protocol["parameterization"]
    opt = protocol["optimizer"]
    wavelengths = wavelengths_for(protocol, fidelity, wavelength_offset_nm)
    radius = int(para["filter_radius_px"] if radius_px is None else radius_px)
    beta = float(para["beta"])
    eta = float(para["eta"])
    latent = np.asarray(latent, dtype=float)
    if adam is None:
        adam = fresh_adam(latent)
    box: dict[str, Any] = {}

    def loss_fn(latent_var):
        design = density_from_latent(latent_var, radius, beta, eta)
        sparams = simulate_s(model, design, wavelengths)
        box["s"] = sparams
        box["design"] = design
        return hinge_loss(device, sparams, protocol)

    meter.mode = "forward"
    vjp, loss = make_vjp(loss_fn)(npa.array(latent))
    sparams = np.array(_detach(box["s"]))
    design = np.array(_detach(box["design"]), dtype=float)
    loss_value = float(_detach(loss))
    b1 = float(opt["adam_beta1"])
    b2 = float(opt["adam_beta2"])
    lr = float(opt["adam_lr"])
    eps = float(opt["adam_eps"])

    def apply_update():
        meter.mode = "adjoint"
        grad = np.asarray(_detach(vjp(np.array(1.0))), dtype=float)
        meter.mode = "forward"
        if not np.isfinite(grad).all():
            raise FloatingPointError("non-finite adjoint gradient")
        adam["t"] = int(adam["t"]) + 1
        step = adam["t"]
        adam["m"] = b1 * adam["m"] + (1.0 - b1) * grad
        adam["v"] = b2 * adam["v"] + (1.0 - b2) * (grad * grad)
        mhat = adam["m"] / (1.0 - b1**step)
        vhat = adam["v"] / (1.0 - b2**step)
        updated = latent - lr * mhat / (np.sqrt(vhat) + eps)
        return updated, adam

    return sparams, design, loss_value, wavelengths, apply_update


def run_adjoint(
    device: str,
    model,
    latent: np.ndarray,
    protocol: dict[str, Any],
    meter: CostMeter,
    iterations: int,
    fidelity: str,
) -> tuple[np.ndarray, float]:
    """Adam on the latent density. Returns the updated latent and the hinge."""
    adam = fresh_adam(np.asarray(latent, dtype=float))
    last_loss = None
    for _ in range(iterations):
        _sparams, _design, last_loss, _waves, apply_update = adjoint_step(
            device, model, latent, protocol, meter, fidelity, adam=adam
        )
        latent, adam = apply_update()
    return latent, last_loss
