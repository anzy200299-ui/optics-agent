"""Density-grid parameterization. Differentiable, no brush generator."""

from __future__ import annotations

import autograd.numpy as npa


def project(rho, beta: float, eta: float = 0.5):
    numerator = npa.tanh(beta * eta) + npa.tanh(beta * (rho - eta))
    denominator = npa.tanh(beta * eta) + npa.tanh(beta * (1.0 - eta))
    return numerator / denominator


def _edge_pad(rho):
    """One-cell edge pad. autograd's pad only differentiates constant mode."""
    vertical = npa.concatenate([rho[:1, :], rho, rho[-1:, :]], axis=0)
    return npa.concatenate([vertical[:, :1], vertical, vertical[:, -1:]], axis=1)


def neighbor_average(rho, radius_px: int):
    """Repeat a 4-neighbor average. Autograd can see every step."""
    out = rho
    for _ in range(int(radius_px)):
        padded = _edge_pad(out)
        center = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        out = (center + up + down + left + right) / 5.0
    return out


def density_from_latent(latent, radius_px: int, beta: float, eta: float = 0.5):
    rho = 1.0 / (1.0 + npa.exp(-latent))
    return project(neighbor_average(rho, radius_px), beta, eta)
