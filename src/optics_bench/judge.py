"""Hard success and the reported figure of merit. The model does not score itself."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from optics_bench.protocol import db_to_power


def powers(sparams) -> np.ndarray:
    """|S|^2 with the same shape as the complex S array."""
    s = np.asarray(sparams)
    return np.real(s * np.conj(s))


def _margin(power: float, lower: float, upper: float) -> float:
    """invrs-gym orthotope margin. Positive means the bound is met."""
    width = upper - lower
    if lower == 0.0:
        return (upper - power) / width
    if upper == 1.0:
        return (power - lower) / width
    return min(power - lower, upper - power) / width


def constraint_margins(device: str, sparams, protocol: dict[str, Any]) -> list[float]:
    """One margin per wavelength and per Table I entry."""
    p = powers(sparams)
    # Shape (wavelength, excited, port). We excite port 0 only.
    p = p[:, 0, :]
    thresholds = protocol["thresholds_db"]
    margins = []
    if device == "waveguide_bend":
        spec = thresholds["bend"]
        s21_floor = db_to_power(spec["s21_min_db"])
        s11_ceil = db_to_power(spec["s11_max_db"])
        for row in p:
            margins.append(_margin(float(row[1]), s21_floor, 1.0))
            margins.append(_margin(float(row[0]), 0.0, s11_ceil))
        return margins
    if device == "beam_splitter":
        spec = thresholds["beam_splitter"]
        through = db_to_power(spec["s21_min_db"])
        reflection = db_to_power(spec["s11_max_db"])
        isolated = db_to_power(spec["s41_max_db"])
        for row in p:
            margins.append(_margin(float(row[0]), 0.0, reflection))
            margins.append(_margin(float(row[1]), through, 1.0))
            margins.append(_margin(float(row[2]), through, 1.0))
            margins.append(_margin(float(row[3]), 0.0, isolated))
        return margins
    raise ValueError(f"unknown device: {device}")


def fom(device: str, sparams, protocol: dict[str, Any]) -> float:
    """Worst Table I margin. This is the number the main table reports."""
    return float(min(constraint_margins(device, sparams, protocol)))


def hinge_loss(device: str, sparams, protocol: dict[str, Any]):
    """Sum of violated margins. Zero once every Table I entry is met.

    This is the optimizer loss. It is not the reported score.
    """
    import autograd.numpy as npa

    # Recompute with autograd arrays so the hinge stays on the trace.
    power = npa.real(sparams * npa.conj(sparams))[:, 0, :]
    thresholds = protocol["thresholds_db"]
    terms = []
    if device == "waveguide_bend":
        spec = thresholds["bend"]
        s21_floor = db_to_power(spec["s21_min_db"])
        s11_ceil = db_to_power(spec["s11_max_db"])
        for i in range(power.shape[0]):
            terms.append(npa.maximum(0.0, -_margin_ag(power[i, 1], s21_floor, 1.0)))
            terms.append(npa.maximum(0.0, -_margin_ag(power[i, 0], 0.0, s11_ceil)))
    elif device == "beam_splitter":
        spec = thresholds["beam_splitter"]
        through = db_to_power(spec["s21_min_db"])
        reflection = db_to_power(spec["s11_max_db"])
        isolated = db_to_power(spec["s41_max_db"])
        for i in range(power.shape[0]):
            terms.append(npa.maximum(0.0, -_margin_ag(power[i, 0], 0.0, reflection)))
            terms.append(npa.maximum(0.0, -_margin_ag(power[i, 1], through, 1.0)))
            terms.append(npa.maximum(0.0, -_margin_ag(power[i, 2], through, 1.0)))
            terms.append(npa.maximum(0.0, -_margin_ag(power[i, 3], 0.0, isolated)))
    else:
        raise ValueError(f"unknown device: {device}")
    return sum(terms)


def _margin_ag(power, lower: float, upper: float):
    import autograd.numpy as npa

    width = upper - lower
    if lower == 0.0:
        return (upper - power) / width
    if upper == 1.0:
        return (power - lower) / width
    return npa.minimum(power - lower, upper - power) / width


def min_medial_px(protocol: dict[str, Any]) -> float:
    """Half the minimum feature, in pixels. 150 nm / 50 nm / 2 = 1.5."""
    return float(protocol["parameterization"]["min_feature_nm"]) / float(
        protocol["resolution_nm"]
    ) / 2.0


def _thin_medial(mask: np.ndarray, limit: float, score_here: np.ndarray) -> bool:
    """True when a medial pixel inside the design is narrower than `limit`."""
    if not np.any(mask):
        return False
    dist = ndimage.distance_transform_edt(mask)
    local_max = dist >= ndimage.maximum_filter(dist, size=3)
    thin = local_max & (dist > 0) & (dist < limit) & score_here & mask
    return bool(np.any(thin))


def length_scale_ok(model, design, protocol: dict[str, Any]) -> bool:
    """Minimum solid and void width on the medial axis.

    Distance is measured on the full device so the access waveguides count.
    Only medial pixels inside the design region are scored.
    """
    density = np.asarray(model.density(np.asarray(design, dtype=float)))
    solid = density >= 0.5
    x0, y0, x1, y1 = model.design_region_coords
    score_here = np.zeros(solid.shape, dtype=bool)
    score_here[x0:x1, y0:y1] = True
    limit = min_medial_px(protocol)
    solid_bad = _thin_medial(solid, limit, score_here)
    void_bad = _thin_medial(~solid, limit, score_here)
    return not (solid_bad or void_bad)


def hard_success(device: str, model, design, sparams, protocol: dict[str, Any]) -> bool:
    electromagnetic = all(m >= 0.0 for m in constraint_margins(device, sparams, protocol))
    return bool(electromagnetic and length_scale_ok(model, design, protocol))


def transmission_report(device: str, sparams, wavelengths_nm) -> list[dict[str, float]]:
    p = powers(sparams)[:, 0, :]
    rows = []
    for wavelength, row in zip(wavelengths_nm, p):
        entry = {"wavelength_nm": float(wavelength)}
        for port, value in enumerate(np.asarray(row)):
            entry[f"p{port + 1}"] = float(value)
        rows.append(entry)
    return rows
