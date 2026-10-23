"""Load the locked protocol. Callers do not edit these numbers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "protocol.yaml"


def load_protocol(path: Path | None = None) -> dict[str, Any]:
    path = path or PROTOCOL_PATH
    with path.open() as f:
        protocol = yaml.safe_load(f)
    if protocol["cost"]["alpha"] != 1.0:
        raise ValueError("alpha is locked at 1")
    return protocol


def db_to_power(db: float) -> float:
    """Convert a power-dB threshold to a linear power |S|^2."""
    return 10.0 ** (db / 10.0)


def wavelengths_for(
    protocol: dict[str, Any], fidelity: str, offset_nm: float = 0.0
) -> list[float]:
    cost = protocol["cost"]
    if fidelity == "low":
        base = cost["low_wavelengths_nm"]
    elif fidelity == "high":
        base = cost["high_wavelengths_nm"]
    else:
        raise ValueError(f"unknown fidelity: {fidelity}")
    return [float(w) + float(offset_nm) for w in base]


def step_cost(protocol: dict[str, Any], fidelity: str, offset_nm: float = 0.0) -> int:
    """Forward plus adjoint, one solve per wavelength."""
    return 2 * len(wavelengths_for(protocol, fidelity, offset_nm))


def judge_cost(protocol: dict[str, Any], offset_nm: float = 0.0) -> int:
    """A spec check is forwards only, at the high-fidelity wavelengths."""
    return len(wavelengths_for(protocol, "high", offset_nm))
