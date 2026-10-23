"""Spec list for the day-2 sweep. The YAML is the lock."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from optics_bench.protocol import ROOT

SPECS_PATH = ROOT / "configs" / "specs.yaml"


def load_specs(path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    path = path or SPECS_PATH
    with path.open() as handle:
        specs = yaml.safe_load(handle)
    for device, rows in specs.items():
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate spec id for {device}")
        nominal = [row for row in rows if row.get("nominal")]
        if len(nominal) != 1:
            raise ValueError(f"{device} needs exactly one nominal spec")
    return specs


def nominal_specs(specs: dict[str, list[dict[str, Any]]]):
    for device, rows in specs.items():
        for row in rows:
            if row.get("nominal"):
                yield device, row
