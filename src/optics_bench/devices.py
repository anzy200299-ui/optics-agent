"""Build the two locked devices. Prefab defaults are intentionally not used."""

from __future__ import annotations

from typing import Any

from ceviche_challenges import units as u
from ceviche_challenges.beam_splitter.model import BeamSplitterModel
from ceviche_challenges.beam_splitter.spec import BeamSplitterSpec
from ceviche_challenges.params import CevicheSimParams
from ceviche_challenges.waveguide_bend.model import WaveguideBendModel
from ceviche_challenges.waveguide_bend.spec import WaveguideBendSpec

from optics_bench.protocol import wavelengths_for


def _nm(value: float):
    return float(value) * u.nm


def make_model(
    device: str,
    protocol: dict[str, Any],
    fidelity: str = "high",
    region_nm: tuple[float, float] | None = None,
    wavelength_offset_nm: float = 0.0,
):
    """Return a ceviche-challenges model.

    `fidelity` and `wavelength_offset_nm` only set the stored wavelength list.
    Solves pass wavelengths explicitly. `region_nm` overrides the design-region
    size for a spec; permittivity, waveguide width, and grid stay locked.
    """
    wavelengths = wavelengths_for(protocol, fidelity, wavelength_offset_nm)
    params = CevicheSimParams(
        resolution=protocol["resolution_nm"] * u.nm,
        wavelengths=u.Array(wavelengths, u.nm),
    )
    if device == "waveguide_bend":
        raw = dict(protocol["devices"]["waveguide_bend"])
        if region_nm is not None:
            raw["variable_region_nm"] = list(region_nm)
        spec = WaveguideBendSpec(
            wg_width=_nm(raw["wg_width_nm"]),
            wg_length=_nm(raw["wg_length_nm"]),
            wg_mode_padding=_nm(raw["wg_mode_padding_nm"]),
            padding=_nm(raw["padding_nm"]),
            port_pml_offset=_nm(raw["port_pml_offset_nm"]),
            variable_region_size=(
                _nm(raw["variable_region_nm"][0]),
                _nm(raw["variable_region_nm"][1]),
            ),
            cladding_permittivity=float(raw["cladding_permittivity"]),
            slab_permittivity=float(raw["slab_permittivity"]),
            input_monitor_offset=_nm(raw["input_monitor_offset_nm"]),
            pml_width=int(raw["pml_width"]),
        )
        return WaveguideBendModel(params, spec)
    if device == "beam_splitter":
        raw = dict(protocol["devices"]["beam_splitter"])
        if region_nm is not None:
            raw["variable_region_nm"] = list(region_nm)
        spec = BeamSplitterSpec(
            wg_width=_nm(raw["wg_width_nm"]),
            wg_length=_nm(raw["wg_length_nm"]),
            wg_separation=_nm(raw["wg_separation_nm"]),
            wg_mode_padding=_nm(raw["wg_mode_padding_nm"]),
            port_pml_offset=_nm(raw["port_pml_offset_nm"]),
            variable_region_size=(
                _nm(raw["variable_region_nm"][0]),
                _nm(raw["variable_region_nm"][1]),
            ),
            cladding_permittivity=float(raw["cladding_permittivity"]),
            slab_permittivity=float(raw["slab_permittivity"]),
            input_monitor_offset=_nm(raw["input_monitor_offset_nm"]),
            design_symmetry=raw["design_symmetry"],
            pml_width=int(raw["pml_width"]),
        )
        return BeamSplitterModel(params, spec)
    raise ValueError(f"unknown device: {device}")
