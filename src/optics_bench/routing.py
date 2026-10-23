"""Two bend families and a keep-out. CMA-ES searches one family at a time.

The arc family is the day-2 quarter-circle. The detour family stays in the
outer lane and cannot redraw that quarter-circle. A clean device walls off
the lane. A blocked device cuts the quarter-circle. Neither family is told
the class except through the pixels it is forbidden to fill.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from optics_bench.baselines import Trace, _record, load_baselines, make_cma
from optics_bench.cost import CostMeter
from optics_bench.devices import make_model
from optics_bench.judge import hinge_loss
from optics_bench.parametric import _boundary_supports, _ribbon, bend_density, waveguide_px
from optics_bench.protocol import ROOT, judge_cost, wavelengths_for
from optics_bench.simulate import install_solve_counter, simulate_s

ROUTING_PATH = ROOT / "configs" / "routing.yaml"


def load_routing(path: Path | None = None) -> dict[str, Any]:
    path = path or ROUTING_PATH
    with path.open() as handle:
        task = yaml.safe_load(handle)
    if list(task["right_family"]) != ["clean", "blocked"]:
        raise ValueError("routing classes are locked")
    if task["right_family"]["clean"] != "arc" or task["right_family"]["blocked"] != "detour":
        raise ValueError("right family for each class is locked")
    if int(task["budget"]) != 80:
        raise ValueError("routing budget is locked at 80")
    return task


def detour_density(model, params, wg_px: int) -> np.ndarray:
    """Outer-lane path from the left port to the bottom port.

    Parameters only nudge the lane. They do not move the path onto the
    quarter-circle. p = 0.5 is a filleted perimeter.
    """
    nx, ny = (int(v) for v in model.design_variable_shape)
    p = np.clip(np.asarray(params, dtype=float).reshape(-1), 0.0, 1.0)
    if p.size != 8:
        raise ValueError("detour dimension is 8")
    lane = float(wg_px) * (0.40 + 0.20 * float(p[0]))
    radius = float(wg_px) * (0.40 + 0.40 * float(p[1]))
    width = float(wg_px) * (0.85 + 0.30 * float(p[2]))
    top = (ny - lane) - (float(p[3]) - 0.5) * 0.30 * float(wg_px)
    right = (nx - lane) - (float(p[4]) - 0.5) * 0.30 * float(wg_px)
    bottom = lane + (float(p[5]) - 0.5) * 0.30 * float(wg_px)
    left = lane + (float(p[6]) - 0.5) * 0.30 * float(wg_px)
    # p7 tapers the far corner. It stays inside the lane.
    radius *= 0.85 + 0.30 * float(p[7])
    room = min(top - ny / 2.0, right - nx / 2.0, right - left, top - bottom) - 2.0
    radius = float(np.clip(radius, 1.0, max(1.0, room)))

    def arc(cx, cy, r, a0, a1):
        angles = np.linspace(a0, a1, 8)
        return np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)

    points = [np.array([left, ny / 2.0])]
    points.append(np.array([left, top - radius]))
    points.extend(arc(left + radius, top - radius, radius, np.pi, 0.5 * np.pi))
    points.append(np.array([right - radius, top]))
    points.extend(arc(right - radius, top - radius, radius, 0.5 * np.pi, 0.0))
    points.append(np.array([right, bottom + radius]))
    points.extend(arc(right - radius, bottom + radius, radius, 0.0, -0.5 * np.pi))
    points.append(np.array([nx / 2.0, bottom]))
    polyline = np.vstack(points)
    widths = np.full(len(polyline), width)
    solid = _ribbon((nx, ny), polyline, widths)
    left_pad, bottom_pad = _boundary_supports(model)
    solid[: int(wg_px), left_pad] = True
    solid[bottom_pad, : int(wg_px)] = True
    return solid.astype(float)


def _port_pads(model, wg_px: int) -> np.ndarray:
    nx, ny = (int(v) for v in model.design_variable_shape)
    pads = np.zeros((nx, ny), dtype=bool)
    left, bottom = _boundary_supports(model)
    pads[: int(wg_px), left] = True
    pads[bottom, : int(wg_px)] = True
    return pads


def keepout_mask(kind: str, model, wg_px: int, task: dict[str, Any]) -> np.ndarray:
    """Boolean mask of pixels the design is not allowed to fill."""
    nx, ny = (int(v) for v in model.design_variable_shape)
    mask = np.zeros((nx, ny), dtype=bool)
    if kind == "clean":
        band = int(round(float(task["wall_top_band_wg"]) * wg_px))
        x0 = int(round(float(task["wall_x_frac"][0]) * nx))
        x1 = int(round(float(task["wall_x_frac"][1]) * nx))
        mask[x0:x1, ny - band : ny] = True
    elif kind == "blocked":
        r0 = float(task["sector_r_frac"][0]) * float(min(nx, ny))
        r1 = float(task["sector_r_frac"][1]) * float(min(nx, ny))
        a0, a1 = (float(v) for v in task["sector_deg"])
        xs = np.arange(nx) + 0.5
        ys = np.arange(ny) + 0.5
        x, y = np.meshgrid(xs, ys, indexing="ij")
        radius = np.hypot(x, y)
        angle = np.degrees(np.arctan2(y, x))
        mask = (radius >= r0) & (radius <= r1) & (angle >= a0) & (angle <= a1)
        frame = int(round(float(task["sector_frame_wg"]) * wg_px))
        mask[nx - frame :, :] = False
        mask[:, ny - frame :] = False
    else:
        raise ValueError(f"unknown keep-out: {kind}")
    mask[_port_pads(model, wg_px)] = False
    return mask


def render_family(family: str, model, params, wg_px: int, keepout: np.ndarray) -> np.ndarray:
    if family == "arc":
        design = bend_density(model, params, wg_px)
    elif family == "detour":
        design = detour_density(model, params, wg_px)
    else:
        raise ValueError(f"unknown family: {family}")
    out = np.array(design, dtype=float, copy=True)
    out[np.asarray(keepout, dtype=bool)] = 0.0
    return out


def mask_rule_family(model, wg_px: int, keepout: np.ndarray) -> str:
    """Pick the family whose nominal geometry misses the keep-out."""
    nominal = np.full(8, 0.5)
    empty = np.zeros_like(keepout, dtype=bool)
    arc_hits = np.any((bend_density(model, nominal, wg_px) >= 0.5) & keepout)
    detour_hits = np.any((detour_density(model, nominal, wg_px) >= 0.5) & keepout)
    if arc_hits and not detour_hits:
        return "detour"
    if detour_hits and not arc_hits:
        return "arc"
    # Both hit, or neither does. Prefer the nominal path that loses fewer pixels.
    arc_loss = int(np.sum((bend_density(model, nominal, wg_px) >= 0.5) & keepout))
    detour_loss = int(np.sum((detour_density(model, nominal, wg_px) >= 0.5) & keepout))
    del empty
    return "arc" if arc_loss <= detour_loss else "detour"


def _np(value):
    value = getattr(value, "_value", value)
    return np.asarray(value)


def _cma_generations(
    trace: Trace,
    model,
    protocol,
    meter: CostMeter,
    config,
    offset: float,
    budget: float,
    family: str,
    keepout: np.ndarray,
    strategy,
    max_generations: int | None,
    finalize: bool,
) -> None:
    pop = int(config["cma"]["popsize"])
    wg_px = waveguide_px(protocol, "waveguide_bend")
    generations = 0
    while True:
        if max_generations is not None and generations >= max_generations:
            break
        spent = meter.C
        fidelity = "low" if spent < budget / 2.0 else "high"
        waves = wavelengths_for(protocol, fidelity, offset)
        if spent + pop * len(waves) > budget:
            break
        points = list(strategy.ask())
        if spent + len(points) * len(waves) > budget:
            trace.notes.append("cma_ask_does_not_fit")
            break
        fitness = []
        succeeded = False
        for point in points:
            vector = np.asarray(point, dtype=float)
            design = render_family(family, model, vector, wg_px, keepout)
            meter.mode = "forward"
            sparams = simulate_s(model, design, waves)
            _record(
                trace,
                "waveguide_bend",
                model,
                protocol,
                meter,
                sparams,
                design,
                waves,
                vector,
                offset,
            )
            loss = float(_np(hinge_loss("waveguide_bend", _np(sparams), protocol)))
            fitness.append(loss if np.isfinite(loss) else 1.0e6)
            if trace.success:
                succeeded = True
                break
        trace.notes.append(f"{family}_{fidelity}_n{len(fitness)}_C{meter.C}")
        generations += 1
        if succeeded:
            trace.notes.append("cma_stop_success")
            return
        strategy.tell(points, fitness)
    if not finalize or trace.success or trace.low_latent is None or trace.low_judged_at_high:
        return
    if meter.C + judge_cost(protocol, offset) > budget:
        trace.notes.append("final_judge_skipped")
        return
    waves = wavelengths_for(protocol, "high", offset)
    design = render_family(family, model, trace.low_latent, wg_px, keepout)
    meter.mode = "forward"
    sparams = simulate_s(model, design, waves)
    _record(
        trace,
        "waveguide_bend",
        model,
        protocol,
        meter,
        sparams,
        design,
        waves,
        trace.low_latent,
        offset,
    )


def _best_low_vector(trace: Trace, start: int):
    """Best low-fidelity parameter vector recorded at or after `start`.

    The vector itself is not on the curve. Callers pass the latent through
    the trace by keeping the proxy fields, so this reads the trace proxy only
    when the improvement happened inside the window.
    """
    window = trace.curve[start:]
    lows = [point for point in window if not point["official"]]
    if not lows:
        return None
    return max(point["fom"] for point in lows)


def run_routing(payload: dict[str, Any]) -> dict[str, Any]:
    """One class, one method, one seed. The keep-out is rebuilt from the class."""
    import os

    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("OMP_NUM_THREADS is not 1")
    from optics_bench.protocol import load_protocol

    protocol = load_protocol()
    config = load_baselines()
    task = load_routing()
    meter = CostMeter()
    install_solve_counter(meter)
    trace = Trace()
    try:
        offset = float(payload["wavelength_offset_nm"])
        model = make_model(
            "waveguide_bend",
            protocol,
            fidelity="low",
            region_nm=tuple(payload["region_nm"]),
            wavelength_offset_nm=offset,
        )
        wg_px = waveguide_px(protocol, "waveguide_bend")
        keepout = keepout_mask(payload["class_name"], model, wg_px, task)
        budget = float(payload["budget"])
        seed = int(payload["seed"])
        method = payload["method"]
        trace.notes.append(f"class_{payload['class_name']}")
        trace.notes.append(f"mask_rule_{mask_rule_family(model, wg_px, keepout)}")
        if method == "always_arc":
            _cma_generations(
                trace, model, protocol, meter, config, offset, budget,
                "arc", keepout, make_cma(seed, config), None, True,
            )
        elif method == "always_detour":
            _cma_generations(
                trace, model, protocol, meter, config, offset, budget,
                "detour", keepout, make_cma(seed, config), None, True,
            )
        elif method == "probe_commit":
            scores = {}
            states = {}
            for family in ("arc", "detour"):
                start = len(trace.curve)
                strategy = make_cma(seed, config)
                _cma_generations(
                    trace, model, protocol, meter, config, offset, budget,
                    family, keepout, strategy, 1, False,
                )
                window = [point["fom"] for point in trace.curve[start:] if not point["official"]]
                scores[family] = max(window) if window else -1.0e9
                states[family] = strategy
                if trace.success:
                    break
            if not trace.success:
                winner = max(scores, key=scores.get)
                trace.notes.append(
                    f"probe_arc_{scores['arc']:.4f}_detour_{scores['detour']:.4f}_chose_{winner}"
                )
                _cma_generations(
                    trace, model, protocol, meter, config, offset, budget,
                    winner, keepout, states[winner], None, True,
                )
        else:
            raise ValueError(f"unknown routing method: {method}")
        if meter.C > budget + 1e-6:
            raise RuntimeError(f"budget exceeded: C={meter.C}")
        result = trace.result(payload, meter)
        result["class_name"] = payload["class_name"]
        design, latent = trace.design, trace.latent
    except Exception:
        try:
            result = trace.result(payload, meter)
        except Exception:
            result = {
                "device": "waveguide_bend",
                "spec_id": payload["spec_id"],
                "method": payload["method"],
                "budget": int(payload["budget"]),
                "seed": int(payload["seed"]),
                "region_nm": list(payload["region_nm"]),
                "wavelength_offset_nm": float(payload["wavelength_offset_nm"]),
                "success": False,
                "C_success": None,
                "best_official_fom": None,
                "best_proxy_fom": None,
                "C_spent": float(meter.C),
                "em_pass": False,
                "length_pass": False,
                "notes": [],
                "curve": [],
                "error": "",
            }
        result["error"] = traceback.format_exc()
        result["class_name"] = payload.get("class_name")
        design, latent = trace.design, trace.latent
    out = Path(payload["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    temporary.replace(out)
    if design is not None:
        np.savez_compressed(
            out.with_suffix(".npz"),
            design=np.asarray(design, dtype=float),
            latent=np.asarray([] if latent is None else latent, dtype=float),
        )
    return result
