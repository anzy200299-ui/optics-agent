"""Four classical controllers on one frozen budget.

Fixed adjoint, random restart, low-dimensional CMA-ES, and the heuristic.
Transmission and linewidth come from the solver and the judge. A controller
never reports its own score.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from optics_bench.actions import validate_action
from optics_bench.cost import CostMeter
from optics_bench.devices import make_model
from optics_bench.judge import (
    fom,
    hinge_loss,
    length_scale_ok,
    transmission_report,
)
from optics_bench.optimize import adjoint_step, fresh_adam
from optics_bench.parameterize import density_from_latent
from optics_bench.parametric import render, waveguide_px
from optics_bench.protocol import ROOT, judge_cost, step_cost, wavelengths_for
from optics_bench.simulate import install_solve_counter, simulate_s

BASELINES_PATH = ROOT / "configs" / "baselines.yaml"


def _locked(value, expected, name: str) -> None:
    if isinstance(expected, float):
        if abs(float(value) - expected) > 1e-12:
            raise ValueError(f"{name} is locked at {expected}")
        return
    if value != expected:
        raise ValueError(f"{name} is locked at {expected}")


def load_baselines(path: Path | None = None) -> dict[str, Any]:
    path = path or BASELINES_PATH
    with path.open() as handle:
        config = yaml.safe_load(handle)
    _locked(int(config["restart_k"]), 2, "restart_k")
    _locked(float(config["init_std"]), 0.1, "init_std")
    _locked(int(config["stagnation_steps"]), 3, "stagnation_steps")
    _locked(float(config["stagnation_eps"]), 0.001, "stagnation_eps")
    _locked(int(config["filter_radius_max_px"]), 4, "filter radius cap")
    if list(config["heuristic_iterations"]) != [5, 1]:
        raise ValueError("heuristic iterations are locked at [5, 1]")
    cma_cfg = config["cma"]
    _locked(int(cma_cfg["dimension"]), 8, "CMA dimension")
    _locked(int(cma_cfg["popsize"]), 4, "CMA popsize")
    _locked(float(cma_cfg["sigma0"]), 0.2, "CMA sigma0")
    _locked(float(cma_cfg["x0"]), 0.5, "CMA x0")
    if [float(v) for v in cma_cfg["bounds"]] != [0.0, 1.0]:
        raise ValueError("CMA bounds are locked at [0, 1]")
    _locked(float(config["saturation_fom"]), 0.02, "saturation_fom")
    if list(config["methods"]) != [
        "fixed_adjoint",
        "random_restart",
        "cmaes",
        "heuristic",
    ]:
        raise ValueError("baseline method list is locked")
    if list(config["phase_a_seeds"]) != [0] or list(config["phase_b_seeds"]) != [0, 1, 2]:
        raise ValueError("phase seeds are locked")
    return config


def _np(value):
    value = getattr(value, "_value", value)
    return np.asarray(value)


def _shape(model) -> tuple[int, int]:
    return tuple(int(v) for v in model.design_variable_shape)


def _is_official(waves, protocol, offset: float) -> bool:
    expected = wavelengths_for(protocol, "high", offset)
    got = [float(w) for w in waves]
    return len(got) == len(expected) and bool(np.allclose(got, expected, atol=1e-6))


def _initial_latent(model, seed: int, init_std: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.normal(0.0, float(init_std), size=_shape(model))


def project_latent(latent, radius: int, protocol) -> np.ndarray:
    para = protocol["parameterization"]
    design = density_from_latent(
        np.asarray(latent, dtype=float),
        int(radius),
        float(para["beta"]),
        float(para["eta"]),
    )
    return _np(design).astype(float)


class Trace:
    """One job's scored evaluations. The reported design is the best official one."""

    def __init__(self):
        self.curve: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.n_illegal = 0
        self.success = False
        self.C_success = None
        self.fom_success = None
        self.best_official_fom = None
        self.best_proxy_fom = None
        self.design = None
        self.latent = None
        self.powers = None
        self.em_pass = False
        self.length_pass = False
        self.reported_official = False
        self.proxy_design = None
        self.proxy_latent = None
        self.proxy_official = False
        self.best_low_fom = None
        self.low_latent = None
        self.low_judged_at_high = False

    def observe(
        self,
        *,
        cost: float,
        fom_value: float,
        fidelity: str,
        official: bool,
        em: bool,
        length: bool,
        powers,
        design,
        latent,
    ) -> None:
        self.curve.append(
            {
                "C": float(cost),
                "fom": float(fom_value),
                "fidelity": fidelity,
                "official": bool(official),
                "em_pass": bool(em),
                "length_pass": bool(length),
            }
        )
        design = np.array(_np(design), dtype=float, copy=True)
        latent_arr = None if latent is None else np.array(_np(latent), dtype=float, copy=True)
        if self.best_proxy_fom is None or fom_value > self.best_proxy_fom:
            self.best_proxy_fom = float(fom_value)
            self.proxy_design = design
            self.proxy_latent = latent_arr
            self.proxy_official = bool(official)
            if self.best_official_fom is None:
                self._report(design, latent_arr, powers, em, length, official)
        if official and (self.best_official_fom is None or fom_value > self.best_official_fom):
            self.best_official_fom = float(fom_value)
            self._report(design, latent_arr, powers, em, length, True)
        if not official and (self.best_low_fom is None or fom_value > self.best_low_fom):
            self.best_low_fom = float(fom_value)
            self.low_latent = None if latent_arr is None else np.array(latent_arr, copy=True)
            self.low_judged_at_high = False
        elif (
            official
            and self.low_latent is not None
            and latent_arr is not None
            and self.low_latent.shape == latent_arr.shape
            and np.allclose(self.low_latent, latent_arr)
        ):
            self.low_judged_at_high = True
        if official and em and length and not self.success:
            self.success = True
            self.C_success = float(cost)
            self.fom_success = float(fom_value)

    def _report(self, design, latent, powers, em, length, official) -> None:
        self.design = np.array(design, dtype=float, copy=True)
        self.latent = None if latent is None else np.array(latent, dtype=float, copy=True)
        self.powers = powers
        self.em_pass = bool(em)
        self.length_pass = bool(length)
        self.reported_official = bool(official)

    def result(self, payload: dict[str, Any], meter: CostMeter) -> dict[str, Any]:
        return {
            "device": payload["device"],
            "spec_id": payload["spec_id"],
            "method": payload["method"],
            "budget": int(payload["budget"]),
            "seed": int(payload["seed"]),
            "region_nm": [int(v) for v in payload["region_nm"]],
            "wavelength_offset_nm": float(payload["wavelength_offset_nm"]),
            "success": bool(self.success),
            "C_success": self.C_success,
            "fom_success": self.fom_success,
            "exhausted": not bool(self.success),
            "C_spent": float(meter.C),
            "n_forward": int(meter.n_forward),
            "n_adjoint": int(meter.n_adjoint),
            "best_official_fom": self.best_official_fom,
            "best_proxy_fom": self.best_proxy_fom,
            "em_pass": bool(self.em_pass),
            "length_pass": bool(self.length_pass),
            "reported_official": bool(self.reported_official),
            "best_powers": self.powers,
            "n_actions": len(self.actions),
            "n_illegal": int(self.n_illegal),
            "n_official": sum(1 for point in self.curve if point["official"]),
            "curve": self.curve,
            "actions": self.actions,
            "notes": self.notes,
            "error": None,
        }


def _record(trace: Trace, device, model, protocol, meter, sparams, design, waves, latent, offset):
    sparams = _np(sparams)
    design = _np(design).astype(float)
    if sparams.dtype == object or not np.isfinite(sparams).all():
        raise RuntimeError("S-parameters did not detach to a finite array")
    official = _is_official(waves, protocol, offset)
    fidelity = "high" if official else "low"
    em = all(margin >= 0.0 for margin in _margins(device, sparams, protocol))
    length = bool(length_scale_ok(model, design, protocol))
    value = float(fom(device, sparams, protocol))
    powers = transmission_report(device, sparams, waves)
    trace.observe(
        cost=meter.C,
        fom_value=value,
        fidelity=fidelity,
        official=official,
        em=em,
        length=length,
        powers=powers,
        design=design,
        latent=latent,
    )
    return official, em, length, value


def _margins(device, sparams, protocol):
    from optics_bench.judge import constraint_margins

    return constraint_margins(device, sparams, protocol)


def _judge_if_room(
    trace: Trace,
    device,
    model,
    latent,
    radius: int,
    protocol,
    meter: CostMeter,
    offset: float,
    origin: float,
    budget_limit: float,
    pending_official: bool,
) -> None:
    """High-fidelity forwards of the latent now in hand, if the budget can pay."""
    if trace.success or pending_official:
        return
    if meter.C - origin + judge_cost(protocol, offset) > budget_limit:
        trace.notes.append("final_judge_skipped")
        return
    waves = wavelengths_for(protocol, "high", offset)
    design = project_latent(latent, radius, protocol)
    meter.mode = "forward"
    sparams = simulate_s(model, design, waves)
    _record(trace, device, model, protocol, meter, sparams, design, waves, latent, offset)


def _emit(trace: Trace, protocol, action: dict[str, Any]) -> None:
    ok, why = validate_action(action, protocol)
    if not ok:
        raise RuntimeError(f"controller emitted an illegal action: {why}: {action}")
    trace.actions.append(action)


def run_fixed_budget(
    trace: Trace,
    device,
    model,
    protocol,
    meter: CostMeter,
    config,
    seed: int,
    offset: float,
    origin: float,
    budget_limit: float,
) -> None:
    """Low until half the sub-budget, then high. One Adam step at a time."""
    radius = int(protocol["parameterization"]["filter_radius_px"])
    latent = _initial_latent(model, seed, float(config["init_std"]))
    adam = fresh_adam(latent)
    pending_official = False
    trace.notes.append(f"init_seed_{int(seed)}")
    while True:
        spent = meter.C - origin
        fidelity = "low" if spent < budget_limit / 2.0 else "high"
        if spent + step_cost(protocol, fidelity, offset) > budget_limit:
            break
        sparams, design, _loss, waves, apply_update = adjoint_step(
            device,
            model,
            latent,
            protocol,
            meter,
            fidelity,
            radius_px=radius,
            adam=adam,
            wavelength_offset_nm=offset,
        )
        official, _em, _length, _value = _record(
            trace, device, model, protocol, meter, sparams, design, waves, latent, offset
        )
        if trace.success:
            return
        try:
            latent, adam = apply_update()
        except FloatingPointError:
            pending_official = official
            trace.notes.append("nonfinite_grad")
            break
        pending_official = False
    _judge_if_room(
        trace,
        device,
        model,
        latent,
        radius,
        protocol,
        meter,
        offset,
        origin,
        budget_limit,
        pending_official,
    )


def run_restart(trace, device, model, protocol, meter, config, seed, offset, budget) -> None:
    """k equal sub-budgets. Stop at the first hard success. Leftover is dropped."""
    copies = int(config["restart_k"])
    if int(budget) % copies != 0:
        raise RuntimeError(f"restart_k={copies} does not divide budget {budget}")
    sub = int(budget) // copies
    for child in range(copies):
        child_seed = int(seed) + child
        trace.notes.append(f"child_{child}_seed_{child_seed}")
        run_fixed_budget(
            trace,
            device,
            model,
            protocol,
            meter,
            config,
            child_seed,
            offset,
            origin=meter.C,
            budget_limit=sub,
        )
        if trace.success:
            trace.notes.append(f"stop_after_child_{child}")
            return


def _fit_iterations(spent, budget_limit, step, options) -> int:
    for count in options:
        if spent + int(count) * step <= budget_limit:
            return int(count)
    return 0


def run_heuristic(trace, device, model, protocol, meter, config, seed, offset, budget) -> None:
    """Adjoint on the density grid. Never calls CMA-ES.

    Fidelity is fixed for the whole action. Stagnation is counted per evaluation,
    and a rollback drops the rest of that action without paying its adjoint.
    """
    radius = int(protocol["parameterization"]["filter_radius_px"])
    cap = int(config["filter_radius_max_px"])
    stall_limit = int(config["stagnation_steps"])
    eps = float(config["stagnation_eps"])
    options = [int(v) for v in config["heuristic_iterations"]]
    rng = np.random.default_rng(int(seed))
    latent = rng.normal(0.0, float(config["init_std"]), size=_shape(model))
    adam = fresh_adam(latent)
    stall = 0
    best = None
    fresh = True
    seen_high = False
    radius_bumped = False
    current_fidelity = None
    pending_official = False
    halt = False
    while not halt:
        spent = meter.C
        fidelity = "low" if spent < budget / 2.0 else "high"
        if fidelity != current_fidelity:
            stall = 0
            best = None
            current_fidelity = fidelity
        step = step_cost(protocol, fidelity, offset)
        count = _fit_iterations(spent, budget, step, options)
        if count == 0:
            break
        if radius_bumped:
            reason = "linewidth_violation"
            radius_bumped = False
        elif fresh:
            reason = "init"
        elif fidelity == "high" and not seen_high:
            reason = "budget_half"
        else:
            reason = "improve_worst_wavelength"
        _emit(
            trace,
            protocol,
            {
                "action": "run_optimizer",
                "method": "adjoint",
                "parameterization": "density_grid",
                "objective": "minimax_transmission",
                "iterations": count,
                "fidelity": fidelity,
                "reason_code": reason,
            },
        )
        fresh = False
        if fidelity == "high":
            seen_high = True
        for _ in range(count):
            if meter.C + step > budget:
                break
            sparams, design, _loss, waves, apply_update = adjoint_step(
                device,
                model,
                latent,
                protocol,
                meter,
                fidelity,
                radius_px=radius,
                adam=adam,
                wavelength_offset_nm=offset,
            )
            official, _em, length, value = _record(
                trace, device, model, protocol, meter, sparams, design, waves, latent, offset
            )
            if trace.success:
                _emit(trace, protocol, {"action": "stop", "reason_code": "spec_met"})
                return
            if best is None or value > best + eps:
                best = value
                stall = 0
            else:
                stall += 1
            if not length and radius < cap:
                radius += 1
                radius_bumped = True
            if stall >= stall_limit:
                _emit(trace, protocol, {"action": "rollback", "reason_code": "stagnation"})
                latent = rng.normal(0.0, float(config["init_std"]), size=_shape(model))
                adam = fresh_adam(latent)
                stall = 0
                best = None
                fresh = True
                pending_official = False
                break
            try:
                latent, adam = apply_update()
            except FloatingPointError:
                pending_official = official
                trace.notes.append("nonfinite_grad")
                halt = True
                break
            pending_official = False
    _judge_if_room(
        trace,
        device,
        model,
        latent,
        radius,
        protocol,
        meter,
        offset,
        0.0,
        budget,
        pending_official,
    )


def make_cma(seed: int, config):
    import cma

    cma_cfg = config["cma"]
    dimension = int(cma_cfg["dimension"])
    start = [float(cma_cfg["x0"])] * dimension
    options = {
        "popsize": int(cma_cfg["popsize"]),
        "bounds": [float(v) for v in cma_cfg["bounds"]],
        "seed": int(seed),
        "maxiter": 10**9,
        "maxfevals": 10**9,
        "tolfun": 1e-18,
        "tolfunhist": 1e-18,
        "tolfunrel": 0,
        "tolx": 1e-18,
        "tolstagnation": 10**9,
        "tolflatfitness": 1e9,
        "tolconditioncov": False,
        "tolfacupx": 1e6,
        "tolupsigma": 1e20,
        "verbose": -9,
    }
    return cma.CMAEvolutionStrategy(start, float(cma_cfg["sigma0"]), options)


def run_cma(trace, device, model, protocol, meter, config, seed, offset, budget) -> None:
    """Minimize the hinge on the 8-parameter geometry. Generations are atomic."""
    strategy = make_cma(seed, config)
    pop = int(config["cma"]["popsize"])
    wg_px = waveguide_px(protocol, device)
    while True:
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
            design = render(device, model, vector, wg_px)
            meter.mode = "forward"
            sparams = simulate_s(model, design, waves)
            _record(
                trace, device, model, protocol, meter, sparams, design, waves, vector, offset
            )
            loss = float(_np(hinge_loss(device, _np(sparams), protocol)))
            fitness.append(loss if np.isfinite(loss) else 1.0e6)
            if trace.success:
                succeeded = True
                break
        trace.notes.append(f"cma_{fidelity}_n{len(fitness)}_C{meter.C}")
        if succeeded:
            trace.notes.append("cma_stop_success")
            return
        strategy.tell(points, fitness)
    # The incumbent is the best low-fidelity sample, not the best high-fidelity one.
    if trace.success or trace.low_latent is None or trace.low_judged_at_high:
        return
    if meter.C + judge_cost(protocol, offset) > budget:
        trace.notes.append("final_judge_skipped")
        return
    waves = wavelengths_for(protocol, "high", offset)
    design = render(device, model, trace.low_latent, wg_px)
    meter.mode = "forward"
    sparams = simulate_s(model, design, waves)
    _record(
        trace,
        device,
        model,
        protocol,
        meter,
        sparams,
        design,
        waves,
        trace.low_latent,
        offset,
    )


def _blank(payload, meter: CostMeter | None, error: str) -> dict[str, Any]:
    spent = 0.0 if meter is None else float(meter.C)
    return {
        "device": payload["device"],
        "spec_id": payload["spec_id"],
        "method": payload["method"],
        "budget": int(payload["budget"]),
        "seed": int(payload["seed"]),
        "region_nm": [int(v) for v in payload["region_nm"]],
        "wavelength_offset_nm": float(payload["wavelength_offset_nm"]),
        "success": False,
        "C_success": None,
        "fom_success": None,
        "exhausted": True,
        "C_spent": spent,
        "n_forward": 0 if meter is None else int(meter.n_forward),
        "n_adjoint": 0 if meter is None else int(meter.n_adjoint),
        "best_official_fom": None,
        "best_proxy_fom": None,
        "em_pass": False,
        "length_pass": False,
        "reported_official": False,
        "best_powers": None,
        "n_actions": 0,
        "n_illegal": 0,
        "n_official": 0,
        "curve": [],
        "actions": [],
        "notes": [],
        "error": error,
    }


def _write_result(path: Path, result: dict[str, Any], design, latent) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    temporary.replace(path)
    if design is None:
        return
    np.savez_compressed(
        path.with_suffix(".npz"),
        design=np.asarray(design, dtype=float),
        latent=np.asarray([] if latent is None else latent, dtype=float),
    )


def _is_done(path: Path) -> bool:
    import json

    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return not row.get("error")


def run_job(payload: dict[str, Any]) -> str:
    """One budget, one device, one seed. Writes JSON even when the solve fails."""
    import os

    out = Path(payload["out"])
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("OMP_NUM_THREADS is not 1; refusing to oversubscribe")
    if _is_done(out):
        return str(out)
    from optics_bench.protocol import load_protocol

    protocol = load_protocol()
    config = load_baselines()
    meter = CostMeter()
    install_solve_counter(meter)
    trace = Trace()
    try:
        offset = float(payload["wavelength_offset_nm"])
        model = make_model(
            payload["device"],
            protocol,
            fidelity="low",
            region_nm=tuple(payload["region_nm"]),
            wavelength_offset_nm=offset,
        )
        method = payload["method"]
        if method == "fixed_adjoint":
            run_fixed_budget(
                trace,
                payload["device"],
                model,
                protocol,
                meter,
                config,
                int(payload["seed"]),
                offset,
                origin=0.0,
                budget_limit=float(payload["budget"]),
            )
        elif method == "random_restart":
            run_restart(
                trace,
                payload["device"],
                model,
                protocol,
                meter,
                config,
                int(payload["seed"]),
                offset,
                int(payload["budget"]),
            )
        elif method == "cmaes":
            run_cma(
                trace,
                payload["device"],
                model,
                protocol,
                meter,
                config,
                int(payload["seed"]),
                offset,
                float(payload["budget"]),
            )
        elif method == "heuristic":
            run_heuristic(
                trace,
                payload["device"],
                model,
                protocol,
                meter,
                config,
                int(payload["seed"]),
                offset,
                float(payload["budget"]),
            )
        else:
            raise ValueError(f"unknown method: {method}")
        if meter.C > float(payload["budget"]) + 1e-6:
            raise RuntimeError(f"budget exceeded: C={meter.C} budget={payload['budget']}")
        result = trace.result(payload, meter)
        design, latent = trace.design, trace.latent
    except Exception:
        try:
            result = trace.result(payload, meter)
        except Exception:
            result = _blank(payload, meter, "")
        result["error"] = traceback.format_exc()
        design, latent = trace.design, trace.latent
    _write_result(out, result, design, latent)
    return str(out)
