"""Day-1 lock check: schema, one adjoint pass, and 20 timed solves."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optics_bench.actions import parse_action  # noqa: E402
from optics_bench.cost import CostMeter  # noqa: E402
from optics_bench.devices import make_model  # noqa: E402
from optics_bench.judge import fom, hard_success, transmission_report  # noqa: E402
from optics_bench.optimize import run_adjoint, zeros_latent  # noqa: E402
from optics_bench.parameterize import density_from_latent  # noqa: E402
from optics_bench.protocol import load_protocol, wavelengths_for  # noqa: E402
from optics_bench.simulate import install_solve_counter, simulate_s  # noqa: E402


def _check_schema(protocol: dict) -> list[dict]:
    samples = [
        (
            '{"action":"run_optimizer","method":"adjoint","parameterization":"density_grid",'
            '"objective":"minimax_transmission","iterations":5,"fidelity":"low",'
            '"reason_code":"improve_worst_wavelength"}',
            True,
        ),
        ('{"action":"rollback"}', True),
        ('{"action":"stop"}', True),
        (
            '{"action":"run_optimizer","method":"cmaes","parameterization":"density_grid",'
            '"objective":"minimax_transmission","iterations":5,"fidelity":"low",'
            '"reason_code":"init"}',
            False,
        ),
        ("{not json", False),
        (
            '{"action":"run_optimizer","method":"adjoint","parameterization":"density_grid",'
            '"objective":"minimax_transmission","iterations":4,"fidelity":"low",'
            '"reason_code":"init"}',
            False,
        ),
    ]
    rows = []
    for text, expect in samples:
        ok, reason, _ = parse_action(text, protocol)
        rows.append({"ok": ok, "reason": reason, "expect": expect, "pass": ok == expect})
        if ok != expect:
            raise SystemExit(f"schema check failed: {text} -> {ok} {reason}")
    return rows


def _section(meter: CostMeter, before: dict) -> dict:
    now = meter.as_dict()
    return {
        "n_forward": now["n_forward"] - before["n_forward"],
        "n_adjoint": now["n_adjoint"] - before["n_adjoint"],
        "C": now["C"] - before["C"],
    }


def _eval(device: str, model, latent, protocol, meter: CostMeter, fidelity: str) -> dict:
    para = protocol["parameterization"]
    design = density_from_latent(
        latent,
        int(para["filter_radius_px"]),
        float(para["beta"]),
        float(para["eta"]),
    )
    wavelengths = wavelengths_for(protocol, fidelity)
    before = meter.n_forward
    sparams = simulate_s(model, design, wavelengths)
    sparams_np = np.asarray(sparams)
    return {
        "fidelity": fidelity,
        "wavelengths_nm": wavelengths,
        "forward_solves": meter.n_forward - before,
        "fom": fom(device, sparams_np, protocol),
        "hard_success": hard_success(device, model, np.asarray(design), sparams_np, protocol),
        "powers": transmission_report(device, sparams_np, wavelengths),
    }


def _time_forwards(model, latent, protocol, meter: CostMeter, n_solves: int) -> dict:
    para = protocol["parameterization"]
    design = np.asarray(
        density_from_latent(
            latent,
            int(para["filter_radius_px"]),
            float(para["beta"]),
            float(para["eta"]),
        )
    )
    wavelength = [float(protocol["cost"]["low_wavelengths_nm"][0])]
    # Warm the port eigenmode once. That solve is not part of the 20.
    simulate_s(model, design, wavelength)
    start_forward = meter.n_forward
    t0 = time.perf_counter()
    for _ in range(n_solves):
        simulate_s(model, design, wavelength)
    elapsed = time.perf_counter() - t0
    counted = meter.n_forward - start_forward
    return {
        "n_requested": n_solves,
        "n_forward_counted": counted,
        "seconds": elapsed,
        "seconds_per_solve": elapsed / n_solves,
        "warmup_solves_excluded_from_timing": 1,
    }


def main() -> None:
    protocol = load_protocol()
    log = {
        "commit": protocol["ceviche_challenges_commit"],
        "schema": _check_schema(protocol),
    }
    meter = CostMeter(alpha=protocol["cost"]["alpha"])
    install_solve_counter(meter)
    mark = meter.as_dict()

    bend = make_model("waveguide_bend", protocol, fidelity="low")
    latent = zeros_latent(bend)
    log["bend_shape"] = {
        "domain": [int(x) for x in bend.shape],
        "design": [int(x) for x in bend.design_variable_shape],
    }
    log["bend_gray"] = _eval("waveguide_bend", bend, latent, protocol, meter, "low")
    log["bend_gray"]["section_cost"] = _section(meter, mark)
    mark = meter.as_dict()
    t0 = time.perf_counter()
    latent, loss = run_adjoint(
        "waveguide_bend", bend, latent, protocol, meter, iterations=5, fidelity="low"
    )
    after = _eval("waveguide_bend", bend, latent, protocol, meter, "low")
    log["bend_adjoint_5"] = {
        "seconds": time.perf_counter() - t0,
        "final_hinge": loss,
        "after": after,
        "section_cost": _section(meter, mark),
    }
    mark = meter.as_dict()
    log["bend_timing_20"] = _time_forwards(bend, latent, protocol, meter, 20)
    log["bend_timing_20"]["section_cost"] = _section(meter, mark)

    mark = meter.as_dict()
    splitter = make_model("beam_splitter", protocol, fidelity="low")
    splitter_latent = zeros_latent(splitter)
    log["splitter_shape"] = {
        "domain": [int(x) for x in splitter.shape],
        "design": [int(x) for x in splitter.design_variable_shape],
    }
    log["splitter_gray"] = _eval(
        "beam_splitter", splitter, splitter_latent, protocol, meter, "low"
    )
    log["splitter_gray"]["section_cost"] = _section(meter, mark)
    mark = meter.as_dict()
    t0 = time.perf_counter()
    splitter_latent, loss = run_adjoint(
        "beam_splitter",
        splitter,
        splitter_latent,
        protocol,
        meter,
        iterations=2,
        fidelity="low",
    )
    after = _eval("beam_splitter", splitter, splitter_latent, protocol, meter, "low")
    log["splitter_adjoint_2"] = {
        "seconds": time.perf_counter() - t0,
        "final_hinge": loss,
        "after": after,
        "section_cost": _section(meter, mark),
    }
    log["cost_total"] = meter.as_dict()

    out = ROOT / "logs" / "day1_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
