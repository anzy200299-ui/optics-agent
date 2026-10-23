"""Day-2 classical baselines. Phase A finds the unsaturated adjoint budget."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy import ndimage

from optics_bench.baselines import load_baselines, make_cma, run_job
from optics_bench.cost import CostMeter
from optics_bench.devices import make_model
from optics_bench.judge import _thin_medial, length_scale_ok, powers
from optics_bench.optimize import adjoint_step, fresh_adam, zeros_latent
from optics_bench.parametric import render, waveguide_px
from optics_bench.protocol import judge_cost, load_protocol, step_cost, wavelengths_for
from optics_bench.simulate import install_solve_counter, simulate_s
from optics_bench.specs import load_specs

JOBS = ROOT / "logs" / "day2" / "jobs"


def _ascii(design) -> str:
    array = np.asarray(design)
    rows = []
    for y in range(array.shape[1] - 1, -1, -1):
        rows.append("".join("#" if array[x, y] >= 0.5 else "." for x in range(array.shape[0])))
    return "\n".join(rows)


def _connected(design) -> bool:
    solid = np.asarray(design) >= 0.5
    labels, _count = ndimage.label(solid)
    left = set(labels[0, :][solid[0, :]].tolist())
    bottom = set(labels[:, 0][solid[:, 0]].tolist())
    left.discard(0)
    bottom.discard(0)
    return bool(left & bottom)


def _medial_tests() -> None:
    limit = 1.5
    solid = np.ones((5, 5), dtype=bool)
    everywhere = np.ones((5, 5), dtype=bool)
    if _thin_medial(solid, limit, everywhere):
        raise SystemExit("5x5 solid was marked thinner than 150 nm")
    if _thin_medial(np.zeros((5, 5), dtype=bool), limit, everywhere):
        raise SystemExit("empty mask was marked thin")
    line = np.zeros((7, 7), dtype=bool)
    line[3, 1:6] = True
    if not _thin_medial(line, limit, np.ones_like(line)):
        raise SystemExit("1 px line passed the 150 nm check")
    outside = np.zeros_like(line)
    outside[0, 0] = True
    if _thin_medial(line, limit, outside):
        raise SystemExit("a thin line outside the design region was scored")
    bar = np.zeros((9, 9), dtype=bool)
    bar[3:6, 1:8] = True
    if _thin_medial(bar, limit, np.ones_like(bar)):
        raise SystemExit("3 px bar was marked thinner than 150 nm")


def _cma_options(config) -> None:
    strategy = make_cma(0, config)
    for _ in range(3):
        if strategy.stop():
            raise SystemExit(f"CMA stopped before the budget rule: {strategy.stop()}")
        points = list(strategy.ask())
        if len(points) != 4:
            raise SystemExit(f"CMA population is {len(points)}, locked at 4")
        flat = [float(value) for point in points for value in point]
        if min(flat) < 0.0 or max(flat) > 1.0:
            raise SystemExit("CMA proposed a point outside [0, 1]")
        strategy.tell(points, [float(np.sum((np.asarray(point) - 0.2) ** 2)) for point in points])
    if strategy.stop():
        raise SystemExit(f"CMA stopped after 3 generations: {strategy.stop()}")


def _geometries(protocol, specs) -> None:
    nominal = np.full(8, 0.5)
    for device, rows in specs.items():
        wg_px = waveguide_px(protocol, device)
        if wg_px != 8:
            raise SystemExit(f"{device} waveguide is {wg_px} cells, expected 8")
        for row in rows:
            model = make_model(
                device,
                protocol,
                fidelity="low",
                region_nm=tuple(row["region_nm"]),
                wavelength_offset_nm=float(row["wavelength_offset_nm"]),
            )
            design = render(device, model, nominal, wg_px)
            if tuple(int(v) for v in design.shape) != tuple(int(v) for v in model.design_variable_shape):
                raise SystemExit(f"{row['id']} render shape {design.shape}")
            full = np.asarray(model.density(design))
            if tuple(full.shape) != tuple(int(v) for v in model.shape):
                raise SystemExit(f"{row['id']} does not inlay")
            if device == "waveguide_bend":
                if not _connected(design) or design[-1, -1] >= 0.5:
                    raise SystemExit(
                        f"{row['id']} quarter-circle is not a bend from the left port "
                        f"to the bottom port\n{_ascii(design)}"
                    )
                fill = float(design.mean())
                if not 0.02 < fill < 0.75:
                    raise SystemExit(f"{row['id']} fill {fill:.3f} is not a trace")
            else:
                x0, y0, _, _ = (int(v) for v in model.design_region_coords)
                nx, ny = design.shape
                guide = np.asarray(model.density_bg)[x0 : x0 + nx, y0 : y0 + ny] >= 0.5
                if not np.any(guide) or not np.all(design[guide] == 1.0):
                    raise SystemExit(f"{row['id']} dropped the through waveguide")
                if not np.any(design[:, -1] >= 0.5):
                    raise SystemExit(f"{row['id']} nominal protrusion does not reach the mirror")
    bend = make_model(
        "waveguide_bend",
        protocol,
        fidelity="low",
        region_nm=(1600, 1600),
    )
    plain = render("waveguide_bend", bend, nominal, 8)
    bulged = nominal.copy()
    bulged[7] = 1.0
    if np.array_equal(plain, render("waveguide_bend", bend, bulged, 8)):
        raise SystemExit("bend bulge parameter has no effect")
    splitter = make_model(
        "beam_splitter",
        protocol,
        fidelity="low",
        region_nm=(3200, 2000),
    )
    base = render("beam_splitter", splitter, nominal, 8)
    hole_off = nominal.copy()
    hole_off[7] = 1.0
    if not np.array_equal(base, render("beam_splitter", splitter, hole_off, 8)):
        raise SystemExit("p5=0.5 still punches a hole")
    punched = nominal.copy()
    punched[5] = 1.0
    punched[7] = 1.0
    if render("beam_splitter", splitter, punched, 8).sum() >= base.sum():
        raise SystemExit("p5>0.5 did not open a hole")


def _nominal_bend(protocol) -> None:
    model = make_model(
        "waveguide_bend",
        protocol,
        fidelity="low",
        region_nm=(1600, 1600),
    )
    design = render("waveguide_bend", model, np.full(8, 0.5), 8)
    meter = CostMeter()
    install_solve_counter(meter)
    waves = wavelengths_for(protocol, "low")
    raw = simulate_s(model, design, waves)
    sparams = np.asarray(getattr(raw, "_value", raw))
    if sparams.dtype == object or sparams.shape != (1, 1, 2):
        raise SystemExit(f"nominal bend S shape {getattr(sparams, 'shape', None)} dtype {sparams.dtype}")
    if meter.n_forward != 1 or meter.n_adjoint != 0:
        raise SystemExit(f"nominal bend solve count {meter.as_dict()}")
    transmission = float(powers(sparams)[0, 0, 1])
    length_ok = bool(length_scale_ok(model, design, protocol))
    print(
        f"nominal bend |S21|^2={transmission:.4f} length_ok={length_ok} "
        f"fill={float(design.mean()):.3f}",
        flush=True,
    )
    if transmission < 0.05:
        raise SystemExit(f"nominal bend transmission {transmission:.4f} is below 0.05\n{_ascii(design)}")
    if not length_ok:
        raise SystemExit(
            "nominal 400 nm bend fails the 150 nm medial check; "
            "the checker is flagging a feature wider than the threshold\n"
            + _ascii(design)
        )


def _adam_regression(protocol) -> None:
    model = make_model("waveguide_bend", protocol, fidelity="low")
    meter = CostMeter()
    install_solve_counter(meter)
    latent = zeros_latent(model)
    adam = fresh_adam(latent)
    first, _design, _loss, _waves, apply0 = adjoint_step(
        "waveguide_bend", model, latent, protocol, meter, "low", adam=adam
    )
    latent, adam = apply0()
    _second, _design, _loss, _waves, apply1 = adjoint_step(
        "waveguide_bend", model, latent, protocol, meter, "low", adam=adam
    )
    latent, adam = apply1()
    third, _design, _loss, _waves, _apply2 = adjoint_step(
        "waveguide_bend", model, latent, protocol, meter, "low", adam=adam
    )
    start = float(powers(np.asarray(getattr(first, "_value", first)))[0, 0, 1])
    after = float(powers(np.asarray(getattr(third, "_value", third)))[0, 0, 1])
    print(
        f"adam |S21|^2 {start:.5f} -> {after:.5f} "
        f"F={meter.n_forward} A={meter.n_adjoint}",
        flush=True,
    )
    if meter.n_forward != 3 or meter.n_adjoint != 2:
        raise SystemExit(f"two Adam steps counted {meter.as_dict()}, expected 3 forward and 2 adjoint")
    if not after > start:
        raise SystemExit(f"two Adam steps did not raise transmission ({start:.5f} -> {after:.5f})")


def preflight() -> None:
    protocol = load_protocol()
    specs = load_specs()
    config = load_baselines()
    if step_cost(protocol, "low") != 2 or step_cost(protocol, "high") != 12:
        raise SystemExit("step cost drifted from the day-1 lock")
    if judge_cost(protocol) != 6:
        raise SystemExit("judge cost drifted from the day-1 lock")
    if list(protocol["cost"]["budgets"]) != [20, 40, 80]:
        raise SystemExit("budgets drifted from the day-1 lock")
    _medial_tests()
    _cma_options(config)
    _geometries(protocol, specs)
    _nominal_bend(protocol)
    _adam_regression(protocol)
    print("preflight ok", flush=True)


def plan_jobs(phase: str, budget: int | None, protocol, specs, config) -> list[dict]:
    if phase == "A":
        budgets = [int(v) for v in protocol["cost"]["budgets"]]
        seeds = [int(v) for v in config["phase_a_seeds"]]
        chosen = [
            (device, row)
            for device, rows in specs.items()
            for row in rows
            if row.get("nominal")
        ]
    elif phase == "B":
        if budget is None or int(budget) not in [int(v) for v in protocol["cost"]["budgets"]]:
            raise SystemExit("phase B needs --budget 20, 40, or 80")
        budgets = [int(budget)]
        seeds = [int(v) for v in config["phase_b_seeds"]]
        chosen = [(device, row) for device, rows in specs.items() for row in rows]
    else:
        raise SystemExit(f"unknown phase {phase}")
    jobs = []
    for device, row in chosen:
        for method in config["methods"]:
            for limit in budgets:
                for seed in seeds:
                    name = f"{device}__{row['id']}__{method}__c{limit}__s{seed}.json"
                    jobs.append(
                        {
                            "device": device,
                            "spec_id": row["id"],
                            "region_nm": [int(v) for v in row["region_nm"]],
                            "wavelength_offset_nm": float(row["wavelength_offset_nm"]),
                            "method": method,
                            "budget": int(limit),
                            "seed": int(seed),
                            "out": str(JOBS / name),
                        }
                    )
    return jobs


def wilson(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.96
    phat = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (phat + z2 / (2.0 * total)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / total + z2 / (4.0 * total * total)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _slim(row: dict) -> dict:
    keep = (
        "device",
        "spec_id",
        "method",
        "budget",
        "seed",
        "region_nm",
        "wavelength_offset_nm",
        "success",
        "C_success",
        "fom_success",
        "exhausted",
        "C_spent",
        "n_forward",
        "n_adjoint",
        "best_official_fom",
        "best_proxy_fom",
        "em_pass",
        "length_pass",
        "reported_official",
        "n_actions",
        "n_illegal",
        "n_official",
        "notes",
        "error",
    )
    return {key: row.get(key) for key in keep}


def choose_main_budget(rows: list[dict], budgets: list[int], tol: float) -> dict:
    adjoint = [row for row in rows if row["method"] == "fixed_adjoint" and not row.get("error")]
    by_budget = {}
    per_device = {}
    for limit in budgets:
        group = [row for row in adjoint if int(row["budget"]) == int(limit)]
        foms = [row["best_official_fom"] for row in group]
        if not group or any(value is None for value in foms):
            return {
                "main_budget": None,
                "error": f"missing official FoM at C={limit}",
                "per_device": {
                    str(limit): {
                        row["device"]: row["best_official_fom"]
                        for row in adjoint
                        if int(row["budget"]) == int(limit)
                    }
                },
            }
        by_budget[str(limit)] = {
            "mean_official_fom": sum(foms) / len(foms),
            "success_rate": sum(1 for row in group if row["success"]) / len(group),
        }
        per_device[str(limit)] = {row["device"]: row["best_official_fom"] for row in group}
    unsaturated = []
    comparisons = []
    for left, right in zip(budgets, budgets[1:]):
        delta_fom = by_budget[str(right)]["mean_official_fom"] - by_budget[str(left)]["mean_official_fom"]
        delta_success = by_budget[str(right)]["success_rate"] - by_budget[str(left)]["success_rate"]
        saturated = delta_fom <= tol and delta_success <= 0.0
        comparisons.append(
            {
                "budget": int(left),
                "next": int(right),
                "delta_fom": delta_fom,
                "delta_success": delta_success,
                "saturated": saturated,
            }
        )
        if not saturated:
            unsaturated.append(int(left))
    last, prev = budgets[-1], budgets[-2]
    delta_fom = by_budget[str(last)]["mean_official_fom"] - by_budget[str(prev)]["mean_official_fom"]
    delta_success = by_budget[str(last)]["success_rate"] - by_budget[str(prev)]["success_rate"]
    last_rose = delta_fom > tol or delta_success > 0.0
    comparisons.append(
        {
            "budget": int(last),
            "next": None,
            "delta_fom": delta_fom,
            "delta_success": delta_success,
            "saturated": not last_rose,
        }
    )
    if last_rose:
        unsaturated.append(int(last))
    chosen = max(unsaturated) if unsaturated else int(budgets[0])
    return {
        "main_budget": chosen,
        "unsaturated": unsaturated,
        "by_budget": by_budget,
        "per_device": per_device,
        "comparisons": comparisons,
        "tolerance": tol,
    }


def summarize(phase: str, jobs: list[dict], protocol, config) -> dict:
    rows = []
    missing = []
    for job in jobs:
        path = Path(job["out"])
        if not path.exists():
            missing.append(job["out"])
            continue
        rows.append(json.loads(path.read_text()))
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row["method"], row["device"], int(row["budget"])), []).append(row)
    aggregates = []
    for key in sorted(groups):
        group = groups[key]
        clean = [row for row in group if not row.get("error")]
        successes = sum(1 for row in clean if row["success"])
        official = [row["best_official_fom"] for row in clean if row["best_official_fom"] is not None]
        actions = sum(int(row["n_actions"]) for row in clean)
        illegal = sum(int(row["n_illegal"]) for row in clean)
        aggregates.append(
            {
                "method": key[0],
                "device": key[1],
                "budget": key[2],
                "n": len(clean),
                "errors": len(group) - len(clean),
                "successes": successes,
                "success_rate": None if not clean else successes / len(clean),
                "wilson95": wilson(successes, len(clean)),
                "mean_official_fom": None if not official else sum(official) / len(official),
                "mean_C_spent": None if not clean else sum(row["C_spent"] for row in clean) / len(clean),
                "illegal_rate": None if actions == 0 else illegal / actions,
            }
        )
    summary = {
        "phase": phase,
        "n_jobs": len(jobs),
        "n_missing": len(missing),
        "missing": missing,
        "aggregates": aggregates,
        "jobs": [_slim(row) for row in rows],
    }
    if phase == "A" and not missing:
        summary["saturation"] = choose_main_budget(
            rows,
            [int(v) for v in protocol["cost"]["budgets"]],
            float(config["saturation_fom"]),
        )
    return summary


def _print_summary(summary: dict) -> None:
    for row in summary["jobs"]:
        official = row["best_official_fom"]
        official_text = "none" if official is None else f"{official:.4f}"
        flag = "ERR" if row.get("error") else ("ok" if row["success"] else "miss")
        print(
            f"{flag:4} {row['method']:16} {row['device']:16} {row['spec_id']:4} "
            f"C{row['budget']:<3} s{row['seed']} spent={row['C_spent']:<5} "
            f"official={official_text} em={row['em_pass']} length={row['length_pass']}",
            flush=True,
        )
    saturation = summary.get("saturation")
    if not saturation:
        return
    print("saturation", json.dumps(saturation, indent=2), flush=True)
    print(f"main_budget: {saturation.get('main_budget')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--phase", choices=("A", "B"))
    parser.add_argument("--budget", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()
    if not args.skip_check:
        preflight()
    if args.check and not args.phase:
        return
    if not args.phase:
        raise SystemExit("pass --check or --phase")
    protocol = load_protocol()
    specs = load_specs()
    config = load_baselines()
    jobs = plan_jobs(args.phase, args.budget, protocol, specs, config)
    pending = []
    for job in jobs:
        path = Path(job["out"])
        if not path.exists():
            pending.append(job)
            continue
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError:
            pending.append(job)
            continue
        if row.get("error"):
            pending.append(job)
    print(f"phase {args.phase}: {len(pending)} to run, {len(jobs) - len(pending)} already done", flush=True)
    if pending:
        JOBS.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("spawn")
        with context.Pool(args.workers) as pool:
            done = 0
            for path in pool.imap_unordered(run_job, pending):
                done += 1
                row = json.loads(Path(path).read_text())
                official = row["best_official_fom"]
                official_text = "none" if official is None else f"{official:.4f}"
                print(
                    f"[{done}/{len(pending)}] {row['method']} {row['device']} {row['spec_id']} "
                    f"C{row['budget']} spent={row['C_spent']} official={official_text} "
                    f"success={row['success']} error={bool(row.get('error'))}",
                    flush=True,
                )
    summary = summarize(args.phase, jobs, protocol, config)
    out = ROOT / "logs" / "day2" / f"phase{args.phase}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _print_summary(summary)
    errors = [row for row in summary["jobs"] if row.get("error")]
    if summary["n_missing"] or errors:
        raise SystemExit(f"{summary['n_missing']} missing, {len(errors)} errors")
    if args.phase == "A" and summary.get("saturation", {}).get("main_budget") is None:
        raise SystemExit("phase A did not produce a main budget")


if __name__ == "__main__":
    main()
