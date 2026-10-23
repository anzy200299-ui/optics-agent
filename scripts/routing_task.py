"""Family-routing measurement. Pilot checks the keep-outs, then the sweep."""

from __future__ import annotations

import argparse
import json
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

from optics_bench.cost import CostMeter
from optics_bench.devices import make_model
from optics_bench.judge import length_scale_ok, powers
from optics_bench.parametric import bend_density, waveguide_px
from optics_bench.protocol import load_protocol, wavelengths_for
from optics_bench.routing import (
    detour_density,
    keepout_mask,
    load_routing,
    mask_rule_family,
    render_family,
    run_routing,
)
from optics_bench.simulate import install_solve_counter, simulate_s

JOBS = ROOT / "logs" / "routing" / "jobs"


def _connected(design) -> bool:
    solid = np.asarray(design) >= 0.5
    if not np.any(solid):
        return False
    labels, _count = ndimage.label(solid)
    left = set(labels[0, :][solid[0, :]].tolist())
    bottom = set(labels[:, 0][solid[:, 0]].tolist())
    left.discard(0)
    bottom.discard(0)
    return bool(left & bottom)


def _grid(samples: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    points = [np.full(8, 0.5)]
    for index in range(8):
        for value in (0.0, 1.0):
            point = np.full(8, 0.5)
            point[index] = value
            points.append(point)
    draws = rng.random((samples, 8))
    points.extend(draws)
    return points


def pilot() -> None:
    protocol = load_protocol()
    task = load_routing()
    wg_px = waveguide_px(protocol, "waveguide_bend")
    model = make_model(
        "waveguide_bend",
        protocol,
        fidelity="low",
        region_nm=(1600, 1600),
    )
    print(f"grid {tuple(int(v) for v in model.design_variable_shape)} wg {wg_px}", flush=True)
    points = _grid(24, 0)
    for kind in ("clean", "blocked"):
        mask = keepout_mask(kind, model, wg_px, task)
        print(f"{kind} keep-out pixels {int(mask.sum())} rule {mask_rule_family(model, wg_px, mask)}", flush=True)
        for family, draw in (("arc", bend_density), ("detour", detour_density)):
            hit = 0
            linked = 0
            for point in points:
                raw = draw(model, point, wg_px) >= 0.5
                cut = raw.copy()
                cut[mask] = False
                hit += int(np.any(raw & mask))
                linked += int(_connected(cut))
            print(
                f"  {family:7} hits {hit}/{len(points)} still_connected {linked}/{len(points)}",
                flush=True,
            )
    meter = CostMeter()
    install_solve_counter(meter)
    waves = wavelengths_for(protocol, "low")
    nominal = np.full(8, 0.5)
    for kind in ("clean", "blocked"):
        mask = keepout_mask(kind, model, wg_px, task)
        for family in ("arc", "detour"):
            design = render_family(family, model, nominal, wg_px, mask)
            before = meter.n_forward
            raw = simulate_s(model, design, waves)
            sparams = np.asarray(getattr(raw, "_value", raw))
            transmission = float(powers(sparams)[0, 0, 1])
            print(
                f"nominal {kind:7} {family:7} |S21|^2={transmission:.4f} "
                f"connected={_connected(design)} length={length_scale_ok(model, design, protocol)} "
                f"solves={meter.n_forward - before}",
                flush=True,
            )


def plan_jobs(task) -> list[dict]:
    jobs = []
    for size in task["sizes_nm"]:
        for kind in task["classes"]:
            for method in task["methods"]:
                for seed in task["seeds"]:
                    name = f"{kind}_{int(size)}__{method}__s{int(seed)}.json"
                    jobs.append(
                        {
                            "device": "waveguide_bend",
                            "spec_id": f"{kind}_{int(size)}",
                            "class_name": kind,
                            "region_nm": [int(size), int(size)],
                            "wavelength_offset_nm": float(task["wavelength_offset_nm"]),
                            "method": method,
                            "budget": int(task["budget"]),
                            "seed": int(seed),
                            "out": str(JOBS / name),
                        }
                    )
    return jobs


def _done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return not row.get("error")


def summarize(task, jobs) -> None:
    rows = []
    for job in jobs:
        path = Path(job["out"])
        if path.exists():
            rows.append(json.loads(path.read_text()))
    right = task["right_family"]
    print("\nmethod class n success mean_fom", flush=True)
    for method in list(task["methods"]) + ["oracle", "mask_rule"]:
        for kind in task["classes"]:
            if method == "oracle":
                chosen = [row for row in rows if row["class_name"] == kind and row["method"] == f"always_{right[kind]}"]
            elif method == "mask_rule":
                chosen = []
                for row in rows:
                    if row["class_name"] != kind:
                        continue
                    note = next((item for item in row["notes"] if item.startswith("mask_rule_")), "")
                    family = note.replace("mask_rule_", "")
                    if row["method"] == f"always_{family}":
                        chosen.append(row)
            else:
                chosen = [row for row in rows if row["method"] == method and row["class_name"] == kind]
            if not chosen:
                continue
            successes = sum(1 for row in chosen if row["success"])
            foms = [row["best_official_fom"] for row in chosen if row["best_official_fom"] is not None]
            mean = sum(foms) / len(foms) if foms else float("nan")
            print(f"{method:16} {kind:8} {len(chosen):3} {successes:3} {mean:+.3f}", flush=True)
    probes = [row for row in rows if row["method"] == "probe_commit" and not row.get("error")]
    correct = 0
    for row in probes:
        note = next((item for item in row["notes"] if "_chose_" in item), "")
        family = note.split("_chose_")[-1] if note else ""
        correct += int(family == right[row["class_name"]])
    if probes:
        print(f"probe chose the right family {correct}/{len(probes)}", flush=True)
    errors = [row for row in rows if row.get("error")]
    print(f"errors {len(errors)}", flush=True)
    out = ROOT / "logs" / "routing" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"jobs": rows, "probe_correct": correct if probes else None}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.pilot or not args.run:
        pilot()
    if not args.run:
        return
    import multiprocessing

    task = load_routing()
    jobs = plan_jobs(task)
    pending = [job for job in jobs if not _done(Path(job["out"]))]
    print(f"routing jobs {len(pending)} pending, {len(jobs) - len(pending)} done", flush=True)
    if pending:
        JOBS.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("spawn")
        with context.Pool(args.workers) as pool:
            done = 0
            for result in pool.imap_unordered(run_routing, pending):
                done += 1
                official = result.get("best_official_fom")
                text = "none" if official is None else f"{official:.3f}"
                print(
                    f"[{done}/{len(pending)}] {result.get('class_name')} {result.get('method')} "
                    f"{result.get('spec_id')} s{result.get('seed')} C={result.get('C_spent')} "
                    f"fom={text} success={result.get('success')} err={bool(result.get('error'))}",
                    flush=True,
                )
    summarize(task, jobs)


if __name__ == "__main__":
    main()
