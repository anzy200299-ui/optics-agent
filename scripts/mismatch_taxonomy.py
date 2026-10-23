"""Taxonomy of PICBench netlists that SAX already simulated.

Official Without-EF labels stay as stored. A functional pass is a spectrum
match at ten decimal places, compared as an unordered set of curves. A
functional error is simulation-valid and not that match. This script does
not rescore Pass@k and does not choose a checkpoint.

For those stored netlists it records a structural class against the expert
``*_ref.json`` and, by simulating again, a port-labeled comparison and a
coarser rounding of the same curves. Coarser rounding is a sensitivity
table. It is not a new official judge.

The structural class is a Weisfeiler-Lehman invariant of the instance graph,
not a proof of isomorphism. ``structure_and_parameters_match`` means that
invariant, the external port names, and the settings after filling documented
primitive defaults all agree.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

from repo_paths import REPO

ROOT = REPO
PIC = ROOT / "third_party" / "PICBench" / "PICBench"
CASES = ROOT / "third_party" / "PICBench" / "testcases"
LOG = ROOT / "logs" / "picbench"
OUT_JSON = LOG / "mismatch_taxonomy.json"
OUT_CASES = LOG / "case_studies.json"
OUT_CSV = LOG / "case_studies.csv"
FIG_DIR = LOG / "audit_figures"

ARMS = (
    "qwen35-9b",
    "deepseek-v4-pro",
    "qwen3.6-27b",
    "glm-5.3",
    "deepseek-v4.1-flash",
    "qwen35-9b-visible-sft-epoch8",
    "qwen35-9b-reference-sft-epoch8",
)
FOCUS = (
    "MZI_ps",
    "WDM_mux",
    "WDM_demux",
    "Clements_8x8",
    "Reck_8x8",
    "Crossbar_8x8",
    "Spanke_8x8",
    "Benes_8x8",
    "Spanke–Benes_8x8",
)
PRECISIONS = (10, 6, 4, 3, 2)
DEFAULTS = {
    "straight": {"length": 10.0, "loss": 0.0, "neff": 2.34, "ng": 3.4, "wl0": 1.55},
    "straight_heat_metal": {
        "length": 10.0,
        "loss": 0.0,
        "neff": 2.34,
        "ng": 3.4,
        "wl0": 1.55,
        "phase_shift": 90.0,
    },
    "mrr": {"kappa": 0.3, "neff": 2.34, "alpha": 0.99, "cwl": 1.55},
}

_SAX = None
_JNP = None
_NP = None
_MODELS: dict = {}
_GOLDEN: dict = {}
_VERSIONS: dict = {}


def _num(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 10)
    return value


def _settings(raw, sax_name: str, fill: bool) -> tuple:
    data = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            data[str(key)] = _num(value)
    if fill:
        for key, value in DEFAULTS.get(sax_name, {}).items():
            data.setdefault(key, _num(value))
    return tuple(sorted(data.items()))


def _split_pin(token: str) -> tuple[str, str]:
    name, port = str(token).split(",", 1)
    return name.strip(), port.strip()


def _instances(doc: dict) -> list[tuple[str, str, tuple, tuple]]:
    models = doc["models"]
    rows = []
    for name, value in doc["netlist"]["instances"].items():
        if isinstance(value, str):
            component = value
            raw = {}
        elif isinstance(value, dict) and isinstance(value.get("component"), str):
            component = value["component"]
            raw = value.get("settings") or {}
        else:
            raise ValueError(f"bad instance {name}")
        if component not in models:
            raise KeyError(component)
        sax_name = models[component]
        rows.append((str(name), str(sax_name), _settings(raw, sax_name, False), _settings(raw, sax_name, True)))
    return rows


def _graph(doc: dict):
    rows = _instances(doc)
    sax_of = {name: sax_name for name, sax_name, _raw, _filled in rows}
    raw_of = {name: raw for name, _sax, raw, _filled in rows}
    filled_of = {name: filled for name, _sax, _raw, filled in rows}
    adj = {name: [] for name in sax_of}
    seen = set()
    duplicate = False
    for left, right in doc["netlist"].get("connections", {}).items():
        a_name, a_port = _split_pin(left)
        b_name, b_port = _split_pin(right)
        for name, port in ((a_name, a_port), (b_name, b_port)):
            if (name, port) in seen:
                duplicate = True
            seen.add((name, port))
        adj[a_name].append((a_port, b_name, b_port))
        adj[b_name].append((b_port, a_name, a_port))
    externals = []
    for external, binding in doc["netlist"].get("ports", {}).items():
        name, port = _split_pin(binding)
        externals.append((str(external), name, port))
        adj[name].append((port, "EXT", "DANGLING"))
    return sax_of, raw_of, filled_of, adj, externals, duplicate


def _colors(sax_of: dict, adj: dict) -> dict[str, int]:
    names = sorted(sax_of)
    palette = {label: index for index, label in enumerate(sorted(set(sax_of.values())))}
    colors = {name: palette[sax_of[name]] for name in names}
    for _ in range(len(names) + 2):
        signatures = {}
        for name in names:
            neigh = []
            for port, other, other_port in adj[name]:
                if other == "EXT":
                    neigh.append((port, other_port, -1))
                else:
                    neigh.append((port, other_port, colors[other]))
            signatures[name] = (colors[name], tuple(sorted(neigh)))
        ranked = {sig: index for index, sig in enumerate(sorted(set(signatures.values())))}
        refined = {name: ranked[signatures[name]] for name in names}
        if refined == colors:
            break
        colors = refined
    return colors


def _invariant(colors: dict[str, int], filled: dict[str, tuple], externals: list[tuple]) -> dict:
    anonymous = tuple(sorted((colors[name], port) for _ext, name, port in externals))
    named = tuple(sorted((ext, colors[name], port) for ext, name, port in externals))
    parameters = tuple(sorted((colors[name], filled[name]) for name in colors))
    return {
        "colors": tuple(sorted(Counter(colors.values()).items())),
        "anonymous_ports": anonymous,
        "named_ports": named,
        "parameters": parameters,
    }


def structural_class(candidate: dict, expert: dict) -> dict:
    try:
        c_sax, c_raw, c_filled, c_adj, c_ext, c_dup = _graph(candidate)
        e_sax, e_raw, e_filled, e_adj, e_ext, e_dup = _graph(expert)
    except KeyError as exc:
        return {"primary": "missing_model_binding", "detail": str(exc)}
    except (ValueError, AttributeError, TypeError) as exc:
        return {"primary": "unparsed", "detail": str(exc)}
    c_models = Counter(c_sax.values())
    e_models = Counter(e_sax.values())
    delta = []
    for key in sorted(set(c_models) | set(e_models)):
        diff = c_models[key] - e_models[key]
        if diff:
            delta.append(f"{diff:+d} {key}")
    base = {
        "model_delta": ", ".join(delta),
        "candidate_instances": len(c_sax),
        "expert_instances": len(e_sax),
        "duplicate_port": bool(c_dup or e_dup),
    }
    if c_models != e_models or c_dup:
        base["primary"] = "model_multiset_differs" if c_models != e_models else "duplicate_connection"
        return base
    c_inv = _invariant(_colors(c_sax, c_adj), c_filled, c_ext)
    e_inv = _invariant(_colors(e_sax, e_adj), e_filled, e_ext)
    if c_inv["colors"] != e_inv["colors"]:
        base["primary"] = "internal_topology_differs"
        return base
    if c_inv["anonymous_ports"] != e_inv["anonymous_ports"]:
        base["primary"] = "external_attachment_differs"
        return base
    if c_inv["named_ports"] != e_inv["named_ports"]:
        base["primary"] = "external_port_permutation"
        return base
    if c_inv["parameters"] != e_inv["parameters"]:
        base["primary"] = "parameters_differ"
        return base
    raw_c = tuple(sorted(c_raw.values()))
    raw_e = tuple(sorted(e_raw.values()))
    if raw_c != raw_e:
        base["primary"] = "explicit_default_only"
        return base
    base["primary"] = "structure_and_parameters_match"
    return base


def _setting_brief(doc: dict) -> list[str]:
    try:
        rows = _instances(doc)
    except (KeyError, ValueError, TypeError):
        return []
    if len(rows) > 12:
        return []
    brief = []
    for _name, sax_name, raw, _filled in rows:
        if raw:
            brief.append(f"{sax_name} {dict(raw)}")
        else:
            brief.append(sax_name)
    return sorted(brief)


def _bipartite_perfect(cost, threshold: float) -> bool:
    n = len(cost)
    adj = [[j for j in range(n) if cost[i][j] <= threshold] for i in range(n)]
    match_r = [-1] * n

    def dfs(left: int, seen: list[bool]) -> bool:
        for right in adj[left]:
            if seen[right]:
                continue
            seen[right] = True
            if match_r[right] == -1 or dfs(match_r[right], seen):
                match_r[right] = left
                return True
        return False

    matched = 0
    for left in range(n):
        if dfs(left, [False] * n):
            matched += 1
    return matched == n


def minimax_max_abs(left, right) -> float:
    """Smallest tau such that some bijection keeps every curve within tau."""
    import numpy as np

    if len(left) != len(right):
        raise ValueError("curve counts differ")
    if not left:
        return 0.0
    cost = []
    for row in left:
        row = np.asarray(row, dtype=np.float64)
        cost.append([float(np.max(np.abs(row - np.asarray(col, dtype=np.float64)))) for col in right])
    values = sorted({entry for line in cost for entry in line})
    lo = 0
    hi = len(values) - 1
    answer = values[-1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if _bipartite_perfect(cost, values[mid]):
            answer = values[mid]
            hi = mid - 1
        else:
            lo = mid + 1
    return float(answer)


def _rounded_set(curves: dict, precision: int):
    import numpy as np

    return frozenset(tuple(np.round(np.asarray(value, dtype=np.float64), precision).tolist()) for value in curves.values())


def _self_check() -> None:
    import numpy as np

    swapped = minimax_max_abs([np.array([0.0, 0.0]), np.array([1.0, 1.0])], [np.array([1.0, 1.0]), np.array([0.0, 0.0])])
    shifted = minimax_max_abs([np.array([0.0]), np.array([1.0])], [np.array([0.2]), np.array([1.2])])
    if swapped != 0.0 or abs(shifted - 0.2) > 1e-12:
        raise RuntimeError(f"minimax self-check failed: {swapped}, {shifted}")
    expert = {
        "netlist": {
            "instances": {"a": "mmi", "b": {"component": "wg", "settings": {"length": 20}}},
            "connections": {"a,O1": "b,I1"},
            "ports": {"I1": "a,I1", "O1": "b,O1"},
        },
        "models": {"mmi": "mmi1x2", "wg": "straight"},
    }
    renamed = {
        "netlist": {
            "instances": {"x": "core", "y": {"component": "delay", "settings": {"length": 20.0}}},
            "connections": {"x,O1": "y,I1"},
            "ports": {"I1": "x,I1", "O1": "y,O1"},
        },
        "models": {"core": "mmi1x2", "delay": "straight"},
    }
    if structural_class(renamed, expert)["primary"] != "structure_and_parameters_match":
        raise RuntimeError("renamed netlist was not recognized")
    extra = json.loads(json.dumps(expert))
    extra["netlist"]["instances"]["c"] = "wg"
    extra["netlist"]["connections"] = {"a,O1": "c,I1", "c,O1": "b,I1"}
    if structural_class(extra, expert)["primary"] != "model_multiset_differs":
        raise RuntimeError("extra device was not recognized")
    omitted = json.loads(json.dumps(expert))
    omitted["netlist"]["instances"]["b"] = "wg"
    expert_default = json.loads(json.dumps(expert))
    expert_default["netlist"]["instances"]["b"] = {"component": "wg", "settings": {"length": 10}}
    if structural_class(omitted, expert_default)["primary"] != "explicit_default_only":
        raise RuntimeError("explicit default was not recognized")
    longer = json.loads(json.dumps(expert))
    if structural_class(omitted, longer)["primary"] != "parameters_differ":
        raise RuntimeError("length change was not recognized")


def _init_worker() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.chdir(PIC)
    import sys

    sys.path.insert(0, str(PIC))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import sax
    from devices import (  # noqa: WPS433
        OSU,
        coupler,
        mmi1x2,
        mrr,
        mzi,
        mzi_2x2,
        mzi_ps,
        mzm,
        mzm_dual,
        straight,
        straight_heat_metal,
    )

    global _SAX, _JNP, _NP, _MODELS, _VERSIONS
    _SAX = sax
    _JNP = jnp
    _NP = np
    _VERSIONS = {"sax": getattr(sax, "__version__", "unknown"), "jax": getattr(jax, "__version__", "unknown")}
    _MODELS = {
        "OSU": OSU,
        "coupler": coupler,
        "mmi1x2": mmi1x2,
        "mrr": mrr,
        "mzi": mzi,
        "mzi_2x2": mzi_2x2,
        "mzi_ps": mzi_ps,
        "mzm": mzm,
        "mzm_dual": mzm_dual,
        "straight": straight,
        "straight_heat_metal": straight_heat_metal,
    }


def _golden(design: str):
    if design not in _GOLDEN:
        path = CASES / design / f"{design}_res.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        _GOLDEN[design] = {key: _NP.asarray(value, dtype=_NP.float64) for key, value in raw.items()}
    return _GOLDEN[design]


def _simulate(doc: dict) -> dict:
    model_data = {}
    for key, value in doc["models"].items():
        if value not in _MODELS:
            raise KeyError(value)
        model_data[key] = _MODELS[value]
    circuit, _info = _SAX.circuit(netlist=doc["netlist"], models=model_data)
    spectrum = circuit(wl=_JNP.linspace(1.51, 1.59, 1000))
    ports = doc["netlist"]["ports"]
    inputs = [key for key in ports if str(key).startswith("I")]
    outputs = [key for key in ports if str(key).startswith("O")]
    curves = {}
    for src in inputs:
        for dst in outputs:
            power = _NP.asarray(abs(spectrum[src, dst]) ** 2, dtype=_NP.float64)
            if not _NP.all(_NP.isfinite(power)):
                raise ValueError(f"non-finite S[{src},{dst}]")
            curves[f"{src}_{dst}"] = power
    return curves


def _simulate_task(task: dict) -> dict:
    try:
        doc = json.loads(task["netlist"])
        curves = _simulate(doc)
        golden = _golden(task["design"])
        same_keys = set(curves) == set(golden)
        metrics = {
            "ok": True,
            "versions": _VERSIONS,
            "n_curves": len(curves),
            "n_golden": len(golden),
            "port_keys_equal": same_keys,
            "set_equal": {},
            "labeled_equal": {},
        }
        for precision in PRECISIONS:
            metrics["set_equal"][str(precision)] = _rounded_set(curves, precision) == _rounded_set(golden, precision)
            if same_keys:
                metrics["labeled_equal"][str(precision)] = all(
                    tuple(_NP.round(curves[key], precision).tolist()) == tuple(_NP.round(golden[key], precision).tolist())
                    for key in golden
                )
            else:
                metrics["labeled_equal"][str(precision)] = None
        if len(curves) == len(golden) and len(curves) > 0:
            metrics["aligned_max_abs"] = minimax_max_abs(list(curves.values()), list(golden.values()))
        else:
            metrics["aligned_max_abs"] = None
        if same_keys and golden:
            metrics["labeled_max_abs"] = max(float(_NP.max(_NP.abs(curves[key] - golden[key]))) for key in golden)
        else:
            metrics["labeled_max_abs"] = None
        if task["save_curves"]:
            step = 5
            metrics["curves"] = {key: value[::step].tolist() for key, value in curves.items()}
            metrics["golden"] = {key: value[::step].tolist() for key, value in golden.items()}
        return metrics
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400], "versions": _VERSIONS}


def _official(verdict: str) -> str | None:
    if verdict.startswith("functional check passed"):
        return "match"
    if verdict.startswith("functional error"):
        return "mismatch"
    return None


def _iter_simulable(arm: str):
    """Yield design, sample filename, and official label for stored simulable netlists.

    Newer score files have verdict strings. Qwen3.5-9B and DeepSeek-V4-Pro
    were scored before that writer existed. Their labels come from
    failure_audit.json, which re-ran the judge, and are checked against the
    legacy bucket list.
    """
    summary_path = LOG / arm / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for design_row in summary["designs"]:
            for detail in design_row["details"]:
                label = _official(detail["verdict"] or "")
                if label is None:
                    continue
                yield design_row["design"], detail["sample"], label
        return
    score = json.loads((LOG / f"{arm}-score.json").read_text(encoding="utf-8"))
    audit = json.loads((LOG / "failure_audit.json").read_text(encoding="utf-8"))[arm]
    by_sample = {(row["design"], row["sample"]): row for row in audit["rows"]}
    for design_row in score["designs"]:
        design = design_row["design"]
        for index, bucket in enumerate(design_row["detail"]):
            official = by_sample[(design, index)]["official"]
            if bucket == "func":
                expected = "spectrum_match"
                label = "match"
            elif bucket == "syntax_only":
                expected = "simulation_valid_spectrum_differs"
                label = "mismatch"
            else:
                if official in {"spectrum_match", "simulation_valid_spectrum_differs"}:
                    raise RuntimeError(f"{arm} {design} s{index}: bucket {bucket} but audit {official}")
                continue
            if official != expected:
                raise RuntimeError(f"{arm} {design} s{index}: bucket {bucket} but audit {official}")
            yield design, f"s{index}.json", label


def _bucket(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value <= 1e-6:
        return "within_1e-6"
    if value <= 1e-4:
        return "within_1e-4"
    if value <= 1e-3:
        return "within_1e-3"
    if value <= 1e-2:
        return "within_1e-2"
    if value <= 1e-1:
        return "within_1e-1"
    return "above_1e-1"


def _collect() -> list[dict]:
    experts = {}
    tasks = []
    missing = []
    for arm in ARMS:
        for design, sample, label in _iter_simulable(arm):
            if design not in experts:
                ref = CASES / design / f"{design}_ref.json"
                if not ref.is_file():
                    raise FileNotFoundError(ref)
                experts[design] = json.loads(ref.read_text(encoding="utf-8"))
            path = LOG / arm / design / sample
            if not path.is_file():
                missing.append({"arm": arm, "design": design, "sample": sample, "official": label})
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            netlist = payload.get("netlist")
            if not isinstance(netlist, str) or not netlist.strip():
                missing.append({"arm": arm, "design": design, "sample": sample, "official": label, "reason": "no netlist string"})
                continue
            try:
                doc = json.loads(netlist)
            except json.JSONDecodeError as exc:
                missing.append({"arm": arm, "design": design, "sample": sample, "official": label, "reason": str(exc)})
                continue
            structure = structural_class(doc, experts[design])
            tasks.append(
                {
                    "arm": arm,
                    "design": design,
                    "sample": sample,
                    "official": label,
                    "evidence": str(path.relative_to(ROOT)),
                    "structure": structure,
                    "candidate_settings": _setting_brief(doc),
                    "expert_settings": _setting_brief(experts[design]),
                    "netlist": netlist,
                    "save_curves": design in {"MZI_ps", "WDM_mux", "WDM_demux"},
                }
            )
    if missing:
        raise RuntimeError(f"{len(missing)} simulable samples have no stored netlist: {missing[:3]}")
    return tasks


def _plot(rows: list[dict]) -> list[str]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    mismatches = [row for row in rows if row["official"] == "mismatch" and row.get("resim_agrees")]
    classes = [
        "model_multiset_differs",
        "duplicate_connection",
        "internal_topology_differs",
        "external_attachment_differs",
        "external_port_permutation",
        "parameters_differ",
        "explicit_default_only",
        "structure_and_parameters_match",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bottoms = [0] * len(ARMS)
    colors = ["#4E79A7", "#F2B56B", "#59A89C", "#E15759", "#B07AA1", "#EDC948", "#76B7B2", "#BAB0AC"]
    for label, color in zip(classes, colors):
        heights = [sum(row["primary"] == label for row in mismatches if row["arm"] == arm) for arm in ARMS]
        if sum(heights) == 0:
            continue
        ax.barh(list(ARMS), heights, left=bottoms, color=color, label=label)
        bottoms = [a + b for a, b in zip(bottoms, heights)]
    ax.set_xlabel("Official spectrum mismatches")
    ax.set_title("Structural class of simulation-valid mismatches")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = FIG_DIR / "structural_class.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    written.append(str(path.relative_to(ROOT)))

    values = [row["aligned_max_abs"] for row in mismatches if isinstance(row.get("aligned_max_abs"), float)]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if values:
        ax.hist([value if value > 0 else 1e-16 for value in values], bins=[1e-16, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0], color="#4E79A7")
        ax.set_xscale("log")
    ax.set_xlabel("Minimax max-|ΔT| after curve alignment")
    ax.set_ylabel("Mismatch samples")
    ax.set_title("Spectral distance of official mismatches")
    fig.tight_layout()
    path = FIG_DIR / "spectrum_distance.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    written.append(str(path.relative_to(ROOT)))

    mzi = [row for row in rows if row["design"] == "MZI_ps" and row.get("curves")]
    if mzi:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        golden = mzi[0]["golden"]
        key = sorted(golden)[0]
        xs = list(range(0, 1000, 5))
        wavelengths = [1.51 + (1.59 - 1.51) * i / 999 for i in xs]
        ax.plot(wavelengths, golden[key], color="black", linewidth=2.0, label="expert")
        drawn = set()
        for row in mzi:
            if row["official"] != "mismatch" or row["arm"] in drawn:
                continue
            if key not in row["curves"]:
                continue
            ax.plot(wavelengths, row["curves"][key], linewidth=1.2, label=row["arm"])
            drawn.add(row["arm"])
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel(key + " power")
        ax.set_title("MZI_ps, one official mismatch per arm")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = FIG_DIR / "mzi_ps_spectra.pdf"
        fig.savefig(path)
        fig.savefig(path.with_suffix(".png"), dpi=200)
        plt.close(fig)
        written.append(str(path.relative_to(ROOT)))

    focus = [row for row in rows if row["design"] in FOCUS and isinstance(row.get("aligned_max_abs"), float)]
    if focus:
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        designs = [name for name in FOCUS if any(row["design"] == name for row in focus)]
        for arm_index, arm in enumerate(ARMS):
            ys = []
            xs = []
            for design_index, design in enumerate(designs):
                for row in focus:
                    if row["arm"] == arm and row["design"] == design and row["official"] == "mismatch":
                        xs.append(design_index + (arm_index - 3) * 0.08)
                        ys.append(row["aligned_max_abs"])
            if ys:
                ax.scatter(xs, ys, s=18, label=arm)
        ax.set_xticks(range(len(designs)))
        ax.set_xticklabels(designs, rotation=30, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("Minimax max-|ΔT|")
        ax.set_title("Case-study mismatches")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        path = FIG_DIR / "case_study_distance.pdf"
        fig.savefig(path)
        fig.savefig(path.with_suffix(".png"), dpi=200)
        plt.close(fig)
        written.append(str(path.relative_to(ROOT)))
    return written


def _summarize(rows: list[dict]) -> dict:
    def agreed(official: str) -> list[dict]:
        return [row for row in rows if row["official"] == official and row.get("resim_agrees")]

    def count_true(items: list[dict], field: str, precision: str) -> int:
        return sum(bool(row.get(field, {}).get(precision)) for row in items)

    mismatches = agreed("mismatch")
    matches = agreed("match")
    by_arm = {}
    for arm in ARMS:
        arm_rows = [row for row in mismatches if row["arm"] == arm]
        by_arm[arm] = {
            "mismatches": len(arm_rows),
            "primary": dict(Counter(row["primary"] for row in arm_rows)),
            "spectral_bucket": dict(Counter(row["spectral_bucket"] for row in arm_rows)),
            "set_equal_6": count_true(arm_rows, "set_equal", "6"),
            "set_equal_4": count_true(arm_rows, "set_equal", "4"),
            "set_equal_3": count_true(arm_rows, "set_equal", "3"),
            "set_equal_2": count_true(arm_rows, "set_equal", "2"),
            "labeled_equal_10": count_true(arm_rows, "labeled_equal", "10"),
            "port_keys_differ": sum(not row.get("port_keys_equal") for row in arm_rows),
        }
    focus = {}
    for design in FOCUS:
        design_rows = [row for row in rows if row["design"] == design]
        focus[design] = {
            "samples": len(design_rows),
            "matches": sum(row["official"] == "match" for row in design_rows),
            "mismatches": sum(row["official"] == "mismatch" for row in design_rows),
            "primary_on_mismatches": dict(
                Counter(row["primary"] for row in design_rows if row["official"] == "mismatch" and row.get("resim_agrees"))
            ),
        }
    return {
        "simulable_samples": len(rows),
        "resim_agrees": sum(bool(row.get("resim_agrees")) for row in rows),
        "resim_disagrees": sum(row.get("resim_status") == "disagrees" for row in rows),
        "resim_error": sum(row.get("resim_status") == "error" for row in rows),
        "mismatches_agreed": len(mismatches),
        "matches_agreed": len(matches),
        "passes_that_fail_labeled_10": sum(
            row.get("port_keys_equal") and row.get("labeled_equal", {}).get("10") is False for row in matches
        ),
        "passes_with_different_port_keys": sum(not row.get("port_keys_equal") for row in matches),
        "by_arm": by_arm,
        "focus": focus,
        "primary_all_mismatches": dict(Counter(row["primary"] for row in mismatches)),
        "spectral_bucket_all_mismatches": dict(Counter(row["spectral_bucket"] for row in mismatches)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    _self_check()
    tasks = _collect()
    print(f"simulable netlists {len(tasks)}", flush=True)
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(processes=args.workers, initializer=_init_worker) as pool:
        payload = [
            {"design": task["design"], "netlist": task["netlist"], "save_curves": task["save_curves"]}
            for task in tasks
        ]
        for index, metrics in enumerate(pool.imap(_simulate_task, payload), start=1):
            results.append(metrics)
            if index % 25 == 0 or index == len(tasks):
                print(f"simulated {index}/{len(tasks)}", flush=True)
    rows = []
    versions = {}
    for task, metrics in zip(tasks, results):
        row = {
            "arm": task["arm"],
            "design": task["design"],
            "sample": task["sample"],
            "official": task["official"],
            "evidence": task["evidence"],
            "primary": task["structure"].get("primary"),
            "model_delta": task["structure"].get("model_delta", ""),
            "candidate_instances": task["structure"].get("candidate_instances"),
            "expert_instances": task["structure"].get("expert_instances"),
            "candidate_settings": task["candidate_settings"],
            "expert_settings": task["expert_settings"],
        }
        if metrics.get("versions"):
            versions = metrics["versions"]
        if not metrics.get("ok"):
            row["resim_status"] = "error"
            row["resim_agrees"] = False
            row["error"] = metrics.get("error")
            row["spectral_bucket"] = "resim_error"
            rows.append(row)
            continue
        set10 = bool(metrics["set_equal"]["10"])
        agrees = (set10 and task["official"] == "match") or ((not set10) and task["official"] == "mismatch")
        row.update(
            {
                "resim_status": "agrees" if agrees else "disagrees",
                "resim_agrees": agrees,
                "n_curves": metrics["n_curves"],
                "n_golden": metrics["n_golden"],
                "port_keys_equal": metrics["port_keys_equal"],
                "set_equal": metrics["set_equal"],
                "labeled_equal": metrics["labeled_equal"],
                "aligned_max_abs": metrics["aligned_max_abs"],
                "labeled_max_abs": metrics["labeled_max_abs"],
                "spectral_bucket": "set_equal_10" if set10 else _bucket(metrics["aligned_max_abs"]),
            }
        )
        if task["save_curves"] and "curves" in metrics:
            row["curves"] = metrics["curves"]
            row["golden"] = metrics["golden"]
        rows.append(row)
    figures = _plot(rows)
    for row in rows:
        row.pop("curves", None)
        row.pop("golden", None)
    summary = _summarize(rows)
    report = {
        "official_criterion_unchanged": True,
        "checkpoint_selection": "not used",
        "judge_resim_versions": versions,
        "rounding": "numpy round, same precisions as a set comparison; 10 decimals remains the official functional test",
        "structural_note": (
            "Classes use a Weisfeiler-Lehman invariant plus external port names and settings. "
            "Primitive defaults are straight length 10, straight_heat_metal phase_shift 90, and the mrr defaults in devices.py. "
            "A match of this invariant is not a formal isomorphism proof."
        ),
        "summary": summary,
        "figures": figures,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    focus_rows = [row for row in rows if row["design"] in FOCUS]
    OUT_CASES.write_text(json.dumps(focus_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "arm",
                "design",
                "sample",
                "official",
                "primary",
                "model_delta",
                "candidate_instances",
                "expert_instances",
                "aligned_max_abs",
                "labeled_max_abs",
                "spectral_bucket",
                "port_keys_equal",
                "set_equal_6",
                "set_equal_4",
                "set_equal_2",
                "labeled_equal_10",
                "resim_status",
                "evidence",
            ],
        )
        writer.writeheader()
        for row in focus_rows:
            writer.writerow(
                {
                    "arm": row["arm"],
                    "design": row["design"],
                    "sample": row["sample"],
                    "official": row["official"],
                    "primary": row["primary"],
                    "model_delta": row.get("model_delta", ""),
                    "candidate_instances": row.get("candidate_instances"),
                    "expert_instances": row.get("expert_instances"),
                    "aligned_max_abs": row.get("aligned_max_abs"),
                    "labeled_max_abs": row.get("labeled_max_abs"),
                    "spectral_bucket": row.get("spectral_bucket"),
                    "port_keys_equal": row.get("port_keys_equal"),
                    "set_equal_6": (row.get("set_equal") or {}).get("6"),
                    "set_equal_4": (row.get("set_equal") or {}).get("4"),
                    "set_equal_2": (row.get("set_equal") or {}).get("2"),
                    "labeled_equal_10": (row.get("labeled_equal") or {}).get("10"),
                    "resim_status": row.get("resim_status"),
                    "evidence": row["evidence"],
                }
            )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
