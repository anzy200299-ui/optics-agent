#!/usr/bin/env python3
"""Recompute the numbers cited in paper/main.tex from stored logs.

This script does not rescore SAX. It reads the saved PICBench score files,
the failure-stage audit, the teacher-filter report, the question JSONL, the
expert netlists, and the NetlistBench oracle reports. It writes
logs/paper_consistency.json.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

from repo_paths import PICBENCH_TRAJ_DIR, REPO

ROOT = REPO
SAX_NAMES = (
    "straight_heat_metal",
    "mmi1x2",
    "mzm_dual",
    "mzi_2x2",
    "mzi_ps",
    "straight",
    "coupler",
    "mzm",
    "mzi",
    "mrr",
    "OSU",
)
FAMILIES = {
    "Clements_4x4": "Computing",
    "Clements_8x8": "Computing",
    "Reck_4x4": "Computing",
    "Reck_8x8": "Computing",
    "NLS": "Computing",
    "U-matrix_block": "Computing",
    "Direct modulator": "Interconnect",
    "QPSK modulator": "Interconnect",
    "8-QAM modulator": "Interconnect",
    "64-QAM modulator": "Interconnect",
    "WDM_mux": "Interconnect",
    "WDM_demux": "Interconnect",
    "Optical hybrid": "Interconnect",
    "OS_2x2": "Switch",
    "Crossbar_4x4": "Switch",
    "Crossbar_8x8": "Switch",
    "Spanke_4x4": "Switch",
    "Spanke_8x8": "Switch",
    "Benes_4x4": "Switch",
    "Benes_8x8": "Switch",
    "Spanke–Benes_4x4": "Switch",
    "Spanke–Benes_8x8": "Switch",
    "MZM": "Devices",
    "MZI_ps": "Devices",
}
WIDTH_PAIRS = [
    ("Clements", "Clements_4x4", "Clements_8x8"),
    ("Reck", "Reck_4x4", "Reck_8x8"),
    ("Crossbar", "Crossbar_4x4", "Crossbar_8x8"),
    ("Spanke", "Spanke_4x4", "Spanke_8x8"),
    ("Benes", "Benes_4x4", "Benes_8x8"),
    ("Spanke--Benes", "Spanke–Benes_4x4", "Spanke–Benes_8x8"),
]
MESH12 = [
    "Clements_4x4",
    "Clements_8x8",
    "Reck_4x4",
    "Reck_8x8",
    "Crossbar_4x4",
    "Crossbar_8x8",
    "Spanke_4x4",
    "Spanke_8x8",
    "Benes_4x4",
    "Benes_8x8",
    "Spanke–Benes_4x4",
    "Spanke–Benes_8x8",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pass_at_k(counts: list[int], n: int, k: int) -> float:
    total = 0.0
    for c in counts:
        if n - c < k:
            ratio = 0.0
        else:
            ratio = math.comb(n - c, k) / math.comb(n, k)
        total += 1.0 - ratio
    return total / len(counts)


def component(value) -> str:
    if isinstance(value, str):
        return value
    return value["component"]


def fingerprint(doc: dict) -> tuple:
    net = doc["netlist"]
    models = doc["models"]
    instances = net["instances"]
    sax_of = {}
    counts: Counter[str] = Counter()
    for name, value in instances.items():
        key = component(value)
        sax = models[key]
        counts[sax] += 1
        sax_of[name] = (sax, counts[sax])
    edges = []
    for left, right in net["connections"].items():
        a_name, a_port = left.split(",", 1)
        b_name, b_port = right.split(",", 1)
        edges.append((sax_of[a_name], a_port, sax_of[b_name], b_port))
    ports = []
    for external, binding in net["ports"].items():
        name, port = binding.split(",", 1)
        ports.append((str(external)[:1], sax_of[name], port))
    return (tuple(sorted(counts.items())), tuple(sorted(edges)), tuple(sorted(ports)))


def settings_signature(doc: dict) -> tuple:
    net = doc["netlist"]
    models = doc["models"]
    rows = []
    for value in net["instances"].values():
        if isinstance(value, str):
            rows.append((models[value], ()))
        else:
            settings = value.get("settings") or {}
            rows.append((models[value["component"]], tuple(sorted((str(k), str(v)) for k, v in settings.items()))))
    return tuple(sorted(rows))


def graph_stats(doc: dict) -> dict:
    net = doc["netlist"]
    models = doc["models"]
    sax = []
    for value in net["instances"].values():
        key = value if isinstance(value, str) else value["component"]
        sax.append(models[key])
    return {
        "instances": len(net["instances"]),
        "connections": len(net["connections"]),
        "ports": len(net["ports"]),
        "sax_types": len(set(sax)),
        "json_chars": len(json.dumps(doc, separators=(",", ":"))),
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    index = (len(sorted_values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] * (high - index) + sorted_values[high] * (index - low)


def bootstrap(design_counts: list[int], n_draw: int = 5, reps: int = 10000, seed: int = 20261021) -> dict:
    import random

    rng = random.Random(seed)
    size = len(design_counts)
    at1 = []
    at5 = []
    for _ in range(reps):
        sample = [design_counts[rng.randrange(size)] for _ in range(size)]
        at1.append(sum(sample) / (size * n_draw))
        at5.append(sum(1 for c in sample if c > 0) / size)
    at1.sort()
    at5.sort()
    return {
        "reps": reps,
        "seed": seed,
        "unit": "design resample with replacement",
        "pass1_point": sum(design_counts) / (size * n_draw),
        "pass5_point": sum(1 for c in design_counts if c > 0) / size,
        "pass1_p2.5": percentile(at1, 0.025),
        "pass1_p97.5": percentile(at1, 0.975),
        "pass5_p2.5": percentile(at5, 0.025),
        "pass5_p97.5": percentile(at5, 0.975),
    }


def quantiles(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "mean": round(sum(ordered) / len(ordered), 2),
        "max": ordered[-1],
    }


def audit_picbench_files(model: str) -> dict:
    root = ROOT / "logs" / "picbench" / model
    rows = []
    for path in sorted(root.glob("*/*.json")):
        payload = load_json(path)
        content = payload.get("content") or ""
        reasoning = payload.get("reasoning_content") or ""
        response = payload.get("response") or ""
        netlist = payload.get("netlist") or ""
        rows.append(
            {
                "design": payload.get("design") or path.parent.name,
                "sample": payload.get("sample"),
                "finish": payload.get("finish_reason"),
                "content_has_result": "<result>" in content,
                "reasoning_has_result": "<result>" in reasoning,
                "response_has_result": "<result>" in response,
                "response_is_content": response == content,
                "response_is_reasoning": bool(reasoning) and response == reasoning,
                "netlist_nonempty": bool(str(netlist).strip()),
                "completion_tokens": (payload.get("usage") or {}).get("completion_tokens"),
                "reasoning_tokens": ((payload.get("usage") or {}).get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                ),
            }
        )
    return {
        "n": len(rows),
        "designs": len({row["design"] for row in rows}),
        "per_design": dict(Counter(row["design"] for row in rows)),
        "content_has_result": sum(row["content_has_result"] for row in rows),
        "reasoning_has_result": sum(row["reasoning_has_result"] for row in rows),
        "response_has_result": sum(row["response_has_result"] for row in rows),
        "response_is_reasoning": sum(row["response_is_reasoning"] for row in rows),
        "netlist_nonempty": sum(row["netlist_nonempty"] for row in rows),
        "fallback_rows": [
            row
            for row in rows
            if (not row["content_has_result"]) or row["response_is_reasoning"] or (not row["netlist_nonempty"])
        ],
    }


def stage_totals(audit_model: dict) -> dict:
    official = Counter(row["official"] for row in audit_model["rows"])
    repaired = Counter(row["repaired"] for row in audit_model["rows"])
    return {"official": dict(official), "repaired": dict(repaired), "n": len(audit_model["rows"])}


def score_from_designs(score: dict) -> dict:
    syntax = [row["syntax"] for row in score["designs"]]
    function = [row["func"] for row in score["designs"]]
    return {
        "n_designs": len(score["designs"]),
        "samples_each": sorted({row["n"] for row in score["designs"]}),
        "syntax_samples": sum(syntax),
        "func_samples": sum(function),
        "syntax_pass1": pass_at_k(syntax, 5, 1),
        "func_pass1": pass_at_k(function, 5, 1),
        "syntax_pass5": pass_at_k(syntax, 5, 5),
        "func_pass5": pass_at_k(function, 5, 5),
        "file_syntax_at1": score["syntax_at1"],
        "file_func_at1": score["func_at1"],
        "file_syntax_at5": score["syntax_at5"],
        "file_func_at5": score["func_at5"],
    }


def family_table(score: dict) -> dict:
    syn = Counter()
    func = Counter()
    denom = Counter()
    by_name = {row["design"]: row for row in score["designs"]}
    for design, family in FAMILIES.items():
        row = by_name[design]
        syn[family] += row["syntax"]
        func[family] += row["func"]
        denom[family] += row["n"]
    return {
        family: {"syntax": syn[family], "func": func[family], "n": denom[family]}
        for family in ("Computing", "Interconnect", "Switch", "Devices")
    }


def width_table(score: dict) -> dict:
    by_name = {row["design"]: row for row in score["designs"]}
    rows = {}
    for label, a, b in WIDTH_PAIRS:
        rows[label] = {
            "4x4": [by_name[a]["syntax"], by_name[a]["func"]],
            "8x8": [by_name[b]["syntax"], by_name[b]["func"]],
        }
    return rows


def mesh_load_failures(audit_model: dict) -> int:
    return sum(
        1
        for row in audit_model["rows"]
        if row["design"] in MESH12 and row["official"] in {"extraction_failure", "json_parse_failure"}
    )


def question_audit() -> dict:
    rows = [json.loads(line) for line in (ROOT / "data/weakness_questions/questions.jsonl").read_text().splitlines() if line.strip()]
    loci = Counter(row["locus"] for row in rows)
    templates = Counter(row["template"] for row in rows)
    exposure = Counter()
    by_locus_explicit = Counter()
    stats = []
    fps = []
    sigs = []
    widths = Counter()
    for row in rows:
        text = row["question"]
        doc = row["reference"]
        named = [name for name in SAX_NAMES if re.search(rf"\b{re.escape(name)}\b", text)]
        has_map = "Map " in text or "models entry" in text
        has_connect = "connect " in text.lower() or "Connect " in text
        has_bind = "Bind circuit port" in text or "bound to" in text
        has_settings = bool(re.search(r"length|phase_shift|cwl", text))
        explicit = bool(named) and has_map and (has_connect or has_bind)
        kind = "explicit_netlist_transcription" if explicit else "functional_specification"
        exposure[kind] += 1
        if named:
            exposure["names_sax_model"] += 1
        if has_map:
            exposure["states_model_map"] += 1
        if has_connect:
            exposure["states_connections"] += 1
        if has_bind:
            exposure["states_ports"] += 1
        if has_settings:
            exposure["states_settings"] += 1
        by_locus_explicit[row["locus"]] += int(explicit)
        stat = graph_stats(doc)
        stat["locus"] = row["locus"]
        stat["prompt_chars"] = len(text)
        stat["prompt_words"] = len(re.findall(r"[A-Za-z0-9_]+", text))
        stats.append(stat)
        fps.append(fingerprint(doc))
        sigs.append(settings_signature(doc))
        if "width" in (row.get("spec") or {}):
            widths[row["spec"]["width"]] += 1
    fp_counts = Counter(fps)
    repeated = sum(1 for count in fp_counts.values() if count > 1)
    by_locus = {}
    for locus in loci:
        subset = [stat for stat in stats if stat["locus"] == locus]
        by_locus[locus] = {
            "n": len(subset),
            "instances": quantiles([stat["instances"] for stat in subset]),
            "connections": quantiles([stat["connections"] for stat in subset]),
            "sax_types": quantiles([stat["sax_types"] for stat in subset]),
            "json_chars": quantiles([stat["json_chars"] for stat in subset]),
            "prompt_words": quantiles([stat["prompt_words"] for stat in subset]),
            "explicit": by_locus_explicit[locus],
        }
    shortest = min(rows, key=lambda row: len(row["question"]))
    longest = max(rows, key=lambda row: (graph_stats(row["reference"])["instances"], len(row["question"])))
    return {
        "n": len(rows),
        "loci": dict(loci),
        "templates": dict(templates),
        "exposure": dict(exposure),
        "widths": dict(widths),
        "unique_fingerprints": len(fp_counts),
        "fingerprints_with_copies": repeated,
        "max_fingerprint_copies": max(fp_counts.values()),
        "unique_settings_signatures": len(set(sigs)),
        "by_locus": by_locus,
        "overall": {
            "instances": quantiles([stat["instances"] for stat in stats]),
            "connections": quantiles([stat["connections"] for stat in stats]),
            "sax_types": quantiles([stat["sax_types"] for stat in stats]),
            "json_chars": quantiles([stat["json_chars"] for stat in stats]),
            "prompt_words": quantiles([stat["prompt_words"] for stat in stats]),
        },
        "examples": {
            "shortest_id": shortest["id"],
            "shortest_locus": shortest["locus"],
            "shortest_words": len(re.findall(r"[A-Za-z0-9_]+", shortest["question"])),
            "longest_id": longest["id"],
            "longest_locus": longest["locus"],
            "longest_instances": graph_stats(longest["reference"])["instances"],
        },
        "_fps": fps,
        "_sigs": sigs,
        "_rows": rows,
    }


def expert_audit(train: dict) -> dict:
    experts = []
    for path in sorted((ROOT / "third_party/PICBench/testcases").glob("*/*_ref.json")):
        doc = load_json(path)
        stat = graph_stats(doc)
        stat["design"] = path.parent.name
        stat["prompt_words"] = len(
            re.findall(
                r"[A-Za-z0-9_]+",
                (path.parent / f"{path.parent.name}_problem_description.txt").read_text(encoding="utf-8"),
            )
        )
        text = (path.parent / f"{path.parent.name}_problem_description.txt").read_text(encoding="utf-8")
        stat["names_sax"] = any(re.search(rf"\b{re.escape(name)}\b", text) for name in SAX_NAMES)
        stat["fp"] = fingerprint(doc)
        stat["sig"] = settings_signature(doc)
        experts.append(stat)
    train_fp = set(train["_fps"])
    train_model = {fp[0] for fp in train["_fps"]}
    train_edges = {fp[1] for fp in train["_fps"]}
    train_sig = set(train["_sigs"])
    exact = []
    model_only = []
    edge_only = []
    sig_hits = []
    for expert in experts:
        if expert["fp"] in train_fp:
            exact.append(expert["design"])
        if expert["fp"][0] in train_model:
            model_only.append(expert["design"])
        if expert["fp"][1] in train_edges:
            edge_only.append(expert["design"])
        if expert["sig"] in train_sig:
            sig_hits.append(expert["design"])
    return {
        "n": len(experts),
        "names_sax_in_problem_text": sum(expert["names_sax"] for expert in experts),
        "instances": quantiles([expert["instances"] for expert in experts]),
        "connections": quantiles([expert["connections"] for expert in experts]),
        "sax_types": quantiles([expert["sax_types"] for expert in experts]),
        "json_chars": quantiles([expert["json_chars"] for expert in experts]),
        "prompt_words": quantiles([expert["prompt_words"] for expert in experts]),
        "exact_fingerprint_overlap": exact,
        "same_model_multiset_as_some_train_item": model_only,
        "same_renamed_edges_as_some_train_item": edge_only,
        "same_settings_signature_as_some_train_item": sig_hits,
    }


def teacher_audit() -> dict:
    report = load_json(ROOT / "logs/sft/picbench_glm53_visible_filter.json")
    rows = report["rows"]
    counts = Counter(row["verdict"] for row in rows)
    by_locus = {}
    added = Counter()
    for row in rows:
        slot = by_locus.setdefault(row["locus"], Counter())
        slot[row["verdict"]] += 1
        if row["verdict"] == "func_repaired":
            added[tuple(row["added_models"])] += 1
    graph_changed = 0
    models_only = 0
    checked = 0
    missing_raw = []
    from filter_picbench_teacher import component_names, locate_result

    builtins = {
        "OSU",
        "coupler",
        "mmi1x2",
        "mrr",
        "mzi",
        "mzi_2x2",
        "mzi_ps",
        "mzm",
        "mzm_dual",
        "straight",
        "straight_heat_metal",
    }
    for row in rows:
        if row["verdict"] != "func_repaired":
            continue
        path = PICBENCH_TRAJ_DIR / f"{row['id']}.json"
        if not path.exists():
            missing_raw.append(row["id"])
            continue
        payload = load_json(path)
        checked += 1
        located = locate_result(payload.get("content") or "")
        if located is None:
            graph_changed += 1
            continue
        doc = located[0]
        present: list[str] = []
        for name in component_names(doc["netlist"]):
            if name not in doc["models"] and name not in present and name in builtins:
                present.append(name)
        netlist_untouched = True
        if present == list(row["added_models"]) and present:
            models_only += 1
        else:
            graph_changed += 1
            netlist_untouched = False
        del netlist_untouched
    jsonl = ROOT.parents[0] / "picbench_glm53_trajectories" / "picbench_glm53_visible.jsonl"
    # PICBENCH path is /mnt/oss/yanzhengan/picbench_glm53_trajectories
    jsonl = PICBENCH_TRAJ_DIR.parent / "picbench_glm53_visible.jsonl"
    n_jsonl = sum(1 for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()) if jsonl.exists() else None
    wrapped = 0
    if jsonl.exists():
        first = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
        roles = [message.get("role") for message in first["messages"]]
        wrapped = int("<result>" in first["messages"][-1]["content"])
        role_pattern = roles
    else:
        role_pattern = None
    return {
        "n_scored": report["n_scored"],
        "n_kept": report["n_kept"],
        "counts": dict(counts),
        "by_locus": {key: dict(value) for key, value in by_locus.items()},
        "added_model_patterns": {",".join(key): value for key, value in added.items()},
        "repair_files_checked": checked,
        "repair_models_only": models_only,
        "repair_not_models_only": graph_changed,
        "missing_raw": missing_raw[:10],
        "jsonl_rows": n_jsonl,
        "jsonl_assistant_has_result": wrapped,
        "jsonl_roles": role_pattern,
        "reference_side_errors": counts.get("error", 0),
        "prefix_mismatches": report["prefix_agreement"]["n_mismatch"],
    }


def device_add_split(path: Path) -> dict:
    report = load_json(path)
    order = 0
    value = 0
    other = 0
    for row in report["failures"]:
        details = row["diagnostics"][0].get("details") or {}
        expected = details.get("expected") or {}
        actual = details.get("actual") or {}
        if not expected or not actual:
            other += 1
            continue
        same_params = expected.get("params") == actual.get("params") and expected.get("model") == actual.get("model")
        same_nodes_as_set = sorted(expected.get("nodes") or []) == sorted(actual.get("nodes") or [])
        same_nodes_as_list = (expected.get("nodes") or []) == (actual.get("nodes") or [])
        if same_params and same_nodes_as_set and not same_nodes_as_list:
            order += 1
        elif same_nodes_as_list and not same_params:
            value += 1
        else:
            other += 1
    return {"failures": len(report["failures"]), "terminal_order": order, "value_only": value, "other": other}


def netlistbench() -> dict:
    out = {}
    for model in ("qwen35-9b", "deepseek-v4-pro"):
        score = load_json(ROOT / "logs/netlistbench" / f"{model}-score.json")
        overall = load_json(ROOT / "logs/netlistbench" / model / "edit/overall_report.json")
        families = {row["family"]: row for row in score["edit"]["families"]}
        family_sum_n = sum(row["total"] for row in families.values())
        family_sum_pass = sum(row["passed"] for row in families.values())
        manip = {key: families[key] for key in families if key != "equivalence_judgment_mixed"}
        out[model] = {
            "edit_root_n": score["edit"]["total"],
            "edit_root_pass": score["edit"]["passed"],
            "overall_total": overall.get("summary", overall.get("total_cases")),
            "manipulation_n": sum(row["total"] for row in manip.values()),
            "manipulation_pass": sum(row["passed"] for row in manip.values()),
            "equivalence_pass": families["equivalence_judgment_mixed"]["passed"],
            "family_n_sum": family_sum_n,
            "family_pass_sum": family_sum_pass,
            "buckets": score["edit"]["bucket_summary"],
            "bucket_pass_sum": sum(bucket["passed"] for bucket in score["edit"]["bucket_summary"].values()),
            "first_error": score["edit"]["first_error_code"],
            "first_error_sum": sum(score["edit"]["first_error_code"].values()),
            "recognition_flat": score["recognition"]["flat"]["pass"],
            "recognition_subckt": score["recognition"]["subckt"]["pass"],
            "recognition_sum": score["recognition"]["flat"]["pass"] + score["recognition"]["subckt"]["pass"],
            "flat_format_ok": score["recognition"]["flat"]["output_format"]["ok"],
            "subckt_format_ok": score["recognition"]["subckt"]["output_format"]["ok"],
            "device_add": device_add_split(ROOT / "logs/netlistbench" / model / "edit/device_add/report.json"),
        }
    questions = [
        json.loads(line)
        for line in (ROOT / "data/netlistbench_questions/questions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    out["synthetic"] = {
        "n": len(questions),
        "loci": dict(Counter(row["locus"] for row in questions)),
    }
    return out


def main() -> None:
    qwen_score = load_json(ROOT / "logs/picbench/qwen35-9b-score.json")
    deep_score = load_json(ROOT / "logs/picbench/deepseek-v4-pro-score.json")
    audit = load_json(ROOT / "logs/picbench/failure_audit.json")
    questions = question_audit()
    experts = expert_audit(questions)
    questions.pop("_fps")
    questions.pop("_sigs")
    questions.pop("_rows")
    report = {
        "qwen_files": audit_picbench_files("qwen35-9b"),
        "deepseek_files": audit_picbench_files("deepseek-v4-pro"),
        "qwen_stages": stage_totals(audit["qwen35-9b"]),
        "deepseek_stages": stage_totals(audit["deepseek-v4-pro"]),
        "qwen_pass": score_from_designs(qwen_score),
        "deepseek_pass": score_from_designs(deep_score),
        "qwen_family": family_table(qwen_score),
        "deepseek_family": family_table(deep_score),
        "deepseek_width": width_table(deep_score),
        "qwen_mesh_load_failures": mesh_load_failures(audit["qwen35-9b"]),
        "qwen_bootstrap_syntax": bootstrap([row["syntax"] for row in qwen_score["designs"]]),
        "qwen_bootstrap_func": bootstrap([row["func"] for row in qwen_score["designs"]]),
        "deepseek_bootstrap_syntax": bootstrap([row["syntax"] for row in deep_score["designs"]]),
        "deepseek_bootstrap_func": bootstrap([row["func"] for row in deep_score["designs"]]),
        "questions": questions,
        "experts": experts,
        "teacher": teacher_audit(),
        "netlistbench": netlistbench(),
    }
    dest = ROOT / "logs" / "paper_consistency.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    brief = {
        "deepseek_content_result": report["deepseek_files"]["content_has_result"],
        "deepseek_reasoning_result": report["deepseek_files"]["reasoning_has_result"],
        "deepseek_response_is_reasoning": report["deepseek_files"]["response_is_reasoning"],
        "deepseek_fallback": report["deepseek_files"]["fallback_rows"],
        "deepseek_stages": report["deepseek_stages"]["official"],
        "qwen_stages": report["qwen_stages"]["official"],
        "qwen_pass": report["qwen_pass"],
        "deepseek_pass": report["deepseek_pass"],
        "exposure": report["questions"]["exposure"],
        "question_overall": report["questions"]["overall"],
        "widths": report["questions"]["widths"],
        "unique_fp": report["questions"]["unique_fingerprints"],
        "fp_copies": report["questions"]["fingerprints_with_copies"],
        "experts": {k: report["experts"][k] for k in report["experts"] if k != "instances"},
        "expert_size": report["experts"]["instances"],
        "teacher_counts": report["teacher"]["counts"],
        "teacher_repair": {
            "checked": report["teacher"]["repair_files_checked"],
            "models_only": report["teacher"]["repair_models_only"],
            "not_models_only": report["teacher"]["repair_not_models_only"],
            "patterns": report["teacher"]["added_model_patterns"],
            "jsonl": report["teacher"]["jsonl_rows"],
            "reference_errors": report["teacher"]["reference_side_errors"],
        },
        "qwen_boot_func": report["qwen_bootstrap_func"],
        "deep_boot_func": report["deepseek_bootstrap_func"],
        "netlist_qwen_add": report["netlistbench"]["qwen35-9b"]["device_add"],
        "netlist_qwen": {
            k: report["netlistbench"]["qwen35-9b"][k]
            for k in ("edit_root_pass", "manipulation_pass", "equivalence_pass", "recognition_sum", "first_error_sum")
        },
        "mesh_load": report["qwen_mesh_load_failures"],
        "examples": report["questions"]["examples"],
    }
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
