"""Data-integrity guard for the hand-drawn paper figures.

Every constant drawn on fig1..fig5 is checked against the score files under
logs/ and the generator manifest. If a file is present and disagrees with the
figure, this stops the build. Missing files are reported and skipped (the
figures then draw the published constants).

Source of the constants: scripts/draw_paper_figures.py check_sources().
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# The constants each figure draws. Keep in sync with the fig*.py sources.
# ---------------------------------------------------------------------------

# fig1: partition of the 120 stored netlists (Table loci).
AUDIT = {
    "qwen35-9b": {
        "extract": 7, "json": 72, "sax": 37,
        "sax_runtime": 25, "model_name": 9, "ports": 3,
        "wrong": 3, "match": 1,
    },
    "deepseek-v4-pro": {
        "extract": 1, "json": 0, "sax": 44,
        "sax_runtime": 12, "model_name": 8, "ports": 24,
        "wrong": 37, "match": 38,
    },
}
# fig1 drawn bars: JSON not loaded / SAX reject / mismatch / match.
FIG1_BARS = {
    "qwen35-9b": (79, 37, 3, 1),
    "deepseek-v4-pro": (1, 44, 37, 38),
}

# fig2: generator manifest (Table 6) and structure overlap.
LOCI = {
    "ellipsis": 280, "comment": 160, "wrong_model": 280,
    "wrong_ports": 80, "functional": 120, "graft": 80,
}
WIDTHS = {2: 59, 3: 76, 5: 70, 6: 75}
MANIFEST_COUNT = 1000
MANIFEST_SEED = 20261020

# fig3: full 1,000-reply teacher filter (logs/sft/picbench_glm53_visible_filter.json).
FILTER_N = 1000
FILTER_FUNC = 901
FILTER_REPAIRED = 84
FILTER_MISMATCH = 15
FILTER_KEPT = 985
# fig3: student counts out of 120, and epoch-8 summary (Table preregistered).
STUDENT = {
    "zeroshot": {"simulable": 4, "match": 1},
    "epoch8": {"simulable": 27, "match": 4,
               "syntax_pass1": 0.225, "func_pass1": 1 / 30,
               "syntax_pass5": 2 / 3, "func_pass5": 1 / 6},
    "reference": {"simulable": 31, "match": 0,
                  "syntax_pass1": 31 / 120, "func_pass1": 0.0,
                  "syntax_pass5": 11 / 24, "func_pass5": 0.0},
}

# fig4 / fig5: the five measured rows of Table published, in that order.
# name: (syntax@1, func@1, syntax@5, func@5) in percent.
PUBLISHED = {
    "Qwen3.5-9B": (3.33, 0.83, 16.67, 4.17),
    "DeepSeek-V4-Pro": (62.50, 31.67, 95.83, 62.50),
    "Qwen3.6-27B": (63.33, 30.83, 91.67, 70.83),
    "GLM-5.3": (81.67, 39.17, 100.00, 54.17),
    "DeepSeek-V4.1-Flash": (53.33, 30.00, 91.67, 58.33),
}


def _expect(name: str, got, want) -> None:
    if isinstance(want, float):
        if abs(got - want) > 1e-9:
            raise SystemExit(f"{name}: log has {got}, figure uses {want}")
    elif got != want:
        raise SystemExit(f"{name}: log has {got}, figure uses {want}")


def check() -> None:
    missing = []

    audit_path = ROOT / "logs/picbench/failure_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        for key, spec in AUDIT.items():
            off = audit[key]["official"]
            sax = off["sax_rejected"] + off["model_name_rejected"] + off["port_rejected"]
            _expect(f"{key}.extract", off["extraction_failure"], spec["extract"])
            _expect(f"{key}.json", off.get("json_parse_failure", 0), spec["json"])
            _expect(f"{key}.sax", sax, spec["sax"])
            _expect(f"{key}.runtime", off["sax_rejected"], spec["sax_runtime"])
            _expect(f"{key}.name", off["model_name_rejected"], spec["model_name"])
            _expect(f"{key}.ports", off["port_rejected"], spec["ports"])
            _expect(f"{key}.wrong", off["simulation_valid_spectrum_differs"], spec["wrong"])
            _expect(f"{key}.match", off["spectrum_match"], spec["match"])
            total = spec["extract"] + spec["json"] + spec["sax"] + spec["wrong"] + spec["match"]
            _expect(f"{key}.total", total, 120)
            bars = FIG1_BARS[key]
            _expect(f"{key}.bar_json", spec["extract"] + spec["json"], bars[0])
            _expect(f"{key}.bar_sax", spec["sax"], bars[1])
            _expect(f"{key}.bar_wrong", spec["wrong"], bars[2])
            _expect(f"{key}.bar_match", spec["match"], bars[3])
            _expect(f"{key}.bar_sum", sum(bars), 120)
    else:
        missing.append(audit_path)

    manifest_path = ROOT / "data/weakness_questions/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _expect("manifest.count", manifest["count"], MANIFEST_COUNT)
        _expect("manifest.seed", manifest["seed"], MANIFEST_SEED)
        _expect("manifest.sax", manifest["sax_checked"], MANIFEST_COUNT)
        for key, n in LOCI.items():
            _expect(f"locus.{key}", manifest["by_locus"][key], n)
        _expect("locus.sum", sum(LOCI.values()), MANIFEST_COUNT)
    else:
        missing.append(manifest_path)

    overlap_path = ROOT / "logs/picbench/structure_overlap.json"
    if overlap_path.exists():
        overlap = json.loads(overlap_path.read_text())
        _expect("fingerprints", overlap["exact_fingerprint_matches"], [])
        for width, count in WIDTHS.items():
            _expect(f"width.{width}", overlap["train_widths"][str(width)], count)
        _expect("width.sum", sum(WIDTHS.values()), LOCI["ellipsis"])
    else:
        missing.append(overlap_path)

    filt_path = ROOT / "logs/sft/picbench_glm53_visible_filter.json"
    if filt_path.exists():
        filt = json.loads(filt_path.read_text())
        _expect("filter.n", filt["n_scored"], FILTER_N)
        _expect("filter.func", filt["counts"]["func"], FILTER_FUNC)
        _expect("filter.repaired", filt["counts"]["func_repaired"], FILTER_REPAIRED)
        _expect("filter.mismatch", filt["counts"]["spectrum_mismatch"], FILTER_MISMATCH)
        _expect("filter.kept", filt["n_kept"], FILTER_KEPT)
        _expect("filter.sum", FILTER_FUNC + FILTER_REPAIRED + FILTER_MISMATCH, FILTER_N)
        _expect("filter.kept_sum", FILTER_FUNC + FILTER_REPAIRED, FILTER_KEPT)
    else:
        missing.append(filt_path)

    sft_path = ROOT / "logs/picbench/qwen35-9b-visible-sft-epoch8/summary.json"
    if sft_path.exists():
        sft = json.loads(sft_path.read_text())
        _expect("sft.n_designs", sft["n_designs"], 24)
        _expect("sft.syntax_pass1", sft["syntax_pass1"], STUDENT["epoch8"]["syntax_pass1"])
        _expect("sft.func_pass1", sft["func_pass1"], STUDENT["epoch8"]["func_pass1"])
        _expect("sft.syntax_pass5", sft["syntax_pass5"], STUDENT["epoch8"]["syntax_pass5"])
        _expect("sft.func_pass5", sft["func_pass5"], STUDENT["epoch8"]["func_pass5"])
        # design-level "syntax" counts simulable-but-mismatched samples and
        # "func" counts matches, so simulable = syntax + func (23 + 4 = 27).
        simulable = sum(d["syntax"] + d["func"] for d in sft["designs"])
        match = sum(d["func"] for d in sft["designs"])
        _expect("sft.simulable", simulable, STUDENT["epoch8"]["simulable"])
        _expect("sft.match", match, STUDENT["epoch8"]["match"])
        _expect("sft.simulable_vs_pass1", simulable / 120, STUDENT["epoch8"]["syntax_pass1"])
    else:
        missing.append(sft_path)

    ref_path = ROOT / "logs/picbench/qwen35-9b-reference-sft-epoch8/summary.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        _expect("ref.n_designs", ref["n_designs"], 24)
        _expect("ref.syntax_pass1", ref["syntax_pass1"], STUDENT["reference"]["syntax_pass1"])
        _expect("ref.func_pass1", ref["func_pass1"], STUDENT["reference"]["func_pass1"])
        _expect("ref.syntax_pass5", ref["syntax_pass5"], STUDENT["reference"]["syntax_pass5"])
        _expect("ref.func_pass5", ref["func_pass5"], STUDENT["reference"]["func_pass5"])
        simulable = sum(d["syntax"] + d["func"] for d in ref["designs"])
        match = sum(d["func"] for d in ref["designs"])
        _expect("ref.simulable", simulable, STUDENT["reference"]["simulable"])
        _expect("ref.match", match, STUDENT["reference"]["match"])
    else:
        missing.append(ref_path)

    if missing:
        print("check_data: missing sources, published constants drawn:")
        for p in missing:
            print("  -", p.relative_to(ROOT))
    else:
        print("check_data: all figure constants agree with the score files")


if __name__ == "__main__":
    check()
