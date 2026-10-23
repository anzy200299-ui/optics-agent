"""Re-audit PICBench replies against the string the judge actually scored.

Official scores stay on the stored netlist (Without EF extraction).
A second pass applies a deterministic format repair that does not read the
expert netlist: strip fences, drop full-line // comments, and keep the first
JSON object. That pass is a modified protocol, not a Without-EF score.

The 33 teacher schema misses are inspected on disk. Nothing is rewritten.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

from repo_paths import PICBENCH_TRAJ_DIR, REPO

ROOT = REPO
PIC = ROOT / "third_party" / "PICBench" / "PICBench"
os.chdir(PIC)
sys.path.insert(0, str(PIC))

from evaluation import evaluate  # noqa: E402


def parse_reason(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("```"):
        return "fence_remains"
    if "..." in text:
        return "ellipsis"
    if "//" in text:
        return "line_comment"
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        message = str(exc)
        if "Extra data" in message:
            return "trailing_text"
        return "other_parse"
    return "parses"


def stage_of(netlist: str | None, design: str) -> tuple[str, str]:
    if not netlist or not str(netlist).strip():
        return "extraction_failure", "empty"
    text = str(netlist)
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return "json_parse_failure", parse_reason(text)
    verdict = str(evaluate(text, design))
    if verdict.startswith("functional check passed"):
        return "spectrum_match", ""
    if verdict.startswith("functional error"):
        return "simulation_valid_spectrum_differs", ""
    if verdict.startswith("wrong model"):
        return "model_name_rejected", verdict[:180]
    if verdict.startswith("wrong ports"):
        return "port_rejected", verdict[:180]
    return "sax_rejected", verdict[:180]


def format_repair(response: str | None, stored: str | None) -> str | None:
    """Deterministic cleanup. Does not invent omitted devices or read a golden."""
    chunks: list[str] = []
    if stored:
        chunks.append(stored)
    if response and "<result>" in response:
        chunks.append(response.split("<result>", 1)[1])
    elif response:
        chunks.append(response)
    decoder = json.JSONDecoder()
    for raw in chunks:
        text = raw.replace("</result>", "")
        text = text.replace("```json", "").replace("```JSON", "").replace("```", "")
        lines = [line for line in text.splitlines() if not line.strip().startswith("//")]
        text = "\n".join(lines)
        start = text.find("{")
        if start < 0:
            continue
        try:
            obj, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return json.dumps(obj)
    return None


def audit_model(name: str) -> dict:
    root = ROOT / "logs" / "picbench" / name
    score = json.loads((ROOT / "logs" / "picbench" / f"{name}-score.json").read_text())
    old = {}
    for row in score["designs"]:
        for index, label in enumerate(row["detail"]):
            old[(row["design"], index)] = label
    rows = []
    official = Counter()
    repaired_stage = Counter()
    moved = Counter()
    for design_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        design = design_dir.name
        for sample_path in sorted(design_dir.glob("s*.json")):
            index = int(sample_path.stem[1:])
            payload = json.loads(sample_path.read_text())
            stored = payload.get("netlist")
            response = payload.get("response") or ""
            kind, detail = stage_of(stored if isinstance(stored, str) else None, design)
            official[kind] += 1
            repaired = format_repair(response, stored if isinstance(stored, str) else None)
            if repaired is None:
                new_kind, new_detail = "extraction_failure", "unrecoverable"
            else:
                new_kind, new_detail = stage_of(repaired, design)
            repaired_stage[new_kind] += 1
            if new_kind != kind:
                moved[f"{kind} -> {new_kind}"] += 1
            rows.append(
                {
                    "design": design,
                    "sample": index,
                    "old_bucket": old.get((design, index)),
                    "official": kind,
                    "official_detail": detail,
                    "repaired": new_kind,
                    "repaired_detail": new_detail,
                    "stored_has_fence": isinstance(stored, str) and "```" in stored,
                    "stored_has_ellipsis": isinstance(stored, str) and "..." in stored,
                    "response_has_fence": "```" in response,
                }
            )
            print(f"{name} {design} s{index}: {kind} | repair {new_kind}", flush=True)
    return {
        "model": name,
        "n": len(rows),
        "official": dict(official),
        "after_format_repair": dict(repaired_stage),
        "transitions": dict(moved),
        "rows": rows,
    }


def teacher_schema() -> dict:
    score_path = ROOT / "logs" / "sft" / "glm-5.3-shouyun-score.json"
    score = json.loads(score_path.read_text())
    root = PICBENCH_TRAJ_DIR
    missing_on_disk = 0
    patched_on_disk = 0
    content_still_missing = 0
    checked = 0
    for row in score["rows"]:
        if row["verdict"] != "syntax":
            continue
        checked += 1
        payload = json.loads((root / f"{row['id']}.json").read_text())
        netlist = json.loads(payload["netlist"])
        values = set(netlist.get("models", {}).values())
        blob = json.dumps(netlist)
        if "straight_heat_metal" not in values and "straight_heat_metal" in blob:
            missing_on_disk += 1
        else:
            patched_on_disk += 1
        content = payload.get("content") or ""
        if "straight_heat_metal" not in content:
            content_still_missing += 1
    return {
        "scored_prefix": score["n"],
        "syntax_rows": checked,
        "netlist_uses_device_but_models_omits_name": missing_on_disk,
        "netlist_already_names_device": patched_on_disk,
        "visible_content_omits_name": content_still_missing,
        "files_on_disk": len(list(root.glob("q*.json"))),
        "note": "The scorer did not write a repaired netlist back. The stored content is the training target.",
    }


def main() -> None:
    report = {
        "qwen35-9b": audit_model("qwen35-9b"),
        "deepseek-v4-pro": audit_model("deepseek-v4-pro"),
        "teacher_prefix": teacher_schema(),
    }
    # Drop per-row payloads from the printed summary; keep them in the file.
    dest = ROOT / "logs" / "picbench" / "failure_audit.json"
    dest.write_text(json.dumps(report, indent=2))
    summary = {
        "qwen_official": report["qwen35-9b"]["official"],
        "qwen_repair": report["qwen35-9b"]["after_format_repair"],
        "qwen_transitions": report["qwen35-9b"]["transitions"],
        "deepseek_official": report["deepseek-v4-pro"]["official"],
        "deepseek_repair": report["deepseek-v4-pro"]["after_format_repair"],
        "deepseek_transitions": report["deepseek-v4-pro"]["transitions"],
        "teacher": report["teacher_prefix"],
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
