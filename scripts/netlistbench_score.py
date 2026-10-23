"""Score one NetlistBench run with the official oracle and write a compact JSON.

The run directory must already contain the model text in the layout written by
``netlistbench_run.py``. This does not generate replies and does not touch the
zero-shot score files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from netlistbench_run import ROOT, jobs

NB = ROOT / "third_party" / "NetlistBench"
PY = "python3"


def missing_outputs(out: Path) -> list[str]:
    missing = []
    for row in jobs():
        dest = out / row["rel"]
        if not dest.is_file() or dest.stat().st_size == 0:
            missing.append(str(row["rel"]))
    return missing


def official_reports(out: Path) -> None:
    edit = out / "edit"
    commands = [
        [
            PY,
            "scripts/run_family_reports.py",
            str(edit),
            "--cases-root",
            "benchmark/Cases/v2_edit_raw",
            "--output-format",
            "spice",
            "--pretty",
        ],
        [
            PY,
            "scripts/eval_recognition.py",
            "--grade-run",
            str(out / "recognition" / "flat"),
            "--cases-root",
            "benchmark/Cases/v2_recognition_raw/flat/cases",
        ],
        [
            PY,
            "scripts/eval_recognition.py",
            "--grade-run",
            str(out / "recognition" / "subckt"),
            "--cases-root",
            "benchmark/Cases/v2_recognition_raw/subckt/cases",
        ],
    ]
    for command in commands:
        subprocess.run(command, cwd=NB, check=True)


def compact(out: Path, name: str, checkpoint: str) -> dict:
    overall = json.loads((out / "edit" / "overall_report.json").read_text(encoding="utf-8"))
    first_codes: Counter[str] = Counter()
    step_kinds: Counter[str] = Counter()
    for path in (out / "edit").glob("*/report.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        for fail in report.get("failures", []):
            errors = [diag for diag in fail.get("diagnostics", []) if diag.get("severity") == "error"]
            if errors:
                first_codes[errors[0]["code"]] += 1
            if "compound" not in path.parent.name:
                continue
            for diag in fail.get("diagnostics", []):
                if diag.get("code") != "COMPOUND_STEP_HINT":
                    continue
                index = (diag.get("details") or {}).get("first_failed_step_index")
                steps = (diag.get("details") or {}).get("step_results") or []
                if isinstance(index, int) and 1 <= index <= len(steps):
                    kind = str(steps[index - 1].get("kind"))
                    mapped = {
                        "device_pin_reconnect": "pin",
                        "device_replace": "replace",
                        "device_add": "add",
                        "node_rename_propagation": "rename",
                        "device_remove": "remove",
                        "device_param_edit": "param",
                    }.get(kind)
                    if mapped:
                        step_kinds[mapped] += 1
    recognition = {}
    for split in ("flat", "subckt"):
        report = json.loads((out / "recognition" / split / "oracle_report.json").read_text(encoding="utf-8"))
        recognition[split] = {
            "n": report["n_graded"],
            "pass": report["overall"]["pass"],
            "pass_rate": report["overall"]["pass_rate"],
            "verdict_counts": report["overall"]["verdict_counts"],
            "output_format": report["overall"]["output_format"],
            "per_task": [
                {
                    "task": row["task"],
                    "name": row["task_display_name"],
                    "n": row["n"],
                    "pass": row["pass"],
                    "pass_rate": row["pass_rate"],
                    "verdict_counts": row["verdict_counts"],
                }
                for row in report["per_task"]
            ],
        }
    summary = overall["summary"]
    return {
        "model": name,
        "checkpoint": checkpoint,
        "oracle": "NetlistBench official IR match and recognition grader",
        "protocol": "temperature 0, one completion, official prompts, SPICE, thinking off, max_new_tokens 4096",
        "edit": {
            "total": summary["total_cases"],
            "passed": summary["passed_cases"],
            "failed": summary["failed_cases"],
            "pass_rate": summary["pass_rate"],
            "bucket_summary": summary["bucket_summary"],
            "families": [
                {
                    "family": row["family"],
                    "total": row["total_cases"],
                    "passed": row["passed_cases"],
                    "failed": row["failed_cases"],
                    "pass_rate": row["pass_rate"],
                }
                for row in overall["family_results"]
            ],
            "first_error_code": dict(first_codes),
            "compound_first_failed_step": dict(step_kinds),
        },
        "recognition": recognition,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--score", default="")
    args = parser.parse_args()
    out = Path(args.run)
    if not out.is_absolute():
        out = ROOT / out
    missing = missing_outputs(out)
    if missing:
        raise SystemExit(f"{len(missing)} missing replies, first {missing[:5]}")
    official_reports(out)
    payload = compact(out, args.name, args.checkpoint)
    score_path = Path(args.score) if args.score else ROOT / "logs" / "netlistbench" / f"{args.name}-score.json"
    score_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    edit = payload["edit"]
    flat = payload["recognition"]["flat"]["pass"]
    sub = payload["recognition"]["subckt"]["pass"]
    print(
        f"edit {edit['passed']}/{edit['total']} recognition {flat + sub}/700 "
        f"flat {flat} subckt {sub} -> {score_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
