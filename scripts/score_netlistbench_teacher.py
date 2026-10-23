"""Score the 1,000 NetlistBench teacher replies with the official oracle.

Each question already stores the reference the generator checked. Spice
replies are compared with ``compare_ir`` after the official parser and
numeric normalization. Recognition replies use ``parse_response`` and the
task grader (T2a, T3, T6). Equivalence replies use ``normalize_answer``.
A reference graded against itself must pass; that check runs before the
teacher files are read as scores.

This does not train a student and does not edit the trajectory files.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from repo_paths import NETLIST_TRAJ_DIR, REPO

NB_SCRIPTS = REPO / "third_party" / "NetlistBench" / "scripts"
sys.path.insert(0, str(NB_SCRIPTS))

from eval_equivalence_label import normalize_answer  # noqa: E402
from eval_ir_match import compare_ir, format_normalized_ir  # noqa: E402
from eval_recognition import eval_t2a, eval_t3, eval_t6, parse_response  # noqa: E402
from netlist_to_ir import parse_netlist_text  # noqa: E402

QUESTIONS = REPO / "data" / "netlistbench_questions" / "questions.jsonl"
OUT = REPO / "logs" / "netlistbench" / "glm53-teacher-1000-score.json"
TASKS = {"semantic_pins": "T2a", "node_incidence": "T3", "neighbor_incidence": "T6"}


def load_ir(text: str) -> dict:
    return format_normalized_ir(parse_netlist_text(text))


def spice_errors(expected: str, actual: str) -> list[str]:
    diagnostics = compare_ir(load_ir(expected), load_ir(actual))
    return [row["code"] for row in diagnostics if row.get("severity") == "error"]


def listing_of(question: str) -> str:
    marker = "\nListing:\n"
    if marker not in question:
        raise ValueError("recognition question has no listing")
    return question.split(marker, 1)[1]


def grade_reference(record: dict) -> None:
    kind = record["reference"]["format"]
    if kind == "spice":
        errors = spice_errors(record["reference"]["text"], record["reference"]["text"])
        if errors:
            raise SystemExit(f"{record['id']} reference is not an IR match: {errors[0]}")
    elif kind == "label":
        label, _ = normalize_answer(record["reference"]["text"])
        if label != record["reference"]["text"]:
            raise SystemExit(f"{record['id']} reference label {record['reference']['text']!r}")
    elif kind == "json":
        task = TASKS[record["locus"]]
        expected = record["reference"]["object"]
        ir = parse_netlist_text(listing_of(record["question"]))
        target = record["spec"].get("target")
        verdict = grade_recognition(task, expected, json.dumps(expected), ir, target)
        if verdict != "pass":
            raise SystemExit(f"{record['id']} reference recognition {verdict}")
    else:
        raise SystemExit(f"{record['id']} unknown format {kind}")


def grade_recognition(task: str, expected: dict, text: str, ir: dict, target: dict | None) -> str:
    parsed, failure = parse_response(text or "")
    if parsed is None:
        return failure or "format_error"
    if task == "T2a":
        verdict, _detail = eval_t2a(expected, parsed)
    elif task == "T3":
        verdict, _detail = eval_t3(expected, parsed, ir)
    elif task == "T6":
        verdict, _detail = eval_t6(expected, parsed, ir, target)
    else:
        raise SystemExit(f"unknown recognition task {task}")
    return verdict


def grade_teacher(record: dict, text: str) -> tuple[bool, str]:
    kind = record["reference"]["format"]
    if not text.strip():
        return False, "empty"
    if kind == "spice":
        try:
            errors = spice_errors(record["reference"]["text"], text)
        except (ValueError, json.JSONDecodeError, KeyError, TypeError):
            return False, "parse"
        if errors:
            return False, errors[0]
        return True, "pass"
    if kind == "label":
        label, _diagnostics = normalize_answer(text)
        if label is None:
            return False, "answer_not_found"
        if label != record["reference"]["text"]:
            return False, "label_mismatch"
        return True, "pass"
    task = TASKS[record["locus"]]
    ir = parse_netlist_text(listing_of(record["question"]))
    verdict = grade_recognition(task, record["reference"]["object"], text, ir, record["spec"].get("target"))
    return verdict == "pass", verdict


def main() -> None:
    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(questions) != 1000:
        raise SystemExit(f"{len(questions)} questions")
    for record in questions:
        grade_reference(record)

    by_locus: dict[str, Counter] = {}
    codes: Counter[str] = Counter()
    missing = []
    passed = 0
    rows = []
    for record in questions:
        path = NETLIST_TRAJ_DIR / f"{record['id']}.json"
        if not path.is_file():
            missing.append(record["id"])
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        ok, code = grade_teacher(record, payload.get("content") or "")
        passed += int(ok)
        codes[code] += 1
        bucket = by_locus.setdefault(record["locus"], Counter())
        bucket["n"] += 1
        bucket["pass"] += int(ok)
        rows.append({"id": record["id"], "locus": record["locus"], "pass": ok, "code": code})

    if missing:
        raise SystemExit(f"{len(missing)} missing trajectories, first {missing[:5]}")
    if sum(bucket["n"] for bucket in by_locus.values()) != 1000:
        raise SystemExit("locus counts do not sum to 1000")

    report = {
        "n": 1000,
        "passed": passed,
        "failed": 1000 - passed,
        "oracle": "NetlistBench official IR match, recognition grader, and equivalence-label normalizer",
        "teacher": "glm-5.3-shouyun",
        "trajectories": str(NETLIST_TRAJ_DIR),
        "questions": str(QUESTIONS),
        "note": "Reference self-check passed on all 1,000 items before teacher grading. No student was trained.",
        "by_locus": {
            name: {"n": by_locus[name]["n"], "pass": by_locus[name]["pass"]}
            for name in sorted(by_locus)
        },
        "codes": dict(codes),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"teacher {passed}/1000", flush=True)
    for name in sorted(by_locus):
        print(f"  {name} {by_locus[name]['pass']}/{by_locus[name]['n']}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
