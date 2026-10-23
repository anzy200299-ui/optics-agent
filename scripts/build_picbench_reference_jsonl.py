#!/usr/bin/env python3
"""Build the Reference-JSON SFT set for the same 985 PICBench questions.

The prompt is the visible-reply prompt: system text, the published example,
and the duplicated user question. The supervised answer is the question's
reference object inside <result>, not the teacher's analysis. The 985 ids
are the rows the teacher filter kept. Each prompt is checked against the
visible-reply JSONL before it is written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from filter_picbench_teacher import training_messages
from repo_paths import (
    PICBENCH_REFERENCE_JSONL,
    PICBENCH_TRAJ_DIR,
    PICBENCH_VISIBLE_JSONL,
    REPO,
)

REPORT = REPO / "logs" / "sft" / "picbench_glm53_visible_filter.json"
QUESTIONS = REPO / "data" / "weakness_questions" / "questions.jsonl"
MANIFEST = REPO / "logs" / "sft" / "picbench_reference_manifest.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    kept_ids = [row["id"] for row in report["rows"] if row["verdict"] in {"func", "func_repaired"}]
    visible = load_jsonl(PICBENCH_VISIBLE_JSONL)
    if len(kept_ids) != len(visible):
        raise SystemExit(f"kept ids {len(kept_ids)} != visible rows {len(visible)}")
    if len(kept_ids) != 985:
        raise SystemExit(f"expected 985 kept questions, found {len(kept_ids)}")
    questions = {}
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            questions[row["id"]] = row
    rows = []
    for qid, vis in zip(kept_ids, visible):
        traj_path = PICBENCH_TRAJ_DIR / f"{qid}.json"
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        if traj.get("id") != qid:
            raise SystemExit(f"{traj_path} id is {traj.get('id')}")
        reference = questions[qid]["reference"]
        if "netlist" not in reference or "models" not in reference:
            raise SystemExit(f"{qid} reference is missing netlist or models")
        rendered = json.dumps(reference, ensure_ascii=False, indent=2)
        answer = f"<result>\n{rendered}\n</result>"
        messages = training_messages(traj["messages"], answer)
        vis_messages = vis["messages"]
        for index in (0, 1, 3):
            if messages[index]["content"] != vis_messages[index]["content"]:
                raise SystemExit(f"{qid} prompt turn {index} differs from the visible-reply file")
        if messages[2]["content"] != vis_messages[2]["content"] or messages[2].get("loss") is not False:
            raise SystemExit(f"{qid} example answer differs from the visible-reply file")
        if "<analysis>" in messages[4]["content"]:
            raise SystemExit(f"{qid} reference answer still contains an analysis")
        rows.append({"messages": messages})
    PICBENCH_REFERENCE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with PICBENCH_REFERENCE_JSONL.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    MANIFEST.write_text(
        json.dumps(
            {
                "n": len(rows),
                "ids": kept_ids,
                "jsonl": str(PICBENCH_REFERENCE_JSONL),
                "target": "question reference inside <result>; no teacher analysis",
                "prompt": "same system, example, and duplicated user turn as the visible-reply JSONL",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {PICBENCH_REFERENCE_JSONL}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
