#!/usr/bin/env python3
"""Training subsets for the follow-up ablations.

These files are not merged into the 985. The published checkpoint stays the
epoch chosen by the training-loss rule. 10/120 and 2/24 are not selection
thresholds for these runs.

exact901
    The 901 teacher replies that matched before the identity-name repair.
matched901
    901 rows drawn from the 985, seed 20261023, so the exact-901 comparison
    is not a comparison of two different sample sizes.
drop-<locus>
    The 985 with one locus removed.
random-<locus>
    A same-size draw from the 985, seed 20261023, one stream per locus.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from repo_paths import PICBENCH_REFERENCE_JSONL, PICBENCH_VISIBLE_JSONL, REPO

REPORT = REPO / "logs" / "sft" / "picbench_glm53_visible_filter.json"
OUT = REPO / "data" / "ablation"
SEED = 20261023


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    kept = [row for row in report["rows"] if row["verdict"] in {"func", "func_repaired"}]
    visible = load_jsonl(PICBENCH_VISIBLE_JSONL)
    reference = load_jsonl(PICBENCH_REFERENCE_JSONL)
    if not (len(kept) == len(visible) == len(reference) == 985):
        raise SystemExit(f"expected 985 aligned rows, got {len(kept)} {len(visible)} {len(reference)}")
    exact = [index for index, row in enumerate(kept) if row["verdict"] == "func"]
    if len(exact) != 901:
        raise SystemExit(f"expected 901 unrepaired matches, got {len(exact)}")
    rng = random.Random(SEED)
    matched = sorted(rng.sample(range(985), 901))

    jobs: list[dict] = []

    def add(name: str, indexes: list[int], kind: str) -> None:
        if len(indexes) != len(set(indexes)):
            raise SystemExit(f"{name} has duplicate indexes")
        vis_path = OUT / f"{name}.visible.jsonl"
        ref_path = OUT / f"{name}.reference.jsonl"
        write_jsonl(vis_path, [visible[index] for index in indexes])
        write_jsonl(ref_path, [reference[index] for index in indexes])
        jobs.append(
            {
                "name": name,
                "kind": kind,
                "n": len(indexes),
                "ids": [kept[index]["id"] for index in indexes],
                "visible_jsonl": str(vis_path),
                "reference_jsonl": str(ref_path),
            }
        )

    add("exact901", exact, "unrepaired-matches")
    add("matched901", matched, "same-count-draw-from-985")

    loci = sorted({row["locus"] for row in kept})
    for locus in loci:
        kept_idx = [index for index, row in enumerate(kept) if row["locus"] != locus]
        removed = 985 - len(kept_idx)
        if removed <= 0:
            raise SystemExit(f"{locus} removed nothing")
        add(f"drop-{locus}", kept_idx, f"remove-{locus}")
        draw = sorted(rng.sample(range(985), len(kept_idx)))
        add(f"random-{locus}", draw, f"same-count-as-drop-{locus}")

    manifest = {
        "seed": SEED,
        "source_n": 985,
        "exact_unrepaired": 901,
        "note": (
            "Subsets of the published 985. Not merged back into that file. "
            "A later training run uses the same 2% loss rule and does not "
            "choose its epoch from the 24 PICBench problems."
        ),
        "jobs": [{key: value for key, value in job.items() if key != "ids"} | {"n_ids": len(job["ids"])} for job in jobs],
        "ids": {job["name"]: job["ids"] for job in jobs},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(jobs)} subset pairs under {OUT}")
    for job in jobs:
        print(f"  {job['name']} {job['n']}")


if __name__ == "__main__":
    main()
