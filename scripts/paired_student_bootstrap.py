"""Paired design-level bootstrap of the two epoch-8 PICBench students.

Teacher-Answer and Reference-JSON are resampled on the same 24 design
indices, with replacement, 10,000 times, seed 20261021. That is the seed
and the percentile used for the zero-shot design bootstrap. The draws are
not a significance test and they do not move the 10/120 or 2/24 bars.

The functional gap is the difference in spectrum-match counts (out of 120).
The design gap is the difference in designs with at least one match (out of 24).
In summary.json, ``func`` is the match count and ``syntax`` counts
simulable-but-mismatched samples, so a design's simulable count is their sum.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from repo_paths import REPO

ROOT = REPO
VISIBLE = ROOT / "logs/picbench/qwen35-9b-visible-sft-epoch8/summary.json"
REFERENCE = ROOT / "logs/picbench/qwen35-9b-reference-sft-epoch8/summary.json"
OUT = ROOT / "logs/picbench/paired_student_bootstrap.json"
REPS = 10000
SEED = 20261021
N_DRAW = 5


def percentile(sorted_values: list[float], q: float) -> float:
    index = (len(sorted_values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] * (high - index) + sorted_values[high] * (index - low)


def interval(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "p2.5": percentile(ordered, 0.025),
        "p97.5": percentile(ordered, 0.975),
    }


def load_rows(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for design in payload["designs"]:
        if design["n"] != N_DRAW:
            raise SystemExit(f"{path.name} {design['design']} n={design['n']}")
        rows[design["design"]] = design
    if len(rows) != 24:
        raise SystemExit(f"{path.name} has {len(rows)} designs")
    return rows


def main() -> None:
    teacher = load_rows(VISIBLE)
    reference = load_rows(REFERENCE)
    names = sorted(teacher)
    if set(names) != set(reference):
        raise SystemExit("design names differ")
    teacher_func = [teacher[name]["func"] for name in names]
    reference_func = [reference[name]["func"] for name in names]
    match_gap = sum(teacher_func) - sum(reference_func)
    design_gap = sum(c > 0 for c in teacher_func) - sum(c > 0 for c in reference_func)
    if match_gap != 4 or design_gap != 4:
        raise SystemExit(f"point estimates moved: matches {match_gap}/120 designs {design_gap}/24")

    rng = random.Random(SEED)
    size = len(names)
    match_samples = []
    design_samples = []
    for _ in range(REPS):
        index = [rng.randrange(size) for _ in range(size)]
        t_func = [teacher_func[i] for i in index]
        r_func = [reference_func[i] for i in index]
        match_samples.append(sum(t_func) - sum(r_func))
        design_samples.append(sum(c > 0 for c in t_func) - sum(c > 0 for c in r_func))

    report = {
        "reps": REPS,
        "seed": SEED,
        "unit": "paired resample of the 24 designs with replacement; both students share each index",
        "not_a_test": "percentile interval of that resample, not a p-value and not a change to the 10/120 or 2/24 bars",
        "n_designs": 24,
        "draws_per_design": N_DRAW,
        "teacher": "qwen35-9b-visible-sft-epoch8",
        "reference": "qwen35-9b-reference-sft-epoch8",
        "functional_matches": {
            "teacher": sum(teacher_func),
            "reference": sum(reference_func),
            "gap": match_gap,
            "denominator": 120,
            "interval": interval(match_samples),
        },
        "functional_designs": {
            "teacher": sum(c > 0 for c in teacher_func),
            "reference": sum(c > 0 for c in reference_func),
            "gap": design_gap,
            "denominator": 24,
            "interval": interval(design_samples),
        },
        "designs": [
            {
                "design": name,
                "teacher_func": teacher[name]["func"],
                "reference_func": reference[name]["func"],
            }
            for name in names
        ],
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    matches = report["functional_matches"]["interval"]
    designs = report["functional_designs"]["interval"]
    print(
        f"matches {match_gap}/120 interval {matches['p2.5']:.2f} to {matches['p97.5']:.2f}",
        flush=True,
    )
    print(
        f"designs {design_gap}/24 interval {designs['p2.5']:.2f} to {designs['p97.5']:.2f}",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
