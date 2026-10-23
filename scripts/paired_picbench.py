"""Design-clustered paired comparison of PICBench Without-EF scores.

The clustering unit is the design. Five samples of one design share a prompt,
so they are not five independent trials. The one pre-specified contrast is
Teacher-Answer epoch 8 minus Reference-JSON epoch 8. Other arms are
descriptive. The sign-flip distribution treats the two students as
exchangeable inside each design. It does not replace the 10/120 functional
bar, and it is not a license to choose a checkpoint.

The score file's per-design ``syntax`` integer counts spectrum mismatches
only. Simulable samples are that integer plus the functional matches.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from repo_paths import REPO

ROOT = REPO
LOG = ROOT / "logs" / "picbench"
OUT = LOG / "paired_design.json"

ARMS = (
    "qwen35-9b",
    "deepseek-v4-pro",
    "qwen3.6-27b",
    "glm-5.3",
    "deepseek-v4.1-flash",
    "qwen35-9b-visible-sft-epoch8",
    "qwen35-9b-reference-sft-epoch8",
)
TEACHER = "qwen35-9b-visible-sft-epoch8"
REFERENCE = "qwen35-9b-reference-sft-epoch8"
ZERO = "qwen35-9b"

# Paper Table per-design order. Used only to keep the report stable.
ORDER = (
    "Clements_4x4",
    "Clements_8x8",
    "Reck_4x4",
    "Reck_8x8",
    "NLS",
    "U-matrix_block",
    "Direct modulator",
    "QPSK modulator",
    "8-QAM modulator",
    "64-QAM modulator",
    "WDM_mux",
    "WDM_demux",
    "Optical hybrid",
    "OS_2x2",
    "Crossbar_4x4",
    "Crossbar_8x8",
    "Spanke_4x4",
    "Spanke_8x8",
    "Benes_4x4",
    "Benes_8x8",
    "Spanke–Benes_4x4",
    "Spanke–Benes_8x8",
    "MZM",
    "MZI_ps",
)
FAMILY = {}
for _name in ORDER[:6]:
    FAMILY[_name] = "computing"
for _name in ORDER[6:13]:
    FAMILY[_name] = "interconnect"
for _name in ORDER[13:22]:
    FAMILY[_name] = "switch"
for _name in ORDER[22:]:
    FAMILY[_name] = "device"

# Regression check against the manuscript. The report still records the files.
EXPECTED = {
    "qwen35-9b": (4, 1),
    "deepseek-v4-pro": (75, 38),
    "qwen3.6-27b": (76, 37),
    "glm-5.3": (98, 47),
    "deepseek-v4.1-flash": (64, 36),
    "qwen35-9b-visible-sft-epoch8": (27, 4),
    "qwen35-9b-reference-sft-epoch8": (31, 0),
}


def sign_flip(diffs: list[int]) -> dict:
    """Exact sign-flip of the paired sum. Zeros stay zero and are not flipped."""
    nonzero = [int(d) for d in diffs if d != 0]
    observed = int(sum(diffs))
    k = len(nonzero)
    base = {
        "n_positive": sum(d > 0 for d in diffs),
        "n_negative": sum(d < 0 for d in diffs),
        "n_tie": sum(d == 0 for d in diffs),
        "sum": observed,
        "nonzero": k,
    }
    if k == 0:
        return {**base, "method": "exact", "assignments": 1, "p_two_sided": "1/1", "p_one_sided_positive": "1/1"}
    if k > 24:
        raise RuntimeError(f"refusing to enumerate 2^{k} sign flips")
    assignments = 1 << k
    extreme_two = 0
    extreme_positive = 0
    for mask in range(assignments):
        total = 0
        for index, value in enumerate(nonzero):
            sign = 1 if (mask >> index) & 1 else -1
            total += sign * value
        if abs(total) >= abs(observed):
            extreme_two += 1
        if total >= observed:
            extreme_positive += 1
    return {
        **base,
        "method": "exact",
        "assignments": assignments,
        "p_two_sided": f"{extreme_two}/{assignments}",
        "p_one_sided_positive": f"{extreme_positive}/{assignments}",
    }


def load_summary(name: str) -> dict[str, dict]:
    """picbench_score.py writes syntax as mismatches only, not as simulable."""
    path = LOG / name / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for design in summary["designs"]:
        if design["n"] != 5:
            raise RuntimeError(f"{name} {design['design']} has n={design['n']}, expected 5")
        match = 0
        mismatch = 0
        for detail in design["details"]:
            verdict = detail["verdict"] or ""
            if verdict.startswith("functional check passed"):
                match += 1
            elif verdict.startswith("functional error"):
                mismatch += 1
        if match != design["func"] or mismatch != design["syntax"]:
            raise RuntimeError(
                f"{name} {design['design']}: details give match/mismatch "
                f"{match}/{mismatch}, summary func/syntax is {design['func']}/{design['syntax']}"
            )
        rows[design["design"]] = {"match": match, "mismatch": mismatch, "simulable": match + mismatch}
    return rows


def load_legacy(name: str) -> dict[str, dict]:
    """Early score files store simulable counts in ``syntax`` and bucket labels in ``detail``.

    ``failure_audit.json`` re-ran the judge on the stored netlist. The two
    files have to name the same match and mismatch samples before a row is kept.
    """
    score = json.loads((LOG / f"{name}-score.json").read_text(encoding="utf-8"))
    audit = json.loads((LOG / "failure_audit.json").read_text(encoding="utf-8"))[name]
    rows: dict[str, dict] = {}
    for row in audit["rows"]:
        slot = rows.setdefault(row["design"], {"match": 0, "mismatch": 0, "n": 0})
        slot["n"] += 1
        if row["official"] == "spectrum_match":
            slot["match"] += 1
        elif row["official"] == "simulation_valid_spectrum_differs":
            slot["mismatch"] += 1
    for design in score["designs"]:
        detail = design["detail"]
        match = sum(label == "func" for label in detail)
        mismatch = sum(label == "syntax_only" for label in detail)
        got = rows[design["design"]]
        if got["n"] != 5 or got["match"] != match or got["mismatch"] != mismatch:
            raise RuntimeError(
                f"{name} {design['design']}: score file {match}/{mismatch}, "
                f"failure audit {got['match']}/{got['mismatch']}"
            )
        if design["func"] != match or design["syntax"] != match + mismatch:
            raise RuntimeError(
                f"{name} {design['design']}: legacy syntax integer is not match+mismatch"
            )
        got["simulable"] = match + mismatch
        del got["n"]
    return rows


def load_arm(name: str) -> dict[str, dict]:
    if (LOG / name / "summary.json").is_file():
        return load_summary(name)
    if (LOG / f"{name}-score.json").is_file():
        return load_legacy(name)
    raise FileNotFoundError(f"no summary.json or score file for {name}")


def contrast(left: dict[str, dict], right: dict[str, dict], designs: list[str], key: str) -> dict:
    diffs = []
    discordant = []
    for design in designs:
        diff = left[design][key] - right[design][key]
        diffs.append(diff)
        if diff != 0:
            discordant.append(
                {
                    "design": design,
                    "family": FAMILY.get(design, "other"),
                    "left": left[design][key],
                    "right": right[design][key],
                    "diff": diff,
                }
            )
    return {"statistic": sign_flip(diffs), "discordant": discordant}


def family_totals(arm: dict[str, dict], designs: list[str]) -> dict:
    totals = defaultdict(lambda: {"match": 0, "simulable": 0, "designs": 0})
    for design in designs:
        family = FAMILY.get(design, "other")
        totals[family]["match"] += arm[design]["match"]
        totals[family]["simulable"] += arm[design]["simulable"]
        totals[family]["designs"] += 1
    return dict(totals)


def _self_check() -> None:
    four = sign_flip([1, 1, 1, 1] + [0] * 20)
    if four["p_two_sided"] != "2/16" or four["p_one_sided_positive"] != "1/16":
        raise RuntimeError(f"sign-flip self-check failed: {four}")
    tied = sign_flip([0, 0, 0])
    if tied["p_two_sided"] != "1/1":
        raise RuntimeError(f"sign-flip zero self-check failed: {tied}")


def main() -> None:
    _self_check()
    loaded = {name: load_arm(name) for name in ARMS}
    design_set = {tuple(sorted(rows)) for rows in loaded.values()}
    if len(design_set) != 1:
        raise RuntimeError("arms do not share a design set")
    designs = list(design_set.pop())
    missing = [name for name in designs if name not in ORDER]
    extra = [name for name in ORDER if name not in designs]
    if missing or extra:
        raise RuntimeError(f"design list drifted, missing={missing}, extra={extra}")
    designs = list(ORDER)

    headline = {}
    for name, rows in loaded.items():
        simulable = sum(row["simulable"] for row in rows.values())
        match = sum(row["match"] for row in rows.values())
        expected = EXPECTED[name]
        headline[name] = {
            "simulable": simulable,
            "match": match,
            "expected_simulable_match": list(expected),
            "matches_manuscript": [simulable, match] == list(expected),
        }

    teacher = loaded[TEACHER]
    reference = loaded[REFERENCE]
    zero = loaded[ZERO]
    prespecified = {
        "left": TEACHER,
        "right": REFERENCE,
        "difference": "left minus right",
        "functional_matches": contrast(teacher, reference, designs, "match"),
        "simulable": contrast(teacher, reference, designs, "simulable"),
        "bars": {
            "functional_samples_10_of_120": {
                "observed": sum(teacher[d]["match"] - reference[d]["match"] for d in designs),
                "bar": 10,
            },
            "designs_with_any_functional_sample_2_of_24": {
                "observed_left": sum(teacher[d]["match"] > 0 for d in designs),
                "observed_right": sum(reference[d]["match"] > 0 for d in designs),
                "bar": 2,
                "note": (
                    "With n=k=5, functional Pass@5 equals the fraction of designs "
                    "with at least one match. Singleton hits clear a 2/24 count "
                    "without clearing 10/120. The sign-flip probability describes "
                    "this decode under exchangeability. It is not that bar."
                ),
            },
        },
    }
    descriptive = {
        "teacher_minus_zero_functional": contrast(teacher, zero, designs, "match"),
        "reference_minus_zero_functional": contrast(reference, zero, designs, "match"),
        "teacher_minus_zero_simulable": contrast(teacher, zero, designs, "simulable"),
        "reference_minus_zero_simulable": contrast(reference, zero, designs, "simulable"),
        "note": "These four contrasts were not the pre-specified student comparison.",
    }
    per_design = []
    for design in designs:
        row = {"design": design, "family": FAMILY[design]}
        for name, rows in loaded.items():
            row[name] = rows[design]
        per_design.append(row)

    report = {
        "unit": "PICBench design",
        "n_designs": len(designs),
        "samples_per_design": 5,
        "official_criterion_unchanged": True,
        "checkpoint_selection": "not used; both students are the epoch-8 cap checkpoints already scored",
        "sources": [
            f"logs/picbench/{name}/summary.json"
            if (LOG / name / "summary.json").is_file()
            else f"logs/picbench/{name}-score.json + logs/picbench/failure_audit.json"
            for name in ARMS
        ],
        "headline_check": headline,
        "prespecified_contrast": prespecified,
        "descriptive_contrasts": descriptive,
        "family_totals": {name: family_totals(rows, designs) for name, rows in loaded.items()},
        "per_design": per_design,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    functional = prespecified["functional_matches"]["statistic"]
    simulable = prespecified["simulable"]["statistic"]
    print(f"wrote {OUT}")
    print(
        "teacher minus reference, functional sum "
        f"{functional['sum']} sign-flip two-sided {functional['p_two_sided']} "
        f"one-sided {functional['p_one_sided_positive']}"
    )
    print(
        "teacher minus reference, simulable sum "
        f"{simulable['sum']} sign-flip two-sided {simulable['p_two_sided']} "
        f"one-sided {simulable['p_one_sided_positive']}"
    )
    failed = [name for name, row in headline.items() if not row["matches_manuscript"]]
    print("headline check", "ok" if not failed else f"DIFFERS {failed}")


if __name__ == "__main__":
    main()
