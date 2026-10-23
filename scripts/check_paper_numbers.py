"""Recompute the counts cited in paper/main.tex from the score files.

The script checks identities. It does not edit the paper.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pass_rates(counts: list[int]) -> tuple[float, float]:
    n = len(counts)
    at1 = sum(counts) / (n * 5)
    at5 = sum(c > 0 for c in counts) / n
    return at1, at5


def bootstrap(syn: list[int], fun: list[int], seed: int = 20261021, draws: int = 10000):
    rng = random.Random(seed)
    n = len(syn)
    rows = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        s = [syn[i] for i in idx]
        f = [fun[i] for i in idx]
        rows.append((*pass_rates(s), *pass_rates(f)[:1], pass_rates(f)[1]))
    # rows are (syn@1, syn@5, fun@1, fun@5) — pass_rates returns both, so rebuild cleanly
    rows = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        s1, s5 = pass_rates([syn[i] for i in idx])
        f1, f5 = pass_rates([fun[i] for i in idx])
        rows.append((s1, f1, s5, f5))
    cols = list(zip(*rows))
    out = []
    for col in cols:
        xs = sorted(col)
        lo = xs[int(0.025 * (draws - 1))]
        hi = xs[int(0.975 * (draws - 1))]
        out.append((lo, hi))
    return out


def main() -> None:
    q = load(ROOT / "logs/picbench/qwen35-9b-score.json")
    d = load(ROOT / "logs/picbench/deepseek-v4-pro-score.json")
    assert q["files"] == d["files"] == 120
    assert q["full_designs"] == d["full_designs"] == 24
    assert (q["syntax_passes"], q["func_passes"]) == (4, 1)
    assert (d["syntax_passes"], d["func_passes"]) == (75, 38)
    q_syn = [row["syntax"] for row in q["designs"]]
    q_fun = [row["func"] for row in q["designs"]]
    d_syn = [row["syntax"] for row in d["designs"]]
    d_fun = [row["func"] for row in d["designs"]]
    assert pass_rates(q_syn) == (4 / 120, 4 / 24)
    assert pass_rates(q_fun) == (1 / 120, 1 / 24)
    assert pass_rates(d_syn) == (75 / 120, 23 / 24)
    assert pass_rates(d_fun) == (38 / 120, 15 / 24)

    audit = load(ROOT / "logs/picbench/failure_audit.json")
    for name, expected in (
        ("qwen35-9b", {"extraction_failure": 7, "json_parse_failure": 72, "sax_rejected": 25, "model_name_rejected": 9, "port_rejected": 3, "simulation_valid_spectrum_differs": 3, "spectrum_match": 1}),
        ("deepseek-v4-pro", {"spectrum_match": 38, "simulation_valid_spectrum_differs": 37, "port_rejected": 24, "sax_rejected": 12, "model_name_rejected": 8, "extraction_failure": 1}),
    ):
        got = audit[name]["official"]
        assert sum(got.values()) == 120
        for key, value in expected.items():
            assert got[key] == value, (name, key, got[key], value)

    filt = load(ROOT / "logs/sft/picbench_glm53_visible_filter.json")
    assert filt["counts"] == {"func": 901, "func_repaired": 84, "spectrum_mismatch": 15}
    assert filt["n_kept"] == 985
    repairs = Counter(tuple(row["added_models"]) for row in filt["rows"] if row["verdict"] == "func_repaired")
    assert repairs[("straight_heat_metal",)] == 76
    assert repairs[("straight",)] == 7
    assert sum(repairs.values()) == 84

    questions = [json.loads(line) for line in (ROOT / "data/weakness_questions/questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(questions) == 1000
    assert Counter(row["locus"] for row in questions) == {
        "ellipsis": 280,
        "wrong_model": 280,
        "comment": 160,
        "functional": 120,
        "wrong_ports": 80,
        "graft": 80,
    }
    named = 0
    sax_phrase = 0
    bind = 0
    portpair = 0
    for row in questions:
        body = row["question"].split("\n", 1)[-1]
        values = set(row["reference"]["models"].values())
        if values and all(re.search(rf"\b{re.escape(value)}\b", body) for value in values):
            named += 1
        if re.search(r"\bSAX model\b", body):
            sax_phrase += 1
        if re.search(r"\bBind\b", body):
            bind += 1
        if re.search(r"\b[A-Za-z0-9_]+,[A-Z][A-Za-z0-9]*\b", body):
            portpair += 1
    assert (named, sax_phrase, bind, portpair) == (1000, 960, 993, 1000)

    q_iv = bootstrap(q_syn, q_fun)
    d_iv = bootstrap(d_syn, d_fun)

    def pct(pair: tuple[float, float]) -> tuple[float, float]:
        return (round(100 * pair[0], 2), round(100 * pair[1], 2))

    assert pct(q_iv[1]) == (0.0, 2.5)
    assert pct(q_iv[3]) == (0.0, 12.5)
    assert pct(d_iv[1]) == (19.17, 45.0)
    assert pct(d_iv[3]) == (41.67, 79.17)

    def verdict_counts(path: Path) -> Counter:
        obj = load(path)
        counts: Counter = Counter()
        per = []
        for design in obj["designs"]:
            assert design["n"] == 5
            syn = fun = 0
            for item in design["details"]:
                verdict = item["verdict"] or ""
                if verdict.startswith("functional check passed"):
                    kind = "match"
                    syn += 1
                    fun += 1
                elif verdict.startswith("functional error"):
                    kind = "sim_diff"
                    syn += 1
                elif verdict.startswith("wrong model"):
                    kind = "model"
                elif verdict.startswith("wrong ports"):
                    kind = "ports"
                elif verdict.startswith("other syntax") and "Missing models" in verdict:
                    kind = "missing"
                elif verdict.startswith("other syntax"):
                    kind = "other"
                elif verdict.startswith("no result"):
                    kind = "no_result"
                else:
                    kind = "unexpected"
                counts[kind] += 1
            per.append((syn, fun))
        assert counts["unexpected"] == 0
        assert sum(counts.values()) == 120
        return counts, per

    q36_counts, q36_per = verdict_counts(ROOT / "logs/picbench/qwen3.6-27b/summary.json")
    glm_counts, glm_per = verdict_counts(ROOT / "logs/picbench/glm-5.3/summary.json")
    assert q36_counts == Counter({"sim_diff": 39, "match": 37, "other": 18, "model": 11, "ports": 11, "missing": 4})
    assert glm_counts == Counter({"sim_diff": 51, "match": 47, "ports": 11, "missing": 10, "model": 1})
    assert pass_rates([row[0] for row in q36_per]) == (76 / 120, 22 / 24)
    assert pass_rates([row[1] for row in q36_per]) == (37 / 120, 17 / 24)
    assert pass_rates([row[0] for row in glm_per]) == (98 / 120, 24 / 24)
    assert pass_rates([row[1] for row in glm_per]) == (47 / 120, 13 / 24)
    flash_counts, flash_per = verdict_counts(ROOT / "logs/picbench/deepseek-v4.1-flash/summary.json")
    assert flash_counts == Counter({"ports": 54, "match": 36, "sim_diff": 28, "missing": 2})
    assert pass_rates([row[0] for row in flash_per]) == (64 / 120, 22 / 24)
    assert pass_rates([row[1] for row in flash_per]) == (36 / 120, 14 / 24)

    sft_counts, sft_per = verdict_counts(ROOT / "logs/picbench/qwen35-9b-visible-sft-epoch8/summary.json")
    assert sft_counts == Counter(
        {"other": 48, "sim_diff": 23, "model": 21, "ports": 15, "missing": 7, "match": 4, "no_result": 2}
    )
    assert pass_rates([row[0] for row in sft_per]) == (27 / 120, 16 / 24)
    assert pass_rates([row[1] for row in sft_per]) == (4 / 120, 4 / 24)
    by_name = {
        design["design"]: (syn, fun)
        for design, (syn, fun) in zip(
            load(ROOT / "logs/picbench/qwen35-9b-visible-sft-epoch8/summary.json")["designs"],
            sft_per,
        )
    }
    assert by_name["Direct modulator"] == (3, 1)
    assert by_name["WDM_demux"] == (2, 1)
    assert by_name["Crossbar_8x8"] == (1, 1)
    assert by_name["Spanke–Benes_4x4"] == (1, 1)
    assert by_name["QPSK modulator"] == (2, 0)
    families = {
        "Computing": ["Clements_4x4", "Clements_8x8", "Reck_4x4", "Reck_8x8", "NLS", "U-matrix_block"],
        "Interconnect": [
            "Direct modulator",
            "QPSK modulator",
            "8-QAM modulator",
            "64-QAM modulator",
            "WDM_mux",
            "WDM_demux",
            "Optical hybrid",
        ],
        "Switch": [
            "OS_2x2",
            "Crossbar_4x4",
            "Crossbar_8x8",
            "Spanke_4x4",
            "Spanke_8x8",
            "Benes_4x4",
            "Benes_8x8",
            "Spanke–Benes_4x4",
            "Spanke–Benes_8x8",
        ],
        "Device": ["MZM", "MZI_ps"],
    }
    family_totals = {
        name: (
            sum(by_name[item][0] for item in items),
            sum(by_name[item][1] for item in items),
        )
        for name, items in families.items()
    }
    assert family_totals == {
        "Computing": (3, 0),
        "Interconnect": (11, 2),
        "Switch": (10, 2),
        "Device": (3, 0),
    }
    ref_counts, ref_per = verdict_counts(ROOT / "logs/picbench/qwen35-9b-reference-sft-epoch8/summary.json")
    assert ref_counts == Counter({"other": 43, "ports": 32, "sim_diff": 31, "model": 13, "missing": 1})
    assert pass_rates([row[0] for row in ref_per]) == (31 / 120, 11 / 24)
    assert pass_rates([row[1] for row in ref_per]) == (0 / 120, 0 / 24)
    ref_by_name = {
        design["design"]: (syn, fun)
        for design, (syn, fun) in zip(
            load(ROOT / "logs/picbench/qwen35-9b-reference-sft-epoch8/summary.json")["designs"],
            ref_per,
        )
    }
    assert ref_by_name["Direct modulator"] == (5, 0)
    assert ref_by_name["Reck_4x4"] == (5, 0)
    assert ref_by_name["MZI_ps"] == (5, 0)
    assert ref_by_name["MZM"] == (4, 0)
    assert sum(row[1] for row in ref_per) == 0
    tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    for needle in (
        "Teacher-Answer, epoch 8 & 22.50 & 3.33 & 66.67 & 16.67 & measured",
        "Reference-JSON, epoch 8 & 25.83 & 0.00 & 45.83 & 0.00 & measured",
        "Direct modulator & Interconnect & 1/0 & 5/5 & 3/3 & 5/1 & 4/4 & 3/1 & 5/0",
        "WDM demultiplexer & Interconnect & 0/0 & 0/0 & 4/4 & 4/4 & 1/1 & 2/1 & 0/0",
        "Crossbar $8\\times8$ & Switch & 0/0 & 3/0 & 3/1 & 5/5 & 4/0 & 1/1 & 0/0",
        "Spanke--Benes $4\\times4$ & Switch & 0/0 & 4/4 & 2/2 & 5/5 & 3/3 & 1/1 & 0/0",
        "Epoch 8 sums to $27$ simulable and $4$ functional",
        "Reference-JSON epoch 8 sums to $13+8+1+9 = 31$ simulable and $0$ functional",
    ):
        assert needle in tex, needle
    print("paper number check ok")


if __name__ == "__main__":
    main()
