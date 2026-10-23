"""Structural overlap between the 1,000 training references and the 24 expert netlists.

Lexical 8-word shingles are not this check. A fingerprint is the multiset of
SAX model ids plus the multiset of directed port-to-port edges after instance
names are replaced by (model, local index). Exact fingerprint equality is
recorded. It is not a proof that no structural information leaked through the
failure labels that shaped the generator.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from repo_paths import REPO

ROOT = REPO
CASES = ROOT / "third_party" / "PICBench" / "testcases"
QUESTIONS = ROOT / "data" / "weakness_questions" / "questions.jsonl"


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
        ports.append((external[0], sax_of[name], port))
    return (tuple(sorted(counts.items())), tuple(sorted(edges)), tuple(sorted(ports)))


def main() -> None:
    train = []
    for line in QUESTIONS.read_text().splitlines():
        if line.strip():
            train.append(json.loads(line))
    test = []
    for path in sorted(CASES.glob("*/*_ref.json")):
        test.append((path.parent.name, json.loads(path.read_text())))
    train_fp = [(row["id"], row["locus"], fingerprint(row["reference"])) for row in train]
    test_fp = [(name, fingerprint(doc)) for name, doc in test]
    train_set = {fp for _i, _l, fp in train_fp}
    exact = []
    for name, fp in test_fp:
        hits = [qid for qid, _locus, other in train_fp if other == fp]
        if hits:
            exact.append({"design": name, "train_ids": hits[:8], "n": len(hits)})
    model_overlap = []
    for name, doc in test:
        test_models = set(doc["models"].values())
        # device vocabulary overlap is expected; record it, do not treat it as leakage
        model_overlap.append({"design": name, "sax_models": sorted(test_models)})
    widths = Counter()
    for row in train:
        spec = row.get("spec") or {}
        if "width" in spec:
            widths[spec["width"]] += 1
    out = {
        "train": len(train),
        "test": len(test),
        "exact_fingerprint_matches": exact,
        "train_widths": dict(widths),
        "test_model_vocab": model_overlap,
        "note": (
            "Exact fingerprint match means the same SAX-model multiset, the same "
            "renamed port edges, and the same external port roles. Zero matches "
            "do not imply the generator was independent of the test labels."
        ),
    }
    dest = ROOT / "logs" / "picbench" / "structure_overlap.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("train", "test", "exact_fingerprint_matches", "train_widths")}, indent=2))


if __name__ == "__main__":
    main()
