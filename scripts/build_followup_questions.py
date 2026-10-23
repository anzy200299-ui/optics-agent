#!/usr/bin/env python3
"""Three question sets that stay out of data/weakness_questions/questions.jsonl.

reweight
    1,000 items whose family counts follow the re-audit, by largest remainder.
    The published Qwen counts used as weights are: ellipsis 44, the other
    json.loads rejects 28, model-name rejects 9, port rejects 3, spectrum
    differs or matches 4, and SAX-runtime rejects 25. Those are Table loci,
    not the hand-set 280/160/280/80/120/80.
random
    985 legal circuits. Each draw picks a constructor uniformly, then a
    parameter uniformly. This is not a failure-locus allocation.
noname
    The existing 1,000 references, rewritten so the prompt states the device
    kind and the exact instance names without a SAX model id. The reference
    JSON still contains the library id. An item whose prompt shares an
    eight-word shingle with the 24 test problems or the in-context example
    is dropped.

Every kept reference is checked with SAX on eight wavelengths. None of these
files replaces the 985 training set.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from repo_paths import PICBENCH_VISIBLE_JSONL, REPO

import build_weakness_questions as builder

OUT = REPO / "data" / "ablation"
QUESTIONS = REPO / "data" / "weakness_questions" / "questions.jsonl"
NONAME = REPO / "data" / "weakness_questions_noname"

# Re-audit counts. Sum 113. Largest remainder then scales them to 1,000.
REAUDIT = {
    "ellipsis": 44,
    "comment": 28,
    "wrong_model": 9,
    "wrong_ports": 3,
    "functional": 4,
    "graft": 25,
}

PHRASE = {
    "mzi": "one-input interferometer",
    "mzm": "single-drive modulator",
    "mzm_dual": "dual-drive modulator",
    "mmi1x2": "one-to-two splitter",
    "straight": "unheated waveguide",
    "straight_heat_metal": "heated waveguide",
    "coupler": "two-by-two evanescent splitter",
    "mzi_2x2": "two-input interferometer",
    "mzi_ps": "two-input interferometer with an internal heater",
    "OSU": "two-by-two crossbar switch",
    "mrr": "microring with three outputs",
}
BANNED = (
    "mmi1x2",
    "straight_heat_metal",
    "mzi_2x2",
    "mzi_ps",
    "mzm_dual",
    "mmi",
    "mzm",
    "mzi",
    "mrr",
    "coupler",
    "straight",
    "OSU",
    "SAX",
    "sax",
)


def largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    denom = sum(weights.values())
    raw = {key: total * value / denom for key, value in weights.items()}
    floors = {key: int(raw[key]) for key in raw}
    left = total - sum(floors.values())
    order = sorted(raw, key=lambda key: (raw[key] - floors[key], weights[key]), reverse=True)
    for key in order[:left]:
        floors[key] += 1
    if sum(floors.values()) != total:
        raise RuntimeError(floors)
    return floors


def init_sax() -> None:
    pic = builder.PIC
    sys.path.insert(0, str(pic))
    import os

    os.chdir(pic)
    import sax as sax_mod
    from devices import (  # noqa: WPS433
        OSU,
        coupler,
        mmi1x2,
        mrr,
        mzi,
        mzi_2x2,
        mzi_ps,
        mzm,
        mzm_dual,
        straight,
        straight_heat_metal,
    )

    builder.sax = sax_mod
    builder.SAX_MODELS = {
        "OSU": OSU,
        "coupler": coupler,
        "mmi1x2": mmi1x2,
        "mrr": mrr,
        "mzi": mzi,
        "mzi_2x2": mzi_2x2,
        "mzi_ps": mzi_ps,
        "mzm": mzm,
        "mzm_dual": mzm_dual,
        "straight": straight,
        "straight_heat_metal": straight_heat_metal,
    }


def accept(record: dict, corpus: dict, seen: set[str]) -> bool:
    key = json.dumps({"template": record["template"], "spec": record["spec"]}, sort_keys=True)
    if key in seen:
        return False
    if builder.forbidden_text(record["question"]):
        return False
    if builder.overlap_size(record["question"], corpus) >= builder.SHINGLE:
        return False
    builder.sax_ok(record["reference"])
    seen.add(key)
    return True


def take(records: list[dict], made: list[dict], corpus: dict, seen: set[str], quota: int) -> None:
    for record in records:
        if len(made) >= quota:
            return
        if accept(record, corpus, seen):
            made.append(record)
    if len(made) < quota:
        raise SystemExit(f"filled {len(made)} of {quota}")


def split_count(total: int, parts: dict[str, int]) -> dict[str, int]:
    return largest_remainder(parts, total)


def build_reweight(rng: random.Random, corpus: dict) -> list[dict]:
    targets = largest_remainder(REAUDIT, 1000)
    wrong = split_count(targets["wrong_model"], {"sandwich": 120, "two": 80, "wire": 80})
    ports_split = targets["wrong_ports"] - len(builder.RING_CWLS)
    if ports_split < 0:
        raise SystemExit("port quota is smaller than the ring list")
    functional = split_count(targets["functional"], {"asym": 60, "modulator": 40, "coupler": 20})
    graft = split_count(targets["graft"], {"ring": 1, "switch": 1})
    seen: set[str] = set()
    made: list[dict] = []

    widths = [2, 3, 5, 6]
    depths = [4, 5, 6, 7, 9]
    blocks = ["mzi", "mzm", "mzm_dual"]
    phase0s = [3, 7, 11, 13, 17]
    lengths = [12, 21, 37]
    bundles = [
        builder.parallel_bundle(w, d, b, p, L, group=f"rw-bundle-{b}")
        for w in widths
        for d in depths
        for b in blocks
        for p in phase0s
        for L in lengths
    ]
    rng.shuffle(bundles)
    before = len(made)
    take(bundles, made, corpus, seen, before + targets["ellipsis"])

    phase_pool = [12, 20, 25, 33, 48, 57, 66, 78, 85, 95, 110, 125, 140, 155, 165, 200, 230, 260, 300]
    length_pool = [11, 13, 14, 16, 18, 19, 23, 26, 28, 32, 34, 38, 41, 44, 47, 55, 62, 71]
    heat: list[dict] = []
    heat_seen: set[str] = set()
    while len(heat) < targets["comment"] + 20:
        k = rng.choice([4, 5, 6, 7])
        phases = rng.sample(phase_pool, k)
        lens = rng.sample(length_pool, k)
        key = json.dumps([phases, lens])
        if key in heat_seen:
            continue
        heat_seen.add(key)
        heat.append(builder.phase_chain(phases, lens, group=f"rw-heat-{len(heat)}"))
    before = len(made)
    take(heat, made, corpus, seen, before + targets["comment"])

    sandwiches = [
        builder.sandwiched(middle, lead, phase, group=f"rw-sand-{lead}-{phase}")
        for middle in ("mzi", "mzm", "mzm_dual")
        for lead in (13, 17, 22, 29, 36, 43)
        for phase in (15, 28, 42, 63, 81, 96, 124)
    ]
    two_ports = [
        builder.two_port(model, len1, len2, group=f"rw-two-{len1}-{len2}")
        for model in ("coupler", "mzi_2x2", "mzi_ps", "OSU")
        for len1 in (12, 19, 27, 33, 46)
        for len2 in (15, 24, 31, 52)
    ]
    wires = [
        builder.wire(model, length, phase, group=f"rw-wire-{length}-{phase}")
        for model in ("straight", "straight_heat_metal")
        for length in (12, 16, 22, 28, 35, 48, 60, 74)
        for phase in (18, 45, 72, 110, 150)
    ]
    rng.shuffle(sandwiches)
    rng.shuffle(two_ports)
    rng.shuffle(wires)
    before = len(made)
    take(sandwiches, made, corpus, seen, before + wrong["sandwich"])
    before = len(made)
    take(two_ports, made, corpus, seen, before + wrong["two"])
    before = len(made)
    take(wires, made, corpus, seen, before + wrong["wire"])

    rings = [builder.ring_ports(cwl, group=f"rw-ring-{cwl}") for cwl in builder.RING_CWLS]
    splits = [
        builder.split_arms(len1, len2, group=f"rw-split-{len1}-{len2}")
        for len1 in (12, 15, 18, 21, 24, 27, 30, 36, 42, 48)
        for len2 in (14, 17, 20, 23, 26, 33, 39, 45, 51, 57)
    ]
    rng.shuffle(splits)
    before = len(made)
    take(rings + splits, made, corpus, seen, before + targets["wrong_ports"])

    asym = [
        builder.asymmetric_mzi(len1, len2, phase, group=f"rw-asym-{len1}-{len2}-{phase}")
        for len1 in (12, 14, 16, 18, 22, 26, 28, 32, 36, 41, 45, 52, 58, 66, 74)
        for len2 in (12, 14, 16, 18, 22, 26, 28, 32, 36, 41, 45, 52, 58, 66, 74)
        if len1 != len2
        for phase in (15, 28, 42, 63, 75, 96, 110, 124, 140, 165, 210, 240)
    ]
    modulators = [
        builder.modulator_line(kind, count, phase, group=f"rw-line-{kind}-{count}-{phase}")
        for kind in ("mzm", "mzm_dual")
        for count in (2, 3, 5, 6)
        for phase in (24, 48, 72, 96, 132)
    ]
    couplers = [
        builder.coupler_phase_line(phase, group=f"rw-cphase-{phase}")
        for phase in (20, 30, 35, 45, 50, 55, 60, 75, 80, 90, 100, 105, 120, 135, 150, 160, 210, 240, 270, 300)
    ]
    rng.shuffle(asym)
    before = len(made)
    take(asym, made, corpus, seen, before + functional["asym"])
    before = len(made)
    take(modulators, made, corpus, seen, before + functional["modulator"])
    before = len(made)
    take(couplers, made, corpus, seen, before + functional["coupler"])

    ring_grafts = [
        builder.mzi_into_ring(len1, len2, phase, cwl, group=f"rw-graft-ring-{len1}-{len2}-{phase}-{cwl}")
        for len1 in range(12, 40, 2)
        for len2 in range(14, 56, 3)
        if len1 != len2
        for phase in (24, 63, 110, 165, 210)
        for cwl in (1.546, 1.547, 1.552, 1.55, 1.554)
    ]
    switches = [
        builder.switch_into_modulator(length, group=f"rw-graft-sw-{length}")
        for length in range(11, 240)
    ]
    rng.shuffle(ring_grafts)
    rng.shuffle(switches)
    before = len(made)
    take(ring_grafts, made, corpus, seen, before + graft["ring"])
    before = len(made)
    take(switches, made, corpus, seen, before + graft["switch"])
    if len(made) != 1000:
        raise SystemExit(f"reweight built {len(made)}")
    return made


def build_random(rng: random.Random, corpus: dict, target: int) -> list[dict]:
    widths = [2, 3, 5, 6]
    depths = [4, 5, 6, 7, 9]
    seen: set[str] = set()
    made: list[dict] = []
    attempts = 0
    while len(made) < target:
        attempts += 1
        if attempts > target * 40:
            raise SystemExit(f"random legal stopped at {len(made)}")
        kind = rng.randrange(12)
        try:
            if kind == 0:
                record = builder.parallel_bundle(
                    rng.choice(widths), rng.choice(depths), rng.choice(["mzi", "mzm", "mzm_dual"]),
                    rng.choice([3, 7, 11, 13, 17]), rng.choice([12, 21, 37]), group="rand-bundle",
                )
            elif kind == 1:
                k = rng.choice([4, 5, 6, 7])
                record = builder.phase_chain(
                    rng.sample(range(10, 320), k), rng.sample(range(11, 80), k), group="rand-heat",
                )
            elif kind == 2:
                record = builder.sandwiched(
                    rng.choice(["mzi", "mzm", "mzm_dual"]), rng.choice(range(12, 50)), rng.choice(range(10, 180)),
                    group="rand-sand",
                )
            elif kind == 3:
                record = builder.two_port(
                    rng.choice(["coupler", "mzi_2x2", "mzi_ps", "OSU"]),
                    rng.choice(range(12, 60)), rng.choice(range(12, 60)), group="rand-two",
                )
            elif kind == 4:
                record = builder.wire(
                    rng.choice(["straight", "straight_heat_metal"]),
                    rng.choice(range(12, 80)), rng.choice(range(10, 200)), group="rand-wire",
                )
            elif kind == 5:
                record = builder.ring_ports(rng.choice(builder.RING_CWLS), group="rand-ring")
            elif kind == 6:
                record = builder.split_arms(rng.choice(range(12, 60)), rng.choice(range(12, 60)), group="rand-split")
            elif kind == 7:
                len1 = rng.choice(range(12, 80))
                len2 = rng.choice([value for value in range(12, 80) if value != len1])
                record = builder.asymmetric_mzi(len1, len2, rng.choice(range(10, 300)), group="rand-asym")
            elif kind == 8:
                record = builder.modulator_line(
                    rng.choice(["mzm", "mzm_dual"]), rng.choice([2, 3, 5, 6]), rng.choice(range(10, 200)),
                    group="rand-line",
                )
            elif kind == 9:
                record = builder.coupler_phase_line(rng.choice(range(10, 300)), group="rand-cphase")
            elif kind == 10:
                record = builder.mzi_into_ring(
                    rng.choice(range(12, 40)), rng.choice(range(14, 56)), rng.choice(range(10, 220)),
                    rng.choice([1.546, 1.547, 1.55, 1.552, 1.554]), group="rand-graft-ring",
                )
            else:
                record = builder.switch_into_modulator(rng.choice(range(11, 240)), group="rand-graft-sw")
        except ValueError:
            continue
        if accept(record, corpus, seen):
            made.append(record)
    return made


def sax_id(value: object, models: dict) -> str:
    if isinstance(value, str):
        return models.get(value, value)
    return models[value["component"]]


def component_key(value: object) -> str:
    if isinstance(value, str):
        return value
    return value["component"]


def rewrite_noname(record: dict) -> dict:
    doc = json.loads(json.dumps(record["reference"]))
    netlist = doc["netlist"]
    models = doc["models"]
    old_names = list(netlist["instances"])
    new_names = {old: f"u{index:02d}" for index, old in enumerate(old_names, start=1)}
    id_to_kind: dict[str, str] = {}
    for value in netlist["instances"].values():
        library = sax_id(value, models)
        if library not in PHRASE:
            raise ValueError(f"no phrase for {library}")
        id_to_kind.setdefault(library, f"kind{len(id_to_kind) + 1:02d}")
    new_instances = {}
    for old, value in netlist["instances"].items():
        library = sax_id(value, models)
        kind = id_to_kind[library]
        if isinstance(value, str):
            new_instances[new_names[old]] = kind
        else:
            new_instances[new_names[old]] = {"component": kind, "settings": value.get("settings", {})}
    new_connections = {}
    for left, right in netlist["connections"].items():
        left_name, left_port = left.split(",", 1)
        right_name, right_port = right.split(",", 1)
        new_connections[f"{new_names[left_name]},{left_port}"] = f"{new_names[right_name]},{right_port}"
    new_ports = {}
    for port, binding in netlist["ports"].items():
        name, device_port = binding.split(",", 1)
        new_ports[port] = f"{new_names[name]},{device_port}"
    new_models = {kind: library for library, kind in id_to_kind.items()}
    rewritten = {
        "netlist": {"instances": new_instances, "connections": new_connections, "ports": new_ports},
        "models": new_models,
    }
    lines = ["Build the circuit named by the following devices, connections, and ports."]
    lines.append("Each kind shares one component key. The models entry for that key is the library identifier of the kind. The identifier is not written here.")
    for old, value in netlist["instances"].items():
        library = sax_id(value, models)
        phrase = PHRASE[library]
        kind = id_to_kind[library]
        settings = ""
        if isinstance(value, dict) and value.get("settings"):
            parts = [f"{key} {val}" for key, val in value["settings"].items()]
            settings = "; settings " + ", ".join(parts)
        lines.append(f"{new_names[old]} is {phrase}, component key {kind}{settings}.")
    for left, right in new_connections.items():
        lines.append(f"Connect {left} to {right}.")
    for port, binding in new_ports.items():
        lines.append(f"Bind port {port} to {binding}.")
    question = "\n".join(lines) + "\n"
    for banned in BANNED:
        if banned in question:
            raise ValueError(f"banned token {banned} in prompt")
    return {
        "locus": record["locus"],
        "template": record["template"],
        "group": record.get("group", ""),
        "question": question,
        "spec": record["spec"],
        "reference": rewritten,
        "source_id": record["id"],
    }


def prompt_messages(question: str, reference: dict) -> dict:
    template = json.loads(PICBENCH_VISIBLE_JSONL.open(encoding="utf-8").readline())
    messages = template["messages"]
    question = question.strip()
    rendered = json.dumps(reference, ensure_ascii=False, indent=2)
    answer = f"<result>\n{rendered}\n</result>"
    query = f"{question}<|im_end|>\n<|im_start|>user\n{question}"
    return {
        "messages": [
            {"role": "system", "content": messages[0]["content"]},
            {"role": "user", "content": messages[1]["content"]},
            {"role": "assistant", "content": messages[2]["content"], "loss": False},
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
    }


def store(name: str, records: list[dict], strategy: str, seed: int, extra: dict) -> None:
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    by_locus: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        record["id"] = f"q{index:04d}"
        record["strategy"] = strategy
        by_locus[record["locus"]] = by_locus.get(record["locus"], 0) + 1
    questions = directory / "questions.jsonl"
    with questions.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    training = OUT / f"{name}.reference.jsonl"
    with training.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(prompt_messages(record["question"], record["reference"]), ensure_ascii=False) + "\n")
    manifest = {
        "strategy": strategy,
        "seed": seed,
        "count": len(records),
        "by_locus": by_locus,
        "questions": str(questions),
        "reference_jsonl": str(training),
        "sax": "eight wavelengths, finite transmission",
        "not_merged_into": str(QUESTIONS),
    }
    manifest.update(extra)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"name": name, "count": len(records), "by_locus": by_locus}, sort_keys=True))


def main() -> None:
    init_sax()
    corpus = builder.build_corpus()
    store(
        "reweight",
        build_reweight(random.Random(20261025), corpus),
        "re-audit largest remainder",
        20261025,
        {"weights": REAUDIT, "scaled_to": 1000},
    )
    store(
        "random_legal",
        build_random(random.Random(20261024), corpus, 985),
        "uniform constructor, not a failure-locus allocation",
        20261024,
        {"target": 985},
    )

    dropped = []
    kept = []
    seen: set[str] = set()
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        try:
            record = rewrite_noname(source)
        except ValueError as exc:
            dropped.append({"id": source.get("id"), "reason": str(exc)})
            continue
        if not accept(record, corpus, seen):
            dropped.append({"id": source.get("id"), "reason": "shingle or duplicate"})
            continue
        kept.append(record)
    NONAME.mkdir(parents=True, exist_ok=True)
    with (NONAME / "questions.jsonl").open("w", encoding="utf-8") as handle:
        for index, record in enumerate(kept, start=1):
            record["id"] = f"n{index:04d}"
            record["strategy"] = "functional prose without a SAX model id"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    training = OUT / "noname.reference.jsonl"
    with training.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(prompt_messages(record["question"], record["reference"]), ensure_ascii=False) + "\n")
    manifest = {
        "strategy": "functional prose without a SAX model id",
        "count": len(kept),
        "dropped": len(dropped),
        "dropped_head": dropped[:12],
        "glossary": PHRASE,
        "questions": str(NONAME / "questions.jsonl"),
        "reference_jsonl": str(training),
        "not_merged_into": str(QUESTIONS),
        "note": "The prompt names the kind. The reference JSON stores the library id. The id is not in the prompt.",
    }
    (NONAME / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"name": "noname", "count": len(kept), "dropped": len(dropped)}))


if __name__ == "__main__":
    main()
