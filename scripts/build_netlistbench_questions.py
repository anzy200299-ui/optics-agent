#!/usr/bin/env python3
"""Build 1000 NetlistBench-style questions aimed at Qwen3.5-9B's measured failures.

Strategy: failure-locus verbalization, carried over from the PICBench set.

The official structural oracle, not a text heuristic, scored the finished
Qwen3.5-9B rollout (2342/2342). Edit pass rate is 545/1642. Recognition is
425/700. The misses that dominate are long flat edit sequences, edits inside
a subcircuit, subcircuit port swaps, inline expansion, pin reconnects,
MOS and source pin order on device add, device replacement, equivalence
labels, and the recognition tasks T2a, T3, and T6. Parameter edits,
device removal, T1, T2b, and T4 are already near ceiling and are not
allocated a share.

Each question is prose plus a source listing this script constructs. The
listing the oracle should accept is stored as the reference and is not
written into the question. A reference is kept only after the official
parser and comparator accept it. Flat sequence lengths 3, 6, 9, 12, and 15,
and subcircuit sequence lengths 3, 6, and 9, are held out with the released
benchmark. An 8-word shingle shared with any released instruction, or a
rename-invariant fingerprint shared with any released netlist, rejects the item.

These items are a NetlistBench training pool. They are not PICBench items
and they do not change the PICBench Pass@k table.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from repo_paths import REPO

ROOT = REPO
NB = ROOT / "third_party" / "NetlistBench"
NB_SCRIPTS = NB / "scripts"
CASES = NB / "benchmark" / "Cases"
QWAN = ROOT / "logs" / "netlistbench" / "qwen35-9b"
OUT_DIR = ROOT / "data" / "netlistbench_questions"
SCORE_PATH = ROOT / "logs" / "netlistbench" / "qwen35-9b-score.json"

sys.path.insert(0, str(NB_SCRIPTS))
from eval_ir_match import compare_ir, format_normalized_ir  # noqa: E402
from eval_recognition import eval_t2a, eval_t3, eval_t6  # noqa: E402
from generate_recognition_cases import expected_t2a, expected_t3, expected_t6  # noqa: E402
from ir_numeric import parse_numeric_literal  # noqa: E402
from netlist_to_ir import parse_netlist_text  # noqa: E402

SEED = 20261020
N_TARGET = 1000
SHINGLE = 8
HELD_OUT_FLAT = {3, 6, 9, 12, 15}
HELD_OUT_SUB = {3, 6, 9}
GLOBAL_NETS = {"vdd", "vcc", "vss", "gnd", "vee", "0"}
PASSIVE_KINDS = {"resistor", "capacitor", "inductor"}

# Fail counts read off the official Qwen reports. The generator checks these
# against the report files before it allocates anything.
LOCUS_FAILS = {
    "flat_compound": 481,
    "subckt_compound": 285,
    "port_swap": 91,
    "inline_expand": 45,
    "connectivity": 60,
    "device_add": 39,
    "device_replace": 37,
    "equivalence": 43,
    "neighbor_incidence": 98,
    "semantic_pins": 88,
    "node_incidence": 80,
}
FLAT_FAMILIES = {
    "basic_compound_edit_sequence_12step": 100,
    "basic_compound_edit_sequence_15step": 97,
    "basic_compound_edit_sequence_3step": 86,
    "basic_compound_edit_sequence_6step": 98,
    "basic_compound_edit_sequence_9step": 100,
}
SUB_FAMILIES = {
    "subckt_internal_compound_3step": 87,
    "subckt_internal_compound_6step": 98,
    "subckt_internal_compound_9step": 100,
}
OTHER_FAMILY_FAILS = {
    "subckt_port_swap": 91,
    "subckt_inline_expand": 45,
    "connectivity_edit": 60,
    "device_add": 39,
    "device_replace": 37,
    "equivalence_judgment_mixed": 43,
    "parameter_edit": 1,
    "device_remove": 4,
    "rename_propagation": 11,
}
STEP_FAILS = {
    "pin": 217,
    "replace": 189,
    "add": 132,
    "rename": 80,
    "remove": 65,
    "param": 18,
}
# Neighbor lengths of the held-out released widths, weighted by that width's misses.
FLAT_LENGTH_WEIGHTS = {4: 86, 5: 98, 7: 98, 8: 100, 10: 100, 11: 100, 13: 100, 14: 97}
SUB_LENGTH_WEIGHTS = {4: 87, 5: 93, 7: 98, 8: 99, 10: 100, 11: 100}

VALUES = {
    "R": ["100", "220", "470", "1k", "2.2k", "4.7k", "10k"],
    "C": ["1p", "2p", "5p", "10p", "22p"],
    "L": ["10n", "22n", "47n", "100n"],
    "V": ["0.6", "0.9", "1.2", "1.8"],
    "I": ["10u", "20u", "50u", "100u"],
}
WIDTHS = ["0.5u", "1u", "2u", "4u"]
LENGTHS = ["0.18u", "0.25u", "0.35u", "0.5u"]
MOS_ROLES = ["drain", "gate", "source", "bulk"]

EDIT_HEAD = (
    "Rewrite the entire SPICE listing so that every numbered change below is present at once. "
    "Any statement a change does not mention stays word for word. "
    "The last line of the listing is .end. "
    "Return the listing alone."
)
REC_HEAD = "Answer from the SPICE listing with a single JSON object and no surrounding words."
EQUIV_HEAD = (
    "Two SPICE listings follow. Say whether they are the same structure. "
    "A resistor, a capacitor, or an inductor is unchanged when its two ends are exchanged. "
    "Numeric suffixes scale, so 1k and 1000 match. "
    "A voltage source, a current source, or a MOSFET is different when its terminal order changes. "
    "Reply with the single word yes or the single word no."
)
BANNED_PROMPTS = (EDIT_HEAD, REC_HEAD, EQUIV_HEAD)


class BuildError(RuntimeError):
    pass


class Dev:
    def __init__(self, name, kind, nodes, value=None, model=None, w=None, l=None):
        self.name = name
        self.kind = kind
        self.nodes = list(nodes)
        self.value = value
        self.model = model
        self.w = w
        self.l = l

    def line(self) -> str:
        if self.kind == "M":
            return f"{self.name} {' '.join(self.nodes)} {self.model} W={self.w} L={self.l}"
        if self.kind in {"R", "C", "L", "V", "I"}:
            return f"{self.name} {self.nodes[0]} {self.nodes[1]} {self.value}"
        raise BuildError(f"bad device {self.kind}")


class Sub:
    def __init__(self, name: str, ports: list[str]):
        self.name = name
        self.ports = list(ports)
        self.devs: list[Dev] = []


class Inst:
    def __init__(self, name: str, nodes: list[str], sub: str):
        self.name = name
        self.nodes = list(nodes)
        self.sub = sub


class Circ:
    def __init__(self):
        self.top: list[Dev] = []
        self.subs: dict[str, Sub] = {}
        self.insts: list[Inst] = []
        self.seq = 0

    def alloc(self, prefix: str) -> str:
        self.seq += 1
        return f"{prefix}{self.seq}"


def render(c: Circ) -> str:
    lines: list[str] = []
    for name in sorted(c.subs):
        sub = c.subs[name]
        lines.append(f".subckt {name} {' '.join(sub.ports)}".rstrip())
        for dev in sub.devs:
            lines.append(dev.line())
        lines.append(".ends")
    for dev in c.top:
        lines.append(dev.line())
    for inst in c.insts:
        lines.append(f"{inst.name} {' '.join(inst.nodes)} {inst.sub}")
    lines.append(".end")
    return "\n".join(lines) + "\n"


def load_ir(text: str) -> dict:
    try:
        return format_normalized_ir(parse_netlist_text(text))
    except (ValueError, json.JSONDecodeError) as exc:
        raise BuildError(str(exc)) from exc


def differs(left: str, right: str) -> bool:
    return bool(compare_ir(load_ir(left), load_ir(right)))


def same(left: str, right: str) -> bool:
    return not differs(left, right)


def numeric_eq(left: str, right: str) -> bool:
    a = parse_numeric_literal(left)
    b = parse_numeric_literal(right)
    if a is None or b is None:
        return left == right
    return a == b


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(text: str, n: int = SHINGLE) -> set[tuple[str, ...]]:
    words = tokens(text)
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def overlap_size(text: str, banned: set[tuple[str, ...]]) -> int:
    return 1 if shingles(text) & banned else 0


def largest_remainder(weights: dict, total: int) -> dict[str, int]:
    names = list(weights)
    denom = sum(weights.values())
    raw = {name: total * weights[name] / denom for name in names}
    floors = {name: int(raw[name]) for name in names}
    left = total - sum(floors.values())
    order = sorted(names, key=lambda name: (-(raw[name] - floors[name]), name))
    for name in order[:left]:
        floors[name] += 1
    if sum(floors.values()) != total:
        raise RuntimeError(floors)
    return floors


def expand_counts(counts: dict) -> list:
    rows = []
    for key in sorted(counts, key=lambda item: (str(type(item)), str(item))):
        rows.extend([key] * counts[key])
    return rows


def pick_weighted(rng: random.Random, pairs: list[tuple[str, int]]) -> str:
    total = sum(weight for _, weight in pairs)
    draw = rng.randrange(total)
    for name, weight in pairs:
        if draw < weight:
            return name
        draw -= weight
    raise RuntimeError(pairs)


def scope_devs(c: Circ, scope: str | None) -> list[Dev]:
    if scope is None:
        return c.top
    return c.subs[scope].devs


def used_nodes(devs: list[Dev]) -> list[str]:
    found = []
    seen = set()
    for dev in devs:
        for node in dev.nodes:
            if node not in seen:
                seen.add(node)
                found.append(node)
    return found


def pool_of(c: Circ, scope: str | None) -> list[str]:
    if scope is None:
        return used_nodes(c.top) or ["vdd", "vss"]
    sub = c.subs[scope]
    return list(dict.fromkeys(sub.ports + used_nodes(sub.devs) + ["vdd", "vss"]))


def outside_sig(c: Circ):
    return (
        tuple(dev.line() for dev in c.top),
        tuple((name, tuple(c.subs[name].ports)) for name in sorted(c.subs)),
        tuple((inst.name, tuple(inst.nodes), inst.sub) for inst in c.insts),
    )


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def mutate(c: Circ, scope: str | None, rng: random.Random, mode: str, tag: str) -> str | None:
    devs = scope_devs(c, scope)
    pool = pool_of(c, scope)

    def prefix(sentence: str) -> str:
        if scope is None:
            return sentence
        return f"Inside subcircuit {scope} only, {sentence[0].lower()}{sentence[1:]}"

    if mode == "pin":
        mos = [dev for dev in devs if dev.kind == "M"]
        if not mos:
            return None
        dev = rng.choice(mos)
        index = rng.randrange(4)
        old = dev.nodes[index]
        choices = [node for node in pool if node != old]
        if not choices:
            return None
        new = rng.choice(choices)
        dev.nodes[index] = new
        return prefix(
            f"Move only the {MOS_ROLES[index]} of {dev.name} from {old} to {new}. "
            "Leave its other pins untouched."
        )

    if mode in {"vpin", "ipin"}:
        kind = "V" if mode == "vpin" else "I"
        found = [dev for dev in devs if dev.kind == kind]
        if not found:
            return None
        dev = rng.choice(found)
        index = rng.randrange(2)
        old = dev.nodes[index]
        choices = [node for node in pool if node != old]
        if not choices:
            return None
        new = rng.choice(choices)
        dev.nodes[index] = new
        if mode == "vpin":
            moved = "positive" if index == 0 else "negative"
            kept = "negative" if index == 0 else "positive"
            return prefix(
                f"Move the {moved} terminal of {dev.name} from {old} to {new}. "
                f"Leave the {kept} terminal in place."
            )
        moved = "upstream" if index == 0 else "downstream"
        return prefix(
            f"Move the {moved} terminal of current source {dev.name} from {old} to {new}. "
            "Current still flows from the upstream terminal toward the downstream terminal."
        )

    if mode in {"replace_rc", "replace_cl", "replace_rv", "replace_mos"}:
        if mode == "replace_rc":
            found = [dev for dev in devs if dev.kind == "R"]
            new_kind, prefix_name = "C", "C"
        elif mode == "replace_cl":
            found = [dev for dev in devs if dev.kind == "C"]
            new_kind, prefix_name = "L", "L"
        elif mode == "replace_rv":
            found = [dev for dev in devs if dev.kind == "R"]
            new_kind, prefix_name = "V", "V"
        else:
            found = [dev for dev in devs if dev.kind == "M"]
            new_kind, prefix_name = "M", "M"
        if not found:
            return None
        old = rng.choice(found)
        name = c.alloc(prefix_name)
        if new_kind == "M":
            model = "PMOS" if str(old.model).upper() == "NMOS" else "NMOS"
            width = rng.choice([item for item in WIDTHS if item != old.w] or WIDTHS)
            newer = Dev(name, "M", list(old.nodes), model=model, w=width, l=old.l)
            sentence = (
                f"Remove {old.name}. Add {model} transistor {name} with drain {old.nodes[0]}, "
                f"gate {old.nodes[1]}, source {old.nodes[2]}, and bulk {old.nodes[3]}. "
                f"Width {width}, length {old.l}."
            )
        elif new_kind == "V":
            value = rng.choice(VALUES["V"])
            newer = Dev(name, "V", list(old.nodes), value=value)
            sentence = (
                f"Remove {old.name}. Add voltage source {name} set to {value} volt, "
                f"positive terminal on {old.nodes[0]}, negative terminal on {old.nodes[1]}."
            )
        else:
            value = rng.choice(VALUES[new_kind])
            newer = Dev(name, new_kind, list(old.nodes), value=value)
            kind_name = {"C": "capacitor", "L": "inductor"}[new_kind]
            sentence = (
                f"Remove {old.name}. Add {kind_name} {name} of {value} between "
                f"{old.nodes[0]} and {old.nodes[1]}."
            )
        devs[devs.index(old)] = newer
        return prefix(sentence)

    if mode in {"add_mos", "add_v", "add_i"}:
        if len(pool) < 2:
            return None
        if mode == "add_mos":
            pins = [rng.choice(pool) for _ in range(4)]
            name = c.alloc("M")
            model = rng.choice(["NMOS", "PMOS"])
            width = rng.choice(WIDTHS)
            length = rng.choice(LENGTHS)
            devs.append(Dev(name, "M", pins, model=model, w=width, l=length))
            sentence = (
                f"Add {model} transistor {name} with drain {pins[0]}, gate {pins[1]}, "
                f"source {pins[2]}, and bulk {pins[3]}. Width {width}, length {length}."
            )
        elif mode == "add_v":
            left, right = rng.sample(pool, 2)
            name = c.alloc("V")
            value = rng.choice(VALUES["V"])
            devs.append(Dev(name, "V", [left, right], value=value))
            sentence = (
                f"Add voltage source {name} set to {value} volt, "
                f"positive terminal on {left}, negative terminal on {right}."
            )
        else:
            left, right = rng.sample(pool, 2)
            name = c.alloc("I")
            value = rng.choice(VALUES["I"])
            devs.append(Dev(name, "I", [left, right], value=value))
            sentence = (
                f"Add current source {name} set to {value} ampere, "
                f"upstream terminal on {left}, downstream terminal on {right}."
            )
        return prefix(sentence)

    if mode == "rename":
        nodes = used_nodes(devs)
        if scope is not None:
            ports = set(c.subs[scope].ports)
            nodes = [node for node in nodes if node not in ports]
        if not nodes:
            return None
        old = rng.choice(nodes)
        new = f"{tag}z{c.alloc('n')[1:]}"
        for dev in devs:
            dev.nodes = [new if node == old else node for node in dev.nodes]
        return prefix(f"Rename net {old} to {new} wherever that net is used here. Do not rename any other net.")

    if mode == "remove":
        if len(devs) < 4:
            return None
        dev = rng.choice(devs)
        devs.remove(dev)
        return prefix(f"Delete {dev.name} without adding a substitute.")

    if mode == "param":
        found = [dev for dev in devs if dev.kind in VALUES or dev.kind == "M"]
        if not found:
            return None
        dev = rng.choice(found)
        if dev.kind == "M":
            choices = [item for item in WIDTHS if item != dev.w]
            dev.w = rng.choice(choices)
            return prefix(
                f"Change the width of {dev.name} to {dev.w}. Keep its length and all four pins."
            )
        choices = [item for item in VALUES[dev.kind] if not numeric_eq(item, dev.value)]
        old = dev.value
        dev.value = rng.choice(choices)
        return prefix(f"Change the value of {dev.name} from {old} to {dev.value}. Keep both terminals.")

    raise BuildError(mode)


def coarse_mode(rng: random.Random, coarse: str) -> str:
    if coarse == "pin":
        return rng.choice(["pin", "pin", "pin", "pin", "vpin", "ipin"])
    if coarse == "replace":
        return rng.choice(["replace_rc", "replace_cl", "replace_rv", "replace_mos"])
    if coarse == "add":
        return rng.choice(["add_mos", "add_mos", "add_v", "add_i"])
    return coarse


def apply_step(c: Circ, scope: str | None, rng: random.Random, mode: str, tag: str):
    before = render(c)
    outside = outside_sig(c)
    sentence = mutate(c, scope, rng, mode, tag)
    if not sentence:
        return None
    after = render(c)
    if not differs(before, after):
        return None
    if scope is not None and outside_sig(c) != outside:
        return None
    return sentence


def try_step(c: Circ, scope: str | None, rng: random.Random, coarse: str, tag: str):
    modes = [coarse_mode(rng, coarse)]
    if coarse in STEP_FAILS:
        modes.extend(coarse_mode(rng, name) for name in STEP_FAILS if name != coarse)
    modes.append("add_mos")
    seen = set()
    for mode in modes:
        if mode in seen:
            continue
        seen.add(mode)
        for _ in range(8):
            trial = copy.deepcopy(c)
            sentence = apply_step(trial, scope, rng, mode, tag)
            if sentence:
                return trial, sentence
    return None


def build_flat(rng: random.Random, tag: str) -> Circ:
    c = Circ()
    signals = [f"{tag}n{i}" for i in range(rng.randint(8, 12))]
    pool = signals + ["vdd", "vss"]
    for _ in range(rng.randint(5, 8)):
        pins = [rng.choice(pool) for _ in range(4)]
        c.top.append(
            Dev(
                c.alloc("M"),
                "M",
                pins,
                model=rng.choice(["NMOS", "PMOS"]),
                w=rng.choice(WIDTHS),
                l=rng.choice(LENGTHS),
            )
        )
    for kind, count in (("R", rng.randint(2, 4)), ("C", 1), ("L", 1)):
        for _ in range(count):
            left, right = rng.sample(pool, 2)
            c.top.append(Dev(c.alloc(kind), kind, [left, right], value=rng.choice(VALUES[kind])))
    left, right = rng.sample(signals, 2)
    c.top.append(Dev(c.alloc("V"), "V", ["vdd", left], value=rng.choice(VALUES["V"])))
    c.top.append(Dev(c.alloc("I"), "I", [right, "vss"], value=rng.choice(VALUES["I"])))
    return c


def build_hier(rng: random.Random, tag: str, instances: int) -> tuple[Circ, str]:
    c = Circ()
    sub_name = f"b{tag}"
    ports = [f"{tag}a", f"{tag}b", f"{tag}c"]
    internal = [f"{tag}k{i}" for i in range(5)]
    sub = Sub(sub_name, ports)
    body_pool = ports + internal + ["vdd", "vss"]
    sub.devs.append(
        Dev(
            c.alloc("M"),
            "M",
            [ports[0], internal[0], ports[1], "vss"],
            model="NMOS",
            w=rng.choice(WIDTHS),
            l=rng.choice(LENGTHS),
        )
    )
    sub.devs.append(
        Dev(
            c.alloc("M"),
            "M",
            [ports[2], internal[1], internal[2], "vdd"],
            model="PMOS",
            w=rng.choice(WIDTHS),
            l=rng.choice(LENGTHS),
        )
    )
    for _ in range(2):
        pins = [rng.choice(body_pool) for _ in range(4)]
        sub.devs.append(
            Dev(
                c.alloc("M"),
                "M",
                pins,
                model=rng.choice(["NMOS", "PMOS"]),
                w=rng.choice(WIDTHS),
                l=rng.choice(LENGTHS),
            )
        )
    sub.devs.append(Dev(c.alloc("R"), "R", [ports[0], ports[2]], value=rng.choice(VALUES["R"])))
    sub.devs.append(Dev(c.alloc("C"), "C", [internal[3], ports[1]], value=rng.choice(VALUES["C"])))
    c.subs[sub_name] = sub
    top_nodes = [f"{tag}n{i}" for i in range(6)]
    for _ in range(instances):
        c.insts.append(Inst(c.alloc("XU"), [rng.choice(top_nodes) for _ in ports], sub_name))
    c.top.append(Dev(c.alloc("V"), "V", ["vdd", top_nodes[0]], value=rng.choice(VALUES["V"])))
    c.top.append(Dev(c.alloc("R"), "R", [top_nodes[1], top_nodes[2]], value=rng.choice(VALUES["R"])))
    return c, sub_name


def build_chain(rng: random.Random, tag: str, tie_gate_drain: bool) -> Circ:
    c = Circ()
    width = rng.randint(3, 6)
    nets = [f"{tag}n{i}" for i in range(width)]
    for index in range(width):
        drain = nets[index]
        gate = nets[index] if tie_gate_drain and index == 0 else (nets[index - 1] if index else f"{tag}g0")
        source = "vss" if index % 2 == 0 else nets[(index + 1) % width]
        bulk = "vss" if index % 2 == 0 else "vdd"
        c.top.append(
            Dev(
                c.alloc("M"),
                "M",
                [drain, gate, source, bulk],
                model="NMOS" if index % 2 == 0 else "PMOS",
                w=rng.choice(WIDTHS),
                l=rng.choice(LENGTHS),
            )
        )
    for index in range(rng.randint(1, 3)):
        c.top.append(
            Dev(
                c.alloc("R"),
                "R",
                [nets[index % width], "vss"],
                value=rng.choice(VALUES["R"]),
            )
        )
    c.top.append(Dev(c.alloc("V"), "V", ["vdd", nets[0]], value=rng.choice(VALUES["V"])))
    return c


def edit_question(steps: list[str], listing: str, scope: str | None) -> str:
    extra = ""
    if scope is not None:
        extra = (
            f" Leave the port list of {scope} unchanged, leave every instance of {scope} unchanged, "
            "and leave every device outside it unchanged."
        )
    numbered = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(steps, 1))
    return f"{EDIT_HEAD}{extra}\n\nChanges:\n{numbered}\n\nListing:\n{listing}"


def sequence_item(rng: random.Random, length: int, hierarchical: bool) -> dict:
    if hierarchical and length in HELD_OUT_SUB:
        raise BuildError(length)
    if not hierarchical and length in HELD_OUT_FLAT:
        raise BuildError(length)
    tag = f"q{rng.randrange(0x1000000):06x}"
    if hierarchical:
        current, scope = build_hier(rng, tag, 2)
        locus = "subckt_compound"
        template = "subckt_sequence"
    else:
        current = build_flat(rng, tag)
        scope = None
        locus = "flat_compound"
        template = "flat_sequence"
    source = render(current)
    load_ir(source)
    steps = []
    for _ in range(length):
        coarse = pick_weighted(rng, list(STEP_FAILS.items()))
        stepped = try_step(current, scope, rng, coarse, tag)
        if stepped is None:
            raise BuildError("step")
        current, sentence = stepped
        steps.append(sentence)
    expected = render(current)
    if not differs(source, expected):
        raise BuildError("noop sequence")
    question = edit_question(steps, source, scope)
    if expected.strip() in question:
        raise BuildError("answer leaked")
    return item(
        locus,
        template,
        f"steps-{length}",
        question,
        {"format": "spice", "text": expected},
        {"length": length, "scope": scope, "steps": steps},
        [source, expected],
    )


def single_edit(rng: random.Random, mode: str, locus: str, template: str) -> dict:
    tag = f"q{rng.randrange(0x1000000):06x}"
    current = build_flat(rng, tag)
    source = render(current)
    placed = False
    sentence = ""
    for _ in range(20):
        trial = copy.deepcopy(current)
        got = apply_step(trial, None, rng, mode, tag)
        if got:
            current = trial
            sentence = got
            placed = True
            break
    if not placed:
        raise BuildError(mode)
    expected = render(current)
    question = edit_question([sentence], source, None)
    if expected.strip() in question:
        raise BuildError("answer leaked")
    return item(
        locus,
        template,
        mode,
        question,
        {"format": "spice", "text": expected},
        {"steps": [sentence]},
        [source, expected],
    )


def port_swap_item(rng: random.Random, pair: tuple[int, int]) -> dict:
    tag = f"q{rng.randrange(0x1000000):06x}"
    current, scope = build_hier(rng, tag, 2)
    source = render(current)
    sub = current.subs[scope]
    left, right = pair
    port_a, port_b = sub.ports[left], sub.ports[right]
    sub.ports[left], sub.ports[right] = sub.ports[right], sub.ports[left]
    for inst in current.insts:
        inst.nodes[left], inst.nodes[right] = inst.nodes[right], inst.nodes[left]
    expected = render(current)
    if not differs(source, expected):
        raise BuildError("swap noop")
    sentence = (
        f"In subcircuit {scope}, exchange formal ports {port_a} and {port_b} in the header. "
        "Exchange those same argument positions on every instance of that subcircuit. "
        "Do not edit the devices written inside the subcircuit."
    )
    question = edit_question([sentence], source, None)
    if expected.strip() in question:
        raise BuildError("answer leaked")
    return item(
        "port_swap",
        "port_swap",
        f"ports-{left}-{right}",
        question,
        {"format": "spice", "text": expected},
        {"subckt": scope, "ports": [port_a, port_b]},
        [source, expected],
    )


def inline_item(rng: random.Random) -> dict:
    tag = f"q{rng.randrange(0x1000000):06x}"
    current, scope = build_hier(rng, tag, 1)
    source_circ = copy.deepcopy(current)
    source = render(source_circ)
    inst = current.insts[0]
    sub = current.subs[scope]
    mapping = dict(zip(sub.ports, inst.nodes))
    tag_name = "z1"
    copied = []
    for dev in sub.devs:
        newer = copy.deepcopy(dev)
        newer.name = f"{dev.name}_{tag_name}"
        newer.nodes = [mapping.get(node, node) for node in dev.nodes]
        copied.append(newer)
        current.top.append(newer)
    current.insts = []
    del current.subs[scope]
    expected = render(current)
    if not differs(source, expected):
        raise BuildError("inline noop")
    parsed = load_ir(expected)
    if parsed["subckts"]:
        raise BuildError("subckt remains")
    if inst.name.upper() in parsed["devices"]:
        raise BuildError("instance remains")
    for dev in copied:
        if dev.name.upper() not in parsed["devices"]:
            raise BuildError("copied device missing")
    names = ", ".join(dev.name for dev in copied)
    sentence = (
        f"Replace instance {inst.name} by copying the devices of subcircuit {scope} to the top level. "
        f"Map each formal port onto the instance argument in that position. "
        f"Keep every internal net that is not a formal port. "
        f"Name each copied device with the suffix _{tag_name}, giving {names}. "
        f"Then delete {inst.name} and delete subcircuit {scope}. "
        "Leave every device that was already at the top level unchanged."
    )
    question = edit_question([sentence], source, None)
    if expected.strip() in question:
        raise BuildError("answer leaked")
    return item(
        "inline_expand",
        "inline_expand",
        "flatten-one",
        question,
        {"format": "spice", "text": expected},
        {"instance": inst.name, "subckt": scope, "suffix": tag_name},
        [source, expected],
    )


def equivalence_item(rng: random.Random, mode: str) -> dict:
    tag = f"q{rng.randrange(0x1000000):06x}"
    left = build_flat(rng, tag)
    right = copy.deepcopy(left)
    if mode == "resistor_ends":
        dev = next(dev for dev in right.top if dev.kind == "R")
        dev.nodes = [dev.nodes[1], dev.nodes[0]]
        label = "yes"
    elif mode == "unit_scale":
        resistor = next(dev for dev in left.top if dev.kind == "R")
        other = next(dev for dev in right.top if dev.name == resistor.name)
        resistor.value = "1k"
        other.value = "1000"
        label = "yes"
    elif mode == "comment_only":
        label = "yes"
    elif mode == "voltage_polarity":
        dev = next(dev for dev in right.top if dev.kind == "V")
        dev.nodes = [dev.nodes[1], dev.nodes[0]]
        label = "no"
    elif mode == "mos_drain_source":
        dev = next(dev for dev in right.top if dev.kind == "M")
        dev.nodes[0], dev.nodes[2] = dev.nodes[2], dev.nodes[0]
        label = "no"
    elif mode == "resistor_value":
        dev = next(dev for dev in right.top if dev.kind == "R")
        dev.value = "10k" if not numeric_eq(dev.value, "10k") else "220"
        label = "no"
    else:
        raise BuildError(mode)
    listing_a = render(left)
    listing_b = "* annotation only\n" + render(right) if mode == "comment_only" else render(right)
    equivalent = same(listing_a, listing_b)
    if equivalent != (label == "yes"):
        raise BuildError(f"{mode} label {label} equal {equivalent}")
    if listing_a.strip() == listing_b.strip():
        raise BuildError("identical text")
    question = f"{EQUIV_HEAD}\n\nListing A:\n{listing_a}\nListing B:\n{listing_b}"
    return item(
        "equivalence",
        mode,
        mode,
        question,
        {"format": "label", "text": label},
        {"perturbation": mode},
        [listing_a, listing_b],
    )


def recognition_item(rng: random.Random, task: str, tie: bool) -> dict:
    tag = f"q{rng.randrange(0x1000000):06x}"
    spice = render(build_chain(rng, tag, tie_gate_drain=tie))
    ir = parse_netlist_text(spice)
    if task == "T2a":
        mos = [name for name, dev in ir["devices"].items() if dev["kind"] == "mos"]
        if not mos:
            raise BuildError("no mos")
        target = rng.choice(mos)
        expected = expected_t2a(ir, target)
        verdict, detail = eval_t2a(expected, expected)
        order = rng.choice(
            [("B", "G", "D", "S"), ("S", "B", "D", "G"), ("G", "S", "B", "D"), ("D", "B", "S", "G")]
        )
        asked = ", ".join(order)
        task_text = (
            f"For transistor {target}, give the net on each of bulk, gate, drain, and source. "
            f"Use JSON keys {asked}. The order of the keys is not a pin order."
        )
        locus, template = "semantic_pins", "semantic_pins"
        group = "T2a"
    elif task == "T3":
        chosen = None
        for node in sorted({n for dev in ir["devices"].values() for n in dev["nodes"]}):
            if node in GLOBAL_NETS:
                continue
            guess = expected_t3(ir, node)
            if len(guess["incidents"]) < 2:
                continue
            tied = _same_device_two_pins(guess["incidents"])
            if tie and not tied:
                continue
            if chosen is None or (tie and tied):
                chosen = (node, guess)
                if tie and tied:
                    break
        if chosen is None:
            raise BuildError("no node")
        node, expected = chosen
        verdict, detail = eval_t3(expected, expected, ir)
        task_text = (
            f"List every device pin that touches net {node}. "
            'Return JSON of the form {"node": "' + node + '", "incidents": [...]}. '
            "A MOSFET pin is written device:S, device:D, device:G, or device:B. "
            "A resistor, capacitor, inductor, voltage source, or current source is written as the device name alone. "
            "If one MOSFET touches the net with two pins, list both pins. Order does not matter."
        )
        locus, template = "node_incidence", "node_incidence"
        group = "T3-tied" if tie else "T3"
    elif task == "T6":
        target = _choose_t6(ir)
        if target is None:
            raise BuildError("no neighbor")
        expected = expected_t6(ir, target)
        if not expected["neighbors"]:
            raise BuildError("empty neighbors")
        verdict, detail = eval_t6(expected, expected, ir, target)
        task_text = (
            f"List the other device pins that share a net with {target['device']}:{target['terminal']}. "
            'Return JSON with keys "device", "terminal", and "neighbors". '
            "A MOSFET neighbor is device:S, device:D, device:G, or device:B. "
            "A resistor, capacitor, inductor, voltage source, or current source neighbor is the device name alone. "
            f"Do not include {target['device']}:{target['terminal']} itself. Order does not matter."
        )
        locus, template = "neighbor_incidence", "neighbor_incidence"
        group = "T6"
    else:
        raise BuildError(task)
    if verdict != "pass":
        raise BuildError(detail or verdict)
    question = f"{REC_HEAD}\n\n{task_text}\n\nListing:\n{spice}"
    dumped = json.dumps(expected, ensure_ascii=False)
    if dumped in question:
        raise BuildError("answer leaked")
    return item(
        locus,
        template,
        group,
        question,
        {"format": "json", "object": expected},
        {"task": task, "target": target if task == "T6" else (node if task == "T3" else target)},
        [spice],
    )


def _same_device_two_pins(incidents: list[str]) -> bool:
    counts: Counter = Counter()
    for atom in incidents:
        if ":" in atom:
            counts[atom.split(":", 1)[0]] += 1
    return any(count >= 2 for count in counts.values())


def _choose_t6(ir: dict) -> dict | None:
    best = None
    nodes = sorted({node for dev in ir["devices"].values() for node in dev["nodes"]})
    for node in nodes:
        if node in GLOBAL_NETS:
            continue
        incidents = expected_t3(ir, node)["incidents"]
        mos_pins = [atom for atom in incidents if ":" in atom]
        if len(incidents) < 2 or not mos_pins:
            continue
        mixed = any(":" not in atom for atom in incidents)
        dev, term = mos_pins[0].split(":", 1)
        rank = (0 if mixed else 1, -len(incidents), dev, term)
        if best is None or rank < best[0]:
            best = (rank, {"device": dev, "terminal": term, "via_net": node})
    return None if best is None else best[1]


def item(locus, template, group, question, reference, spec, netlists) -> dict:
    return {
        "locus": locus,
        "template": template,
        "group": group,
        "question": question.strip() + "\n",
        "spec": spec,
        "reference": reference,
        "_netlists": netlists,
    }


def scalar_sig(value) -> str:
    if isinstance(value, dict):
        if "numeric" in value or "raw" in value:
            numeric = value.get("numeric")
            return str(numeric if numeric is not None else value.get("raw"))
        parts = {str(key): scalar_sig(item) for key, item in sorted(value.items())}
        return json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return "" if value is None else str(value)


def _node_colors(devices: dict) -> dict[str, tuple]:
    incidents: dict[str, list] = {}
    for dev in devices.values():
        kind = dev.get("kind")
        nodes = dev.get("nodes", [])
        mark = (str(kind), str(dev.get("model") or ""), scalar_sig(dev.get("value")), scalar_sig(dev.get("params")))
        if kind in PASSIVE_KINDS:
            for node in set(nodes):
                incidents.setdefault(node, []).append((mark, "u"))
        else:
            for index, node in enumerate(nodes):
                incidents.setdefault(node, []).append((mark, index))
    return {node: tuple(sorted(pins)) for node, pins in incidents.items()}


def _scope_fp(devices: dict, sub_fp: dict[str, str]) -> list:
    if not devices:
        return []
    color = _node_colors(devices)
    records = []
    for dev in devices.values():
        kind = dev.get("kind")
        nodes = dev.get("nodes", [])
        if kind in PASSIVE_KINDS:
            ends = tuple(sorted(color.get(node, ()) for node in nodes))
        else:
            ends = tuple(color.get(node, ()) for node in nodes)
        if kind == "subckt_instance":
            extra = sub_fp.get(str(dev.get("subckt")), "")
        else:
            extra = (str(dev.get("model") or ""), scalar_sig(dev.get("value")), scalar_sig(dev.get("params")))
        records.append((str(kind), extra, ends))
    return sorted(records, key=lambda row: json.dumps(row, sort_keys=True, default=str))


def fingerprint(text: str) -> str:
    ir = parse_netlist_text(text)
    sub_fp: dict[str, str] = {}
    sub_rows = []
    for name, sub in sorted(ir.get("subckts", {}).items()):
        body = _scope_fp(sub.get("devices", {}), {})
        colors = _node_colors(sub.get("devices", {}))
        ports = [colors.get(port, ()) for port in sub.get("ports", [])]
        blob = json.dumps({"body": body, "ports": ports}, sort_keys=True, default=str, separators=(",", ":"))
        sub_fp[name] = blob
        sub_rows.append(blob)
    payload = {
        "directives": ir.get("directives", []),
        "subs": sorted(sub_rows),
        "top": _scope_fp(ir.get("devices", {}), sub_fp),
    }
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def load_banned() -> tuple[set[tuple[str, ...]], set[str], dict]:
    banned: set[tuple[str, ...]] = set()
    system_bits = [
        "Apply the requested edit and output the COMPLETE rewritten netlist.",
        "Use valid and conventional SPICE netlist syntax.",
        "Preserve every device, comment, and directive from the source unless the instruction modifies them.",
        "Must end with .end. Do NOT return only changed lines or only the new device.",
        "No code fences, no explanations.",
        "You read circuit netlists and answer structured-recognition questions.",
        "Return only a single JSON object. Do not include any prose, explanations, or markdown code fences in your response.",
    ]
    for text in system_bits:
        banned |= shingles(text)
    instructions = 0
    for path in CASES.rglob("case.json"):
        case = json.loads(path.read_text(encoding="utf-8"))
        for text in _instructions(case):
            banned |= shingles(text)
            instructions += 1
    prints = set()
    skipped = 0
    spice_files = 0
    for path in list((CASES / "v2_edit_raw").rglob("*.sp")) + list((CASES / "v2_recognition_raw").rglob("*.sp")):
        if path.name not in {"source.sp", "expected.sp", "spice.sp"}:
            continue
        spice_files += 1
        try:
            prints.add(fingerprint(path.read_text(encoding="utf-8", errors="replace")))
        except Exception:
            skipped += 1
    stats = {
        "instruction_texts": instructions,
        "spice_files": spice_files,
        "spice_fingerprints": len(prints),
        "spice_unparsed": skipped,
        "shingles": len(banned),
    }
    return banned, prints, stats


def _instructions(obj) -> list[str]:
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "instruction" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_instructions(value))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_instructions(value))
    return found


def accept(record: dict, banned: set, prints: set[str], seen: set[str]) -> str | None:
    if overlap_size(record["question"], banned):
        return "shingle"
    primary = []
    for text in record["_netlists"]:
        try:
            digest = fingerprint(text)
        except Exception as exc:
            return f"fingerprint {exc}"
        if digest in prints:
            return "benchmark fingerprint"
        primary.append(digest)
    key = tuple(sorted(primary))
    if key in seen:
        return "duplicate"
    seen.add(key)
    return None


def self_test() -> None:
    resistor = "R1 a b 1k\n.end\n"
    swapped = "R1 b a 1k\n.end\n"
    scaled = "R1 a b 1000\n.end\n"
    commented = "* note\nR1 a b 1k\n.end\n"
    voltage = "V1 a b 0.9\n.end\n"
    voltage_swap = "V1 b a 0.9\n.end\n"
    mos = "M1 d g s b nmos W=1u L=0.18u\n.end\n"
    mos_swap = "M1 s g d b nmos W=1u L=0.18u\n.end\n"
    if differs(resistor, swapped) or differs(resistor, scaled) or differs(resistor, commented):
        raise SystemExit("symmetric or scaled pairs should match")
    if same(voltage, voltage_swap) or same(mos, mos_swap):
        raise SystemExit("ordered terminals should differ")
    renamed = "R1 n1 n2 1k\nC1 n2 n3 1p\n.end\n"
    renamed_b = "R1 x y 1k\nC1 y z 1p\n.end\n"
    if fingerprint(renamed) != fingerprint(renamed_b):
        raise SystemExit("fingerprint is not rename-invariant")
    if fingerprint(resistor) != fingerprint(swapped):
        raise SystemExit("fingerprint split a symmetric resistor")
    if fingerprint(voltage) == fingerprint(mos):
        raise SystemExit("fingerprint collapsed different devices")
    rng = random.Random(1)
    for length in (4, 14):
        row = sequence_item(rng, length, False)
        if not differs(row["_netlists"][0], row["reference"]["text"]):
            raise SystemExit("flat sequence noop")
    row = sequence_item(rng, 4, True)
    if not differs(row["_netlists"][0], row["reference"]["text"]):
        raise SystemExit("subckt sequence noop")
    port_swap_item(rng, (0, 2))
    inline_item(rng)
    equivalence_item(rng, "unit_scale")
    equivalence_item(rng, "voltage_polarity")
    recognition_item(rng, "T2a", False)
    recognition_item(rng, "T3", True)
    recognition_item(rng, "T6", False)
    for text in BANNED_PROMPTS:
        if len(tokens(text)) < SHINGLE:
            raise SystemExit("prompt too short to shingle")


def measured_reports() -> dict:
    overall = json.loads((QWAN / "edit" / "overall_report.json").read_text(encoding="utf-8"))
    families = {row["family"]: row for row in overall["family_results"]}
    for name, fails in {**FLAT_FAMILIES, **SUB_FAMILIES, **OTHER_FAMILY_FAILS}.items():
        got = families[name]["failed_cases"]
        if got != fails:
            raise SystemExit(f"{name} failed_cases {got} != {fails}")
    if sum(FLAT_FAMILIES.values()) != LOCUS_FAILS["flat_compound"]:
        raise SystemExit("flat compound mass")
    if sum(SUB_FAMILIES.values()) != LOCUS_FAILS["subckt_compound"]:
        raise SystemExit("subckt compound mass")
    step_kinds = Counter()
    first_codes = Counter()
    for path in (QWAN / "edit").glob("*/report.json"):
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
    if dict(step_kinds) != STEP_FAILS:
        raise SystemExit(f"step kinds {dict(step_kinds)} != {STEP_FAILS}")
    recognition = {}
    for split in ("flat", "subckt"):
        report = json.loads((QWAN / "recognition" / split / "oracle_report.json").read_text(encoding="utf-8"))
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
    task_fail = {}
    for split in recognition.values():
        for row in split["per_task"]:
            task_fail[row["task"]] = row["n"] - row["pass"]
    expected_tasks = {"T6": 98, "T2a": 88, "T3": 80, "T5": 9, "T1": 0, "T2b": 0, "T4": 0}
    if task_fail != expected_tasks:
        raise SystemExit(f"recognition fails {task_fail}")
    edit_summary = overall["summary"]
    return {
        "model": "Qwen3.5-9B",
        "oracle": "NetlistBench official IR match and recognition grader",
        "edit": {
            "total": edit_summary["total_cases"],
            "passed": edit_summary["passed_cases"],
            "failed": edit_summary["failed_cases"],
            "pass_rate": edit_summary["pass_rate"],
            "bucket_summary": edit_summary["bucket_summary"],
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
        "not_used_for_synthesis": {
            "reason": "Pass rate is already near ceiling, so these families get no share of the 1000.",
            "parameter_edit_fails": 1,
            "device_remove_fails": 4,
            "rename_propagation_fails": 11,
            "T1_fails": 0,
            "T2b_fails": 0,
            "T4_fails": 0,
            "T5_fails": 9,
        },
    }


def build_schedule(rng: random.Random) -> list[tuple]:
    locus_n = largest_remainder(LOCUS_FAILS, N_TARGET)
    if sum(locus_n.values()) != N_TARGET:
        raise SystemExit(locus_n)
    flat_lengths = largest_remainder(FLAT_LENGTH_WEIGHTS, locus_n["flat_compound"])
    sub_lengths = largest_remainder(SUB_LENGTH_WEIGHTS, locus_n["subckt_compound"])
    if set(flat_lengths) & HELD_OUT_FLAT or set(sub_lengths) & HELD_OUT_SUB:
        raise SystemExit("held-out length allocated")
    jobs: list[tuple] = []
    jobs.extend(("sequence", length, False) for length in expand_counts(flat_lengths))
    jobs.extend(("sequence", length, True) for length in expand_counts(sub_lengths))
    port_n = largest_remainder({(0, 1): 1, (0, 2): 1, (1, 2): 1}, locus_n["port_swap"])
    jobs.extend(("port", pair) for pair in expand_counts(port_n))
    jobs.extend(("inline",) for _ in range(locus_n["inline_expand"]))
    connect = largest_remainder({"pin": 30, "vpin": 8, "ipin": 7}, locus_n["connectivity"])
    jobs.extend(("connect", mode) for mode in expand_counts(connect))
    adds = largest_remainder({"add_mos": 18, "add_v": 7, "add_i": 4}, locus_n["device_add"])
    jobs.extend(("add", mode) for mode in expand_counts(adds))
    replaces = largest_remainder(
        {"replace_rc": 8, "replace_cl": 6, "replace_rv": 5, "replace_mos": 8},
        locus_n["device_replace"],
    )
    jobs.extend(("replace", mode) for mode in expand_counts(replaces))
    equiv = largest_remainder(
        {
            "resistor_ends": 6,
            "unit_scale": 5,
            "comment_only": 5,
            "voltage_polarity": 6,
            "mos_drain_source": 5,
            "resistor_value": 5,
        },
        locus_n["equivalence"],
    )
    jobs.extend(("equiv", mode) for mode in expand_counts(equiv))
    jobs.extend(("recog", "T6", False) for _ in range(locus_n["neighbor_incidence"]))
    jobs.extend(("recog", "T2a", False) for _ in range(locus_n["semantic_pins"]))
    tied = locus_n["node_incidence"] // 2
    jobs.extend(("recog", "T3", True) for _ in range(tied))
    jobs.extend(("recog", "T3", False) for _ in range(locus_n["node_incidence"] - tied))
    if len(jobs) != N_TARGET:
        raise SystemExit(f"schedule {len(jobs)} locus {locus_n}")
    rng.shuffle(jobs)
    return jobs


def make_record(rng: random.Random, job: tuple) -> dict:
    kind = job[0]
    if kind == "sequence":
        return sequence_item(rng, job[1], job[2])
    if kind == "port":
        return port_swap_item(rng, job[1])
    if kind == "inline":
        return inline_item(rng)
    if kind == "connect":
        template = {"pin": "mos_pin", "vpin": "voltage_terminal", "ipin": "current_terminal"}[job[1]]
        return single_edit(rng, job[1], "connectivity", template)
    if kind == "add":
        template = {"add_mos": "add_mos", "add_v": "add_voltage", "add_i": "add_current"}[job[1]]
        return single_edit(rng, job[1], "device_add", template)
    if kind == "replace":
        return single_edit(rng, job[1], "device_replace", job[1])
    if kind == "equiv":
        return equivalence_item(rng, job[1])
    if kind == "recog":
        return recognition_item(rng, job[1], job[2])
    raise BuildError(job)


def main() -> int:
    self_test()
    score = measured_reports()
    banned, prints, decontam_stats = load_banned()
    for text in BANNED_PROMPTS:
        hit = shingles(text) & banned
        if hit:
            sample = " ".join(next(iter(hit)))
            raise SystemExit(f"prompt collides with a released instruction: {sample}")
    rng = random.Random(SEED)
    jobs = build_schedule(rng)
    seen: set[str] = set()
    kept = []
    overlap_hist = Counter()
    rejects = Counter()
    for index, job in enumerate(jobs, start=1):
        last = "none"
        for _ in range(40):
            try:
                record = make_record(rng, job)
            except BuildError as exc:
                last = str(exc)
                rejects["build"] += 1
                continue
            size = overlap_size(record["question"], banned)
            reason = accept(record, banned, prints, seen)
            if reason:
                last = reason
                rejects[reason] += 1
                continue
            overlap_hist[size] += 1
            kept.append(record)
            break
        else:
            raise SystemExit(f"failed job {job} after retries: {last}")
        if index % 100 == 0:
            print(f"kept {len(kept)}/{index}", flush=True)
    if len(kept) != N_TARGET:
        raise SystemExit(len(kept))
    for index, record in enumerate(kept, start=1):
        record.pop("_netlists", None)
        record["id"] = f"n{index:04d}"
        record["strategy"] = "failure_locus_verbalization"
        record["question_sha"] = hashlib.sha256(record["question"].encode()).hexdigest()[:12]
        reference = record["reference"]
        if reference["format"] == "spice":
            load_ir(reference["text"])
        elif reference["format"] == "label" and reference["text"] not in {"yes", "no"}:
            raise SystemExit(record["id"])
    by_locus = Counter(record["locus"] for record in kept)
    by_template = Counter(record["template"] for record in kept)
    by_group = Counter(record["group"] for record in kept)
    if dict(by_locus) != largest_remainder(LOCUS_FAILS, N_TARGET):
        # group names differ from locus keys only when several templates share a locus
        pass
    expected_loci = largest_remainder(LOCUS_FAILS, N_TARGET)
    if dict(by_locus) != expected_loci:
        raise SystemExit(f"locus counts {dict(by_locus)} != {expected_loci}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = OUT_DIR / "questions.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "strategy": "failure_locus_verbalization",
        "benchmark": "NetlistBench",
        "seed": SEED,
        "count": len(kept),
        "shingle_reject": SHINGLE,
        "held_out_flat_step_counts": sorted(HELD_OUT_FLAT),
        "held_out_subckt_step_counts": sorted(HELD_OUT_SUB),
        "allocation": "largest remainder over official Qwen3.5-9B fail counts",
        "by_locus": dict(by_locus),
        "by_template": dict(by_template),
        "by_group": dict(by_group),
        "overlap_hist": {str(key): overlap_hist[key] for key in sorted(overlap_hist)},
        "oracle_checked": len(kept),
        "decontamination": decontam_stats,
        "rejects_during_sampling": dict(rejects),
        "score_path": str(SCORE_PATH),
        "path": str(jsonl),
        "note": (
            "Reference answers are held out of the question. "
            "The 2342 released cases are excluded by instruction shingles and netlist fingerprints. "
            "This pool is not mixed into PICBench."
        ),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    score["synthesis"] = {
        "count": N_TARGET,
        "by_locus": dict(by_locus),
        "held_out_flat_step_counts": sorted(HELD_OUT_FLAT),
        "held_out_subckt_step_counts": sorted(HELD_OUT_SUB),
        "path": str(jsonl),
    }
    SCORE_PATH.write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest, "edit_pass": score["edit"]["pass_rate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
