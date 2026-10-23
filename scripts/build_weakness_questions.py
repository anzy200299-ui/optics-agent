#!/usr/bin/env python3
"""Build 1000 PICBench-style questions aimed at Qwen3.5-9B's measured failures.

Strategy: failure-locus verbalization.

The zero-shot 9B almost never reaches a SAX spectrum. On 120 samples the
mass is ellipsis (34), // comments (28), prose after the JSON (15), markdown
fences (13), the wrong SAX model id (11), a missing <result> (7), other
syntax (5), and bad ports (3). Four samples simulate; one matches.

Each question is the prose form of a netlist this script constructs. The
netlist is checked with SAX before it is kept. The question does not contain
that JSON, and it does not add format rules that Table III omits. What changes
is the circuit:

- ellipsis: a wide bundle of identical blocks, so a correct JSON is long and
  repetitive. Width is 2, 3, 5, or 6. Widths 4 and 8 are held out with the
  test meshes.
- comment: a chain in which every phase shifter has its own length and phase,
  so the numbers have to live in settings.
- wrong model: matched groups that share the wiring and differ only by a SAX
  id the 9B collapsed (mzi / mzm / mzm_dual, straight / straight_heat_metal,
  coupler / mzi_2x2 / mzi_ps / OSU).
- wrong ports: the external port map is part of the spec, including every
  microring port.
- functional: one-axis moves away from the test items that reached SAX
  (direct modulator, MZM, NLS). Test parameter signatures are excluded.
- graft: two of those fragments wired in an order the 24 test items do not use.

An 8-word shingle shared with any of the 24 problem texts, or with the
in-context example, rejects the item.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

from repo_paths import REPO

ROOT = REPO
PIC = ROOT / "third_party" / "PICBench" / "PICBench"
CASES = ROOT / "third_party" / "PICBench" / "testcases"
OUT_DIR = ROOT / "data" / "weakness_questions"
SEED = 20261020
N_TARGET = 1000
SHINGLE = 8

# SAX model -> (inputs, outputs). Composites come from devices.py.
PORTS = {
    "mmi1x2": (["I1"], ["O1", "O2"]),
    "coupler": (["I1", "I2"], ["O1", "O2"]),
    "mzi": (["I1"], ["O1"]),
    "mzi_2x2": (["I1", "I2"], ["O1", "O2"]),
    "mzi_ps": (["I1", "I2"], ["O1", "O2"]),
    "mzm": (["I1"], ["O1"]),
    "mzm_dual": (["I1"], ["O1"]),
    "OSU": (["I1", "I2"], ["O1", "O2"]),
    "straight": (["I1"], ["O1"]),
    "straight_heat_metal": (["I1"], ["O1"]),
    "mrr": (["I1"], ["O1", "O2", "O3"]),
}

HELD_OUT_WIDTHS = {4, 8}
WDM_CWLS = (1.543, 1.550, 1.554, 1.556)
RING_CWLS = (1.543, 1.546, 1.547, 1.55, 1.552, 1.554, 1.556)

EXAMPLE_Q = (
    "Create a Mach-Zehnder interferometer (MZI) with a single input and output, "
    "featuring a path length difference of L. Use the built-in multimode interferometer (MMI) component. \n"
    "Parameters:\nL = 10 microns \n\n"
    "Design the circuit and generate the corresponding JSON netlist based on the description. "
    "Try to understand the requirements and give reasoning steps in natural language to achieve it."
)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(text: str, n: int = SHINGLE) -> set[tuple[str, ...]]:
    words = tokens(text)
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def load_forbidden() -> set[tuple[str, ...]]:
    banned: set[tuple[str, ...]] = set()
    banned |= shingles(EXAMPLE_Q)
    for path in CASES.glob("*/*_problem_description.txt"):
        banned |= shingles(path.read_text(encoding="utf-8"))
    return banned


def component_of(value) -> str:
    if isinstance(value, str):
        return value
    return value["component"]


def check_bindings(doc: dict) -> None:
    net = doc["netlist"]
    instances = net["instances"]
    models = doc["models"]
    available: dict[str, set[str]] = {}
    for name, value in instances.items():
        sax_model = models[component_of(value)]
        inputs, outputs = PORTS[sax_model]
        available[name] = set(inputs + outputs)
        if sax_model in {"mzi", "mzm", "mzm_dual", "mzi_2x2", "mzi_ps", "OSU", "mmi1x2", "coupler"}:
            if isinstance(value, dict) and value.get("settings"):
                raise ValueError(f"composite {name} should keep default settings")
    used: dict[str, set[str]] = {name: set() for name in instances}
    for left, right in net["connections"].items():
        for end in (left, right):
            inst, port = end.split(",")
            if port not in available[inst]:
                raise ValueError(f"unknown port {end}")
            if port in used[inst]:
                raise ValueError(f"port used twice: {end}")
            used[inst].add(port)
    exported = []
    for external, binding in net["ports"].items():
        if not external.startswith(("I", "O")):
            raise ValueError(f"external port {external}")
        inst, port = binding.split(",")
        if port not in available[inst]:
            raise ValueError(f"bad binding {binding}")
        if port in used[inst]:
            raise ValueError(f"exported port already connected: {binding}")
        used[inst].add(port)
        exported.append(external)
    if any(name[1:] == "4" or name[1:] == "8" for name in exported if name[0] in "IO" and name[1:].isdigit()):
        # Width 4 or 8 is the held-out mesh size. A lone I4 inside a larger
        # legal width cannot happen because widths are 2, 3, 5, 6.
        pass
    width_in = sum(1 for name in exported if name.startswith("I") and name[1:].isdigit())
    if width_in in HELD_OUT_WIDTHS:
        raise ValueError(f"held-out width {width_in}")


def sax_ok(doc: dict) -> None:
    import jax.numpy as jnp

    models = {key: SAX_MODELS[value] for key, value in doc["models"].items()}
    circuit, _ = sax.circuit(netlist=doc["netlist"], models=models)
    wl = jnp.linspace(1.51, 1.59, 8)
    result = circuit(wl=wl)
    needed = [name for name in doc["netlist"]["ports"] if name.startswith("I")]
    outs = [name for name in doc["netlist"]["ports"] if name.startswith("O")]
    for src in needed:
        for dst in outs:
            value = result[src, dst]
            if not jnp.all(jnp.isfinite(value)):
                raise ValueError(f"non-finite S[{src},{dst}]")


def item(locus: str, template: str, group: str, question: str, doc: dict, spec: dict) -> dict:
    check_bindings(doc)
    return {
        "locus": locus,
        "template": template,
        "group": group,
        "question": question.strip() + "\n",
        "spec": spec,
        "reference": doc,
    }


def parallel_bundle(width: int, depth: int, block: str, phase0: int, length: int, group: str) -> dict:
    if width in HELD_OUT_WIDTHS:
        raise ValueError(width)
    instances = {}
    connections = {}
    ports = {}
    for mode in range(1, width + 1):
        names = [f"{block}_{mode}_{stage}" for stage in range(1, depth + 1)]
        for name in names:
            instances[name] = block
        tail = f"ps_{mode}"
        instances[tail] = {
            "component": "ps",
            "settings": {"phase_shift": phase0 + 3 * mode, "length": length},
        }
        for left, right in zip(names, names[1:]):
            connections[f"{left},O1"] = f"{right},I1"
        connections[f"{names[-1]},O1"] = f"{tail},I1"
        ports[f"I{mode}"] = f"{names[0]},I1"
        ports[f"O{mode}"] = f"{tail},O1"
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {block: block, "ps": "straight_heat_metal"},
    }
    question = f"""Problem Description-bundle-{block}-w{width}-d{depth}:

Build {width} parallel chains, one chain for each spatial mode. The modes are independent: nothing crosses between them.
Mode m, with m running from 1 to {width}, contains the blocks {block}_m_1, {block}_m_2, ... , {block}_m_{depth} followed by a phase shifter ps_m.
Every {block}_m_stage instance is the built-in SAX model {block} at its default settings. The models entry for that component key is {block}.
ps_m is the built-in SAX model straight_heat_metal. Its length is {length} microns and its phase_shift is {phase0 + 3}*m degrees, so mode 1 is {phase0 + 3} degrees, mode 2 is {phase0 + 6} degrees, and so on through mode {width}.
Inside mode m, connect O1 of {block}_m_stage to I1 of {block}_m_(stage+1) for every stage from 1 to {depth - 1}, then connect O1 of {block}_m_{depth} to I1 of ps_m.
Circuit port Im is bound to {block}_m_1,I1. Circuit port Om is bound to ps_m,O1.
"""
    # The phase sentence above is ambiguous if read as a formula. State the list.
    phases = [phase0 + 3 * mode for mode in range(1, width + 1)]
    phase_line = ", ".join(f"ps_{mode} phase_shift {phases[mode - 1]}" for mode in range(1, width + 1))
    question = f"""Problem Description-bundle-{block}-w{width}-d{depth}-{phase0}:

Build {width} parallel optical chains with no cross links between modes.
Mode m (m = 1..{width}) is the series {block}_m_1 .. {block}_m_{depth}, then ps_m.
Each {block}_m_stage uses SAX model {block} with default settings. Map the component key {block} to {block}.
Each ps_m uses SAX model straight_heat_metal with length {length} microns. The heater phases, in degrees, are: {phase_line}.
On mode m, connect O1 of each {block} block to I1 of the next block, and connect O1 of {block}_m_{depth} to I1 of ps_m.
Bind circuit port Im to {block}_m_1,I1 and circuit port Om to ps_m,O1.
"""
    return item(
        "ellipsis",
        "parallel_bundle",
        group,
        question,
        doc,
        {"width": width, "depth": depth, "block": block, "phase0": phase0, "length": length, "phases": phases},
    )


def phase_chain(phases: list[int], lengths: list[int], group: str) -> dict:
    if 180 in phases and len(phases) == 1:
        raise ValueError("bare 180 degree phase")
    instances = {}
    connections = {}
    for index, (phase, length) in enumerate(zip(phases, lengths), start=1):
        instances[f"ps_{index}"] = {
            "component": "ps",
            "settings": {"phase_shift": phase, "length": length},
        }
    for index in range(1, len(phases)):
        connections[f"ps_{index},O1"] = f"ps_{index + 1},I1"
    ports = {"I1": "ps_1,I1", "O1": f"ps_{len(phases)},O1"}
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {"ps": "straight_heat_metal"},
    }
    details = "; ".join(
        f"ps_{i} length {length} microns, phase_shift {phase} degrees"
        for i, (phase, length) in enumerate(zip(phases, lengths), start=1)
    )
    question = f"""Problem Description-heater-chain-{len(phases)}:

Build one series chain of {len(phases)} heaters. There is a single circuit input and a single circuit output.
The instances are ps_1 through ps_{len(phases)}. Each one is SAX model straight_heat_metal, and the models entry maps ps to straight_heat_metal.
Settings, which differ on every instance: {details}.
Connect O1 of ps_i to I1 of ps_{{i+1}} for each i from 1 to {len(phases) - 1}.
Bind circuit port I1 to ps_1,I1 and circuit port O1 to ps_{len(phases)},O1.
"""
    return item(
        "comment",
        "heater_chain",
        group,
        question,
        doc,
        {"phases": phases, "lengths": lengths},
    )


def sandwiched(middle: str, lead: int, phase: int, group: str) -> dict:
    instances = {
        "lead": {"component": "wg", "settings": {"length": lead}},
        "core": middle,
        "tail": {"component": "ps", "settings": {"phase_shift": phase, "length": 16}},
    }
    connections = {"lead,O1": "core,I1", "core,O1": "tail,I1"}
    ports = {"I1": "lead,I1", "O1": "tail,O1"}
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {"wg": "straight", middle: middle, "ps": "straight_heat_metal"},
    }
    role = {
        "mzi": "a one-input one-output interferometer",
        "mzm": "a single-drive modulator",
        "mzm_dual": "a dual-drive modulator",
    }[middle]
    question = f"""Problem Description-core-{middle}-{lead}-{phase}:

Build a one-input, one-output path of three instances: lead, core, and tail, in that order.
lead is SAX model straight with length {lead} microns. Map wg to straight.
core is {role}. Its SAX model is {middle}, at default settings. Map {middle} to {middle}.
tail is SAX model straight_heat_metal with length 16 microns and phase_shift {phase} degrees. Map ps to straight_heat_metal.
Connect lead,O1 to core,I1 and core,O1 to tail,I1.
Bind circuit port I1 to lead,I1 and circuit port O1 to tail,O1.
"""
    return item(
        "wrong_model",
        "one_port_core",
        group,
        question,
        doc,
        {"middle": middle, "lead": lead, "phase": phase},
    )


def two_port(model: str, len1: int, len2: int, group: str) -> dict:
    instances = {
        "in1": {"component": "wg", "settings": {"length": len1}},
        "in2": {"component": "wg", "settings": {"length": len2}},
        "core": model,
    }
    connections = {"in1,O1": "core,I1", "in2,O1": "core,I2"}
    ports = {"I1": "in1,I1", "I2": "in2,I1", "O1": "core,O1", "O2": "core,O2"}
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {"wg": "straight", model: model},
    }
    role = {
        "coupler": "a directional coupler",
        "mzi_2x2": "a two-input two-output interferometer",
        "mzi_ps": "a two-input interferometer with heaters",
        "OSU": "a 2x2 optical switch unit",
    }[model]
    question = f"""Problem Description-twoport-{model}-{len1}-{len2}:

Build a two-input, two-output element fed by two delay lines.
in1 and in2 are SAX model straight, with lengths {len1} and {len2} microns. Map wg to straight.
core is {role}. Its SAX model is {model}, at default settings. Map {model} to {model}.
Connect in1,O1 to core,I1 and in2,O1 to core,I2.
Bind I1 to in1,I1, I2 to in2,I1, O1 to core,O1, and O2 to core,O2.
"""
    return item(
        "wrong_model",
        "two_port_core",
        group,
        question,
        doc,
        {"model": model, "len1": len1, "len2": len2},
    )


def wire(model: str, length: int, phase: int, group: str) -> dict:
    settings = {"length": length}
    if model == "straight_heat_metal":
        settings["phase_shift"] = phase
    instances = {"dev": {"component": "dev", "settings": settings}}
    ports = {"I1": "dev,I1", "O1": "dev,O1"}
    doc = {
        "netlist": {"instances": instances, "connections": {}, "ports": ports},
        "models": {"dev": model},
    }
    if model == "straight":
        detail = f"a passive delay, SAX model straight, length {length} microns, with no heater phase"
    else:
        detail = (
            f"a heater, SAX model straight_heat_metal, length {length} microns "
            f"and phase_shift {phase} degrees"
        )
    question = f"""Problem Description-wire-{model}-{length}-{phase}:

Build a single device named dev. It is {detail}. Map dev to {model}.
There are no internal connections.
Bind circuit port I1 to dev,I1 and circuit port O1 to dev,O1.
"""
    return item(
        "wrong_model",
        "single_wire",
        group,
        question,
        doc,
        {"model": model, "length": length, "phase": phase},
    )


def ring_ports(cwl: float, group: str) -> dict:
    if (cwl, cwl, cwl, cwl) == WDM_CWLS:
        raise ValueError("wdm signature")
    instances = {"ring": {"component": "ring", "settings": {"cwl": cwl}}}
    ports = {"I1": "ring,I1", "O1": "ring,O1", "O2": "ring,O2", "O3": "ring,O3"}
    doc = {
        "netlist": {"instances": instances, "connections": {}, "ports": ports},
        "models": {"ring": "mrr"},
    }
    question = f"""Problem Description-ring-ports-{cwl}:

Place one add-drop microring named ring. Its SAX model is mrr and its resonance cwl is {cwl} micron. Map ring to mrr.
Export every physical port of the ring. The bindings are I1 = ring,I1, O1 = ring,O1, O2 = ring,O2, O3 = ring,O3.
Do not drop O3, and do not rename the bindings to words such as through or drop.
"""
    return item("wrong_ports", "ring_ports", group, question, doc, {"cwl": cwl})


def split_arms(len1: int, len2: int, group: str) -> dict:
    instances = {
        "split": "mmi",
        "arm1": {"component": "wg", "settings": {"length": len1}},
        "arm2": {"component": "wg", "settings": {"length": len2}},
    }
    connections = {"split,O1": "arm1,I1", "split,O2": "arm2,I1"}
    ports = {"I1": "split,I1", "O1": "arm1,O1", "O2": "arm2,O1"}
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {"mmi": "mmi1x2", "wg": "straight"},
    }
    question = f"""Problem Description-split-arms-{len1}-{len2}:

Build a one-input, two-output splitter followed by two delays. Do not recombine the arms.
split is SAX model mmi1x2 at default settings. Map mmi to mmi1x2.
arm1 and arm2 are SAX model straight, with lengths {len1} and {len2} microns. Map wg to straight.
Connect split,O1 to arm1,I1 and split,O2 to arm2,I1.
Bind I1 to split,I1, O1 to arm1,O1, and O2 to arm2,O1.
"""
    return item(
        "wrong_ports",
        "split_arms",
        group,
        question,
        doc,
        {"len1": len1, "len2": len2},
    )


def asymmetric_mzi(len1: int, len2: int, phase: int, group: str) -> dict:
    if {len1, len2} == {10, 20} or (len1 == 10 and len2 == 10):
        raise ValueError("test-like lengths")
    instances = {
        "split": "mmi",
        "arm1": {"component": "wg", "settings": {"length": len1}},
        "arm2": {"component": "wg", "settings": {"length": len2}},
        "heat": {"component": "ps", "settings": {"phase_shift": phase, "length": 15}},
        "join": "cpl",
    }
    connections = {
        "split,O1": "arm1,I1",
        "arm1,O1": "heat,I1",
        "heat,O1": "join,I1",
        "split,O2": "arm2,I1",
        "arm2,O1": "join,I2",
    }
    ports = {"I1": "split,I1", "O1": "join,O1", "O2": "join,O2"}
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {"mmi": "mmi1x2", "wg": "straight", "ps": "straight_heat_metal", "cpl": "coupler"},
    }
    question = f"""Problem Description-asymmetric-interferometer-{len1}-{len2}-{phase}:

Build an unbalanced interferometer with one circuit input and two circuit outputs.
split is SAX model mmi1x2. join is SAX model coupler. Both stay at default settings.
Map mmi to mmi1x2 and cpl to coupler.
The upper arm is arm1, a straight of length {len1} microns, followed by heat, a straight_heat_metal of length 15 microns and phase_shift {phase} degrees.
The lower arm is arm2, a straight of length {len2} microns, with no heater.
Map wg to straight and ps to straight_heat_metal.
Connections: split,O1 to arm1,I1; arm1,O1 to heat,I1; heat,O1 to join,I1; split,O2 to arm2,I1; arm2,O1 to join,I2.
Bind I1 to split,I1, O1 to join,O1, and O2 to join,O2.
"""
    return item(
        "functional",
        "asymmetric_mzi",
        group,
        question,
        doc,
        {"len1": len1, "len2": len2, "phase": phase},
    )


def modulator_line(kind: str, count: int, phase: int, group: str) -> dict:
    if count < 2:
        raise ValueError("count collides with the single-block test modulator")
    instances = {}
    connections = {}
    for index in range(1, count + 1):
        instances[f"mod_{index}"] = kind
    instances["heat"] = {"component": "ps", "settings": {"phase_shift": phase, "length": 18}}
    for index in range(1, count):
        connections[f"mod_{index},O1"] = f"mod_{index + 1},I1"
    connections[f"mod_{count},O1"] = "heat,I1"
    ports = {"I1": "mod_1,I1", "O1": "heat,O1"}
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {kind: kind, "ps": "straight_heat_metal"},
    }
    question = f"""Problem Description-modulator-line-{kind}-{count}-{phase}:

Build a series of {count} modulators, then one heater. The circuit has one input and one output.
mod_1 through mod_{count} each use SAX model {kind} at default settings. Map {kind} to {kind}.
heat uses SAX model straight_heat_metal with length 18 microns and phase_shift {phase} degrees. Map ps to straight_heat_metal.
Connect O1 of mod_i to I1 of mod_{{i+1}} for i from 1 to {count - 1}, and connect O1 of mod_{count} to I1 of heat.
Bind I1 to mod_1,I1 and O1 to heat,O1.
"""
    return item(
        "functional",
        "modulator_line",
        group,
        question,
        doc,
        {"kind": kind, "count": count, "phase": phase},
    )


def coupler_phase_line(phase: int, group: str) -> dict:
    if phase == 180:
        raise ValueError("test NLS phase")
    instances = {
        "c1": "coupler",
        "heat": {"component": "ps", "settings": {"phase_shift": phase, "length": 14}},
        "c2": "coupler",
        "c3": "coupler",
    }
    connections = {
        "c1,O1": "heat,I1",
        "heat,O1": "c2,I1",
        "c1,O2": "c2,I2",
        "c2,O1": "c3,I1",
        "c2,O2": "c3,I2",
    }
    ports = {"I1": "c1,I1", "I2": "c1,I2", "O1": "c3,O1", "O2": "c3,O2"}
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {"coupler": "coupler", "ps": "straight_heat_metal"},
    }
    question = f"""Problem Description-coupler-phase-line-{phase}:

Build a two-mode line of three couplers with one heater on the first upper path.
c1, c2, and c3 are SAX model coupler at default settings. Map coupler to coupler.
heat is SAX model straight_heat_metal, length 14 microns, phase_shift {phase} degrees. Map ps to straight_heat_metal.
Connections: c1,O1 to heat,I1; heat,O1 to c2,I1; c1,O2 to c2,I2; c2,O1 to c3,I1; c2,O2 to c3,I2.
Bind I1 to c1,I1, I2 to c1,I2, O1 to c3,O1, and O2 to c3,O2.
"""
    return item(
        "functional",
        "coupler_phase_line",
        group,
        question,
        doc,
        {"phase": phase},
    )


def mzi_into_ring(len1: int, len2: int, phase: int, cwl: float, group: str) -> dict:
    instances = {
        "split": "mmi",
        "arm1": {"component": "wg", "settings": {"length": len1}},
        "arm2": {"component": "wg", "settings": {"length": len2}},
        "heat": {"component": "ps", "settings": {"phase_shift": phase, "length": 15}},
        "join": "cpl",
        "ring": {"component": "ring", "settings": {"cwl": cwl}},
    }
    connections = {
        "split,O1": "arm1,I1",
        "arm1,O1": "heat,I1",
        "heat,O1": "join,I1",
        "split,O2": "arm2,I1",
        "arm2,O1": "join,I2",
        "join,O1": "ring,I1",
    }
    ports = {
        "I1": "split,I1",
        "O1": "ring,O1",
        "O2": "ring,O2",
        "O3": "ring,O3",
        "O4": "join,O2",
    }
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {
            "mmi": "mmi1x2",
            "wg": "straight",
            "ps": "straight_heat_metal",
            "cpl": "coupler",
            "ring": "mrr",
        },
    }
    question = f"""Problem Description-interferometer-into-ring-{len1}-{len2}-{phase}-{cwl}:

Build an unbalanced interferometer and send only its first output into a microring.
split is mmi1x2. join is coupler. ring is mrr with cwl {cwl}. arm1 and arm2 are straights of lengths {len1} and {len2} microns.
heat is straight_heat_metal, length 15 microns, phase_shift {phase} degrees, placed after arm1.
Map mmi to mmi1x2, cpl to coupler, wg to straight, ps to straight_heat_metal, and ring to mrr.
Connections: split,O1 to arm1,I1; arm1,O1 to heat,I1; heat,O1 to join,I1; split,O2 to arm2,I1; arm2,O1 to join,I2; join,O1 to ring,I1.
Bind I1 to split,I1, O1 to ring,O1, O2 to ring,O2, O3 to ring,O3, and O4 to join,O2.
"""
    return item(
        "graft",
        "mzi_into_ring",
        group,
        question,
        doc,
        {"len1": len1, "len2": len2, "phase": phase, "cwl": cwl},
    )


def switch_into_modulator(length: int, group: str) -> dict:
    instances = {
        "sw": "OSU",
        "mod": "mzm",
        "side": {"component": "wg", "settings": {"length": length}},
    }
    connections = {"sw,O1": "mod,I1", "sw,O2": "side,I1"}
    ports = {"I1": "sw,I1", "I2": "sw,I2", "O1": "mod,O1", "O2": "side,O1"}
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {"OSU": "OSU", "mzm": "mzm", "wg": "straight"},
    }
    question = f"""Problem Description-switch-then-modulator-{length}:

Build a 2x2 switch, modulate only the first output, and delay the second output.
sw is SAX model OSU at default settings. mod is SAX model mzm at default settings.
side is SAX model straight with length {length} microns.
Map OSU to OSU, mzm to mzm, and wg to straight.
Connect sw,O1 to mod,I1 and sw,O2 to side,I1.
Bind I1 to sw,I1, I2 to sw,I2, O1 to mod,O1, and O2 to side,O1.
"""
    return item(
        "graft",
        "switch_into_modulator",
        group,
        question,
        doc,
        {"length": length},
    )


def longest_overlap(question: str, banned: set[tuple[str, ...]]) -> int:
    words = tokens(question)
    for size in range(12, 4, -1):
        if len(words) < size:
            continue
        for i in range(len(words) - size + 1):
            if tuple(words[i : i + size]) in banned and size >= SHINGLE:
                return size
            if size < SHINGLE:
                break
        if size < SHINGLE:
            break
    # recount properly for reporting
    best = 0
    for size in range(min(12, len(words)), 4, -1):
        gram = {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}
        # banned only stores 8-grams; compare equal size by rebuilding when needed
        if size == SHINGLE and gram & banned:
            best = max(best, size)
            break
    return best


def overlap_size(question: str, corpus_shingles: dict[int, set[tuple[str, ...]]]) -> int:
    words = tokens(question)
    best = 0
    for size, banned in corpus_shingles.items():
        if len(words) < size:
            continue
        for i in range(len(words) - size + 1):
            if tuple(words[i : i + size]) in banned:
                best = max(best, size)
                break
    return best


def build_corpus() -> dict[int, set[tuple[str, ...]]]:
    texts = [EXAMPLE_Q]
    texts.extend(path.read_text(encoding="utf-8") for path in CASES.glob("*/*_problem_description.txt"))
    corpus = {size: set() for size in range(5, 13)}
    for text in texts:
        words = tokens(text)
        for size in corpus:
            if len(words) >= size:
                corpus[size].update(tuple(words[i : i + size]) for i in range(len(words) - size + 1))
    return corpus


def forbidden_text(question: str) -> str | None:
    lowered = question.lower()
    for needle in ("64-qam", "8-qam", "64 qam", "8 qam", "n=4", "n=8", "parameters:n=4", "parameters:n=8"):
        if needle in lowered:
            return needle
    if "1.543" in question and "1.554" in question and "1.556" in question:
        return "wdm wavelength set"
    return None


def generate(rng: random.Random) -> list[dict]:
    made: list[dict] = []
    seen: set[str] = set()

    def add(record: dict) -> None:
        key = json.dumps(record["spec"], sort_keys=True) + "|" + record["template"] + "|" + record.get("group", "")
        # group is shared inside a contrast set, so include the distinguishing model too
        key = json.dumps({"template": record["template"], "spec": record["spec"]}, sort_keys=True)
        if key in seen:
            raise ValueError(f"duplicate spec {key}")
        seen.add(key)
        made.append(record)

    # Ellipsis: 280 bundles. Width held out of {{4, 8}}.
    widths = [2, 3, 5, 6]
    depths = [4, 5, 6, 7, 9]
    blocks = ["mzi", "mzm", "mzm_dual"]
    phase0s = [3, 7, 11, 13, 17]
    lengths = [12, 21, 37]
    bundle_keys = [
        (w, d, b, p, L)
        for w in widths
        for d in depths
        for b in blocks
        for p in phase0s
        for L in lengths
    ]
    rng.shuffle(bundle_keys)
    for width, depth, block, phase0, length in bundle_keys[:280]:
        add(parallel_bundle(width, depth, block, phase0, length, group=f"bundle-{block}"))

    # Comment: 160 heater chains with distinct numbers.
    phase_pool = [12, 20, 25, 33, 48, 57, 66, 78, 85, 95, 110, 125, 140, 155, 165, 200, 230, 260, 300]
    length_pool = [11, 13, 14, 16, 18, 19, 23, 26, 28, 32, 34, 38, 41, 44, 47, 55, 62, 71]
    heat_seen: set[str] = set()
    while len(heat_seen) < 160:
        k = rng.choice([4, 5, 6, 7])
        phases = rng.sample(phase_pool, k)
        lens = rng.sample(length_pool, k)
        key = json.dumps([phases, lens])
        if key in heat_seen:
            continue
        heat_seen.add(key)
        add(phase_chain(phases, lens, group=f"heat-{len(heat_seen)}"))

    # Wrong model: sandwiches, two-port quads, and straight/heater pairs.
    leads = [13, 17, 22, 29, 36, 43]
    tail_phases = [15, 28, 42, 63, 81, 96, 124]
    sandwich_keys = [(m, lead, phase) for m in ("mzi", "mzm", "mzm_dual") for lead in leads for phase in tail_phases]
    rng.shuffle(sandwich_keys)
    for middle, lead, phase in sandwich_keys[:120]:
        add(sandwiched(middle, lead, phase, group=f"sand-{lead}-{phase}"))

    two_keys = [
        (model, a, b)
        for model in ("coupler", "mzi_2x2", "mzi_ps", "OSU")
        for a in (12, 19, 27, 33, 46)
        for b in (15, 24, 31, 52)
    ]
    rng.shuffle(two_keys)
    for model, len1, len2 in two_keys[:80]:
        add(two_port(model, len1, len2, group=f"two-{len1}-{len2}"))

    wire_keys = [
        (model, length, phase)
        for model in ("straight", "straight_heat_metal")
        for length in (12, 16, 22, 28, 35, 48, 60, 74)
        for phase in (18, 45, 72, 110, 150)
    ]
    rng.shuffle(wire_keys)
    for model, length, phase in wire_keys[:80]:
        add(wire(model, length, phase, group=f"wire-{length}-{phase}"))

    # Ports: 7 rings + 73 splitters.
    for cwl in RING_CWLS:
        add(ring_ports(cwl, group=f"ring-{cwl}"))
    split_keys = [
        (a, b)
        for a in (12, 15, 18, 21, 24, 27, 30, 36, 42, 48)
        for b in (14, 17, 20, 23, 26, 33, 39, 45, 51, 57)
    ]
    rng.shuffle(split_keys)
    for len1, len2 in split_keys[:73]:
        add(split_arms(len1, len2, group=f"split-{len1}-{len2}"))

    # Functional shifts away from the test items.
    arm_pool = [12, 14, 16, 18, 22, 26, 28, 32, 36, 41, 45, 52, 58, 66, 74]
    phase_pool_f = [15, 28, 42, 63, 75, 96, 110, 124, 140, 165, 210, 240]
    asym_keys = [(a, b, p) for a in arm_pool for b in arm_pool if a != b for p in phase_pool_f]
    rng.shuffle(asym_keys)
    for len1, len2, phase in asym_keys[:60]:
        add(asymmetric_mzi(len1, len2, phase, group=f"asym-{len1}-{len2}-{phase}"))
    for kind in ("mzm", "mzm_dual"):
        for count in (2, 3, 5, 6):
            for phase in (24, 48, 72, 96, 132):
                add(modulator_line(kind, count, phase, group=f"line-{kind}-{count}-{phase}"))
    for phase in (20, 30, 35, 45, 50, 55, 60, 75, 80, 90, 100, 105, 120, 135, 150, 160, 210, 240, 270, 300):
        add(coupler_phase_line(phase, group=f"cphase-{phase}"))

    # Grafts.
    graft_keys = [
        (a, b, p, cwl)
        for a, b in ((12, 28), (16, 41), (22, 36), (18, 52), (26, 14))
        for p in (24, 63, 110, 165)
        for cwl in (1.546, 1.547, 1.552, 1.55)
    ]
    rng.shuffle(graft_keys)
    for len1, len2, phase, cwl in graft_keys[:40]:
        add(mzi_into_ring(len1, len2, phase, cwl, group=f"graft-ring-{len1}-{len2}-{phase}-{cwl}"))
    for length in (13, 17, 21, 25, 29, 33, 37, 43, 49, 55, 61, 67, 73, 15, 19, 23, 27, 31, 35, 39,
                   41, 45, 47, 51, 53, 57, 59, 63, 65, 69, 71, 75, 77, 79, 81, 83, 85, 87, 89, 91):
        add(switch_into_modulator(length, group=f"graft-sw-{length}"))

    return made


def main() -> None:
    global SAX_MODELS, sax
    os.chdir(PIC)
    sys.path.insert(0, str(PIC))
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

    sax = sax_mod
    SAX_MODELS = {
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
    rng = random.Random(SEED)
    corpus = build_corpus()
    records = generate(rng)
    if len(records) != N_TARGET:
        raise SystemExit(f"expected {N_TARGET} questions, built {len(records)}")

    rejected = []
    kept = []
    max_overlap = 0
    overlap_hist: dict[int, int] = {}
    for record in records:
        reason = forbidden_text(record["question"])
        size = overlap_size(record["question"], corpus)
        max_overlap = max(max_overlap, size)
        overlap_hist[size] = overlap_hist.get(size, 0) + 1
        if reason or size >= SHINGLE:
            rejected.append({"reason": reason or f"{size}-gram", "question": record["question"][:240]})
            continue
        sax_ok(record["reference"])
        kept.append(record)

    if rejected:
        print(json.dumps(rejected[:8], indent=2))
        raise SystemExit(f"rejected {len(rejected)}")

    for index, record in enumerate(kept, start=1):
        record["id"] = f"q{index:04d}"
        record["strategy"] = "failure_locus_verbalization"
        digest = hashlib.sha256(record["question"].encode()).hexdigest()[:12]
        record["question_sha"] = digest

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = OUT_DIR / "questions.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_locus: dict[str, int] = {}
    by_template: dict[str, int] = {}
    for record in kept:
        by_locus[record["locus"]] = by_locus.get(record["locus"], 0) + 1
        by_template[record["template"]] = by_template.get(record["template"], 0) + 1
    manifest = {
        "strategy": "failure_locus_verbalization",
        "seed": SEED,
        "count": len(kept),
        "shingle_reject": SHINGLE,
        "max_word_overlap_with_picbench_24": max_overlap,
        "overlap_hist": {str(k): overlap_hist[k] for k in sorted(overlap_hist)},
        "held_out_widths": sorted(HELD_OUT_WIDTHS),
        "by_locus": by_locus,
        "by_template": by_template,
        "sax_checked": len(kept),
        "path": str(jsonl),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
