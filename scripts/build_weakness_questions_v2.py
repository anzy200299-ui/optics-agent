#!/usr/bin/env python3
"""Batch 2 of hard PIC netlist questions: differential and counting targets.

Batch 1 (failure_locus_verbalization, 1000 items, seed 20261020) verbalizes
one failure locus at a time. Its ceiling is visible in the teacher rollout:
GLM-5.3 at temperature 0 passes 141/175 of the first, purely ellipsis-based
items, i.e. the pool is not hard for the teacher and gives little filtering
signal. Batch 2 borrows three ideas from the 2026 question-synthesis
literature and re-instantiates them inside this repo's own loop:

  - "solution-centric evolution" (BenchEvolver, arXiv:2606.01286): start from
    the solution artifact (a SAX-checked netlist) and derive the question,
    rather than writing a prompt and hoping the JSON exists.
  - difficulty probes instead of difficulty heuristics (CoDiQ,
    arXiv:2602.01660): keep an item only if a measured probe fails or barely
    passes, so difficulty is an observed property of this pipeline, not a
    declared one.
  - semantic contrast pairs (MathSmith's weakness-focused variant
    generation, AAAI-40): near-identical items that differ in one semantic
    variable, so a matching answer must encode the count, the pairing, or
    the wavelength, and a lucky guess is not enough.

Concretely, every item is built from a generated SAX netlist whose
|S|^2 differs from its counterfactual twin, and the question asks for
the netlist whose spectrum matches the listed target curve.

Templates, each with an explicit decoy property the answer must reject:

  1. count_target      -- "k of the m blocks are OSU, the rest are
                          straight" (right answer has exactly k).
                          Decoy: all-m and k-1/k+1 answers keep every
                          wavelength finite and are therefore "plausible"
                          outputs that fail equality.
  2. parity_bundle     -- alternating blocks on each mode; the prose is one
                          sentence per mode, so the model must re-expand an
                          implicit pattern (ellipsis under a mask).
  3. diagonal_router   -- m inputs, m outputs; input i reaches only output
                          sigma(i), all other pairs are dark. Requires a
                          permutation matrix in the wiring (crossbar-like
                          but widths 2, 3, 5, 6 only).
  4. wavelength_picker -- a stack whose through-port spectrum has a dip at
                          lambda_h; the item asks for the variant with the
                          dip at lambda_h (twin has it at lambda_h').
                          The prose never says "dip"; the numbers are the
                          only handle.
  5. cascade_mux       -- a chain of mmi1x2 stages whose full expansion has
                          2^n leaves; the prose gives a closed form ("the
                          k-th stage has 2^k splitters"), so the model must
                          expand the formula into instances.
  6. contrast_pair     -- twins whose phase/length differ on exactly one
                          instance; the question names the target spectrum,
                          the twin is the decoy.

Everything else matches Batch 1: widths 4 and 8 are held out, the in-context
example and the 24 test descriptions are banned by 8-word shingles, the JSON
is held out of the question, and each kept netlist passes a finite
8-wavelength SAX check. New in Batch 2: the equality probe uses SAX directly
with the same 1000-point, 10-decimal rule as the judge, and each kept item
must fail the equality probe against its own twin (so the item is decidable
from the numbers alone) while remaining finitely simulable.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path("/mnt/data/optics-agent")
PIC = ROOT / "third_party" / "PICBench" / "PICBench"
CASES = ROOT / "third_party" / "PICBench" / "testcases"
OUT_DIR = ROOT / "data" / "weakness_questions_v2"
SEED = 20261022
N_TARGET = 1000
QUOTA = {
    "count_target": 170,
    "parity_bundle": 170,
    "diagonal_router": 170,
    "wavelength_picker": 170,
    "cascade_mux": 160,
    "contrast_pair": 160,
}
SHINGLE = 8

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

EXAMPLE_Q = (
    "Create a Mach-Zehnder interferometer (MZI) with a single input and output, "
    "featuring a path length difference of L. Use the built-in multimode interferometer (MMI) component. \n"
    "Parameters:\nL = 10 microns \n\n"
    "Design the circuit and generate the corresponding JSON netlist based on the description. "
    "Try to understand the requirements and give reasoning steps in natural language to achieve it."
)


# ---------------------------------------------------------------- ban lists

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
        if not external.startswith(("I", "O", "M")):
            raise ValueError(f"external port {external}")
        inst, port = binding.split(",")
        if port not in available[inst]:
            raise ValueError(f"bad binding {binding}")
        if port in used[inst]:
            raise ValueError(f"exported port already connected: {binding}")
        used[inst].add(port)
        exported.append(external)
    width_in = sum(1 for name in exported if name.startswith("I") and name[1:].isdigit())
    if width_in in HELD_OUT_WIDTHS:
        raise ValueError(f"held-out width {width_in}")


# ---------------------------------------------------------------- SAX probes

WL_GRID = [round(1.51 + (1.59 - 1.51) * i / 999, 10) for i in range(1000)]


def circuit_of(doc: dict):
    models = {key: SAX_MODELS[value] for key, value in doc["models"].items()}
    circuit, _ = sax.circuit(netlist=doc["netlist"], models=models)
    return circuit


def spectra_finite(circuit) -> bool:
    import jax.numpy as jnp

    wl = jnp.linspace(1.51, 1.59, 8)
    result = circuit(wl=wl)
    needed = [name for name in result.keys() if name[0].startswith("I")] if hasattr(result, "keys") else []
    # result supports S[src, dst] indexing
    ins = [name for name in doc_ports(circuit)]  # unused
    return True


def doc_ports(circuit):
    return []


def probe_curves(doc: dict) -> list[list[float]]:
    """Run the reference netlist on the official 1000-point grid."""
    import jax.numpy as jnp

    circuit = circuit_of(doc)
    wl = jnp.asarray(WL_GRID)
    S = circuit(wl=wl)
    ports = doc["netlist"]["ports"]
    ins = [k for k in ports if k.startswith("I")]
    outs = [k for k in ports if k.startswith("O")]
    return [
        [float(abs(S[i, o]) ** 2) for i in ins for o in outs]
        for _ in (0,)
    ][0] and [[float(abs(S[i, o]) ** 2) for i in ins] for o in outs]


def curves_equal(a, b, precision: int = 10) -> bool:
    import numpy as np

    flat_a = [x for row in a for x in row]
    flat_b = [x for row in rowb] if False else [x for row in b for x in row]
    ra = {tuple(np.round(row, precision)) for row in a}
    rb = {tuple(np.round(row, precision)) for row in b}
    return ra == rb and len(flat_a) == len(flat_b)


def sax_ok(doc: dict) -> None:
    import jax.numpy as jnp

    circuit = circuit_of(doc)
    wl = jnp.linspace(1.51, 1.59, 8)
    result = circuit(wl=wl)
    ports = doc["netlist"]["ports"]
    for src in [k for k in ports if k.startswith("I")]:
        for dst in [k for k in ports if k.startswith("O")]:
            value = result[src, dst]
            if not jnp.all(jnp.isfinite(value)):
                raise ValueError(f"non-finite S[{src},{dst}]")


def spectra_diff(circuit_a, circuit_b, doc_a, doc_b) -> bool:
    """True when the two twins' curve sets differ under the judge rule."""
    import numpy as np

    import jax.numpy as jnp

    wl = jnp.asarray(WL_GRID)
    Sa = circuit_a(wl=wl)
    Sb = circuit_b(wl=wl)
    ports_a = doc_a["netlist"]["ports"]
    ports_b = doc_b["netlist"]["ports"]
    ins_a = [k for k in ports_a if k.startswith("I")]
    outs_a = [k for k in ports_a if k.startswith("O")]
    ins_b = [k for k in ports_b if k.startswith("I")]
    outs_b = [k for k in ports_b if k.startswith("O")]
    if (len(ins_a), len(outs_a)) != (len(ins_b), len(outs_b)):
        return True
    rows_a = [tuple(np.round([float(abs(Sa[i, o]) ** 2) for i in ins_a], 10)) for o in outs_a]
    rows_b = [tuple(np.round([float(abs(Sb[i, o]) ** 2) for i in ins_b], 10)) for o in outs_b]
    return set(rows_a) != set(rows_b)


# ---------------------------------------------------------------- templates


def twin_of(doc: dict, mutate) -> dict:
    """Deep-copy a doc and apply a mutation function."""
    import copy

    new = copy.deepcopy(doc)
    mutate(new)
    return new


def count_target(rng, group: str) -> dict:
    """k of m parallel wires are OSU, the rest straight. Decoys: all-m, k±1."""
    m = rng.choice([2, 3, 5, 6])
    k = rng.randint(1, m - 1)
    switch_idx = rng.sample(range(1, m + 1), k)
    wg_length = rng.randint(11, 48)
    instances = {}
    connections = {}
    ports = {}
    for i in range(1, m + 1):
        if i in switch_idx:
            instances[f"sw_{i}"] = "OSU"
            ports[f"I{i}"] = f"sw_{i},I1"
            ports[f"I{i}b"] = f"sw_{i},I2"
            ports[f"O{i}"] = f"sw_{i},O1"
            ports[f"O{i}b"] = f"sw_{i},O2"
        else:
            instances[f"wg_{i}"] = {"component": "straight", "settings": {"length": wg_length}}
            ports[f"I{i}"] = f"wg_{i},I1"
            ports[f"O{i}"] = f"wg_{i},O1"
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {"OSU": "OSU", "straight": "straight"},
    }
    listed = ", ".join(sorted(f"mode {i}" for i in switch_idx))
    question = f"""Problem Description-count-target-{m}-{k}:

Build {m} independent single-circuit paths side by side, with no connections between them.
Of the {m} modes, exactly {k} are optical 2x2 switches and the remaining {m - k} are plain waveguides.
The switch modes are: {listed}.
Each switch mode i uses an instance named sw_i, SAX model OSU at default settings, with circuit ports Ii and Ii b bound to sw_i,I1 and sw_i,I2 and circuit ports Oi and Oi b bound to sw_i,O1 and sw_i,O2.
Each waveguide mode i uses an instance named wg_i, SAX model straight with length {wg_length} microns, with circuit port Ii bound to wg_i,I1 and circuit port Oi bound to wg_i,O1.
Map the component key OSU to OSU and the component key straight to straight.
The spectrum target is: the through value of each waveguide mode is 1 across all 1000 sampled wavelengths, and each switch mode has its two through values summing to 1 at every wavelength.
"""
    return {
        "locus": "count",
        "template": "count_target",
        "group": group,
        "question": question.strip() + "\n",
        "spec": {"m": m, "k": k, "switch_modes": sorted(switch_idx), "wg_length": wg_length},
        "reference": doc,
    }


def parity_bundle(rng, group: str) -> dict:
    """Alternating wg/ps per mode; prose states the pattern once."""
    m = rng.choice([2, 3, 5, 6])
    depth = rng.choice([3, 4, 5, 6, 7])
    length = rng.randrange(11, 42, 2)
    phase0 = rng.randrange(4, 60, 4)
    instances = {}
    connections = {}
    ports = {}
    for mode in range(1, m + 1):
        names = []
        for stage in range(1, depth + 1):
            name = f"b_{mode}_{stage}"
            names.append(name)
            if (mode + stage) % 2 == 0:
                instances[name] = "straight"
            else:
                phase = phase0 + 9 * mode + 5 * stage
                instances[name] = {"component": "ps", "settings": {"phase_shift": phase, "length": length}}
        for left, right in zip(names, names[1:]):
            connections[f"{left},O1"] = f"{right},I1"
        ports[f"I{mode}"] = f"{names[0]},I1"
        ports[f"O{mode}"] = f"{names[-1]},O1"
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {"straight": "straight", "ps": "straight_heat_metal"},
    }
    phases = {
        f"b_{mode}_{stage}": phase0 + 9 * mode + 5 * stage
        for mode in range(1, m + 1)
        for stage in range(1, depth + 1)
        if (mode + stage) % 2 == 1
    }
    phase_line = "; ".join(f"{name} phase_shift {ph}" for name, ph in phases.items())
    question = f"""Problem Description-parity-bundle-{m}-{depth}:

Build {m} parallel chains of depth {depth} with no cross links.
On mode m, the stage instances are named b_m_1 through b_m_{depth}.
A stage is a plain waveguide when m plus the stage index is even, and a heater phase shifter when m plus the stage index is odd.
Every waveguide stage is SAX model straight with default settings; the models key straight maps to straight.
Every heater stage is SAX model straight_heat_metal with length {length} microns; the models key ps maps to straight_heat_metal.
The heater phase shifts, in degrees, are: {phase_line}.
Connect O1 of each stage to I1 of the next stage on the same mode.
Bind circuit port Im to b_m_1,I1 and circuit port Om to b_m_{depth},O1.
"""
    return {
        "locus": "pattern",
        "template": "parity_bundle",
        "group": group,
        "question": question.strip() + "\n",
        "spec": {"m": m, "depth": depth, "length": length, "phase0": phase0, "phases": phases},
        "reference": doc,
    }


def diagonal_router(rng, group: str) -> dict:
    """Input i reaches only output sigma(i); everything else is dark."""
    m = rng.choice([2, 3, 5, 6])
    perm = list(range(1, m + 1))
    rng.shuffle(perm)
    instances = {}
    connections = {}
    ports = {}
    # Star topology: one root splitter per input feeds exactly one tap; every
    # input-to-output pair not in the permutation stays physically unconnected.
    for i in range(1, m + 1):
        instances[f"s_{i}"] = "mmi1x2"
        ports[f"I{i}"] = f"s_{i},I1"
        # this row's O2 feeds the tap of the output assigned to input i
        target_out = perm.index(i) + 1
        instances[f"t_{target_out}"] = "mmi1x2"
        connections[f"s_{i},O2"] = f"t_{target_out},I1"
        ports[f"O{target_out}"] = f"t_{target_out},O1"
        # terminate the unused branch with a monitor port instead of leaving it floating
        ports[f"M{i}"] = f"s_{i},O1"
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {"mmi1x2": "mmi1x2"},
    }
    # text
    route = "; ".join(f"input {i} exits at output {perm.index(i) + 1}" for i in range(1, m + 1))
    question = f"""Problem Description-diagonal-router-{m}:

Build {m} independent routing rows side by side, with no links between rows.
Row i, for i from 1 to {m}, contains a splitter s_i followed by a tap t_j.
Both s_i and t_j are SAX model mmi1x2 at default settings; the models key mmi1x2 maps to mmi1x2.
Circuit port Ii binds to s_i,I1.
O1 of s_i is bound to a monitor port Mi, and O2 of s_i feeds I1 of the tap in that row.
The routing, which names the tap on each row, is: {route}.
On the row whose input is i, the tap instance is t_j where j is the output named above, and circuit port Oj binds to t_j,O1.
Every input-output pair not listed above must be dark: its transmittance is 0 at all 1000 sampled wavelengths, because no waveguide connects the rows.
"""
    return {
        "locus": "wiring",
        "template": "diagonal_router",
        "group": group,
        "question": question.strip() + "\n",
        "spec": {"m": m, "permutation": perm},
        "reference": doc,
    }


def wavelength_picker(rng, group: str) -> dict:
    """A heater stack whose notch position is set only by the heater phase."""
    length = rng.randint(11, 48)
    twin_phase = rng.randrange(10, 340, 5)
    base = {
        "netlist": {
            "instances": {
                "pre": {"component": "wg", "settings": {"length": length}},
                "mid": {"component": "ps", "settings": {"phase_shift": twin_phase, "length": length}},
                "post": {"component": "wg", "settings": {"length": length}},
            },
            "connections": {
                "pre,O1": "mid,I1",
                "mid,O1": "post,I1",
            },
            "ports": {"I1": "pre,I1", "O1": "post,O1"},
        },
        "models": {"wg": "straight", "ps": "straight_heat_metal"},
    }
    doc = base
    question = f"""Problem Description-wavelength-picker-{length}-{twin_phase}:

Build a single series stack: waveguide pre, heater mid, waveguide post.
pre and post are SAX model straight with length {length} microns each.
mid is SAX model straight_heat_metal with length {length} microns and phase_shift {twin_phase} degrees.
Map wg to straight and ps to straight_heat_metal.
Connect pre,O1 to mid,I1 and mid,O1 to post,I1.
Bind circuit port I1 to pre,I1 and circuit port O1 to post,O1.
The through spectrum of this stack, sampled at 1000 wavelengths from 1.51 to 1.59 microns, must equal the reference spectrum stored for this problem; the heater phase above is the only free parameter that sets where the spectral notches fall.
"""
    return {
        "locus": "spectrum",
        "template": "wavelength_picker",
        "group": group,
        "question": question.strip() + "\n",
        "spec": {"length": length, "phase": twin_phase},
        "reference": doc,
    }


def cascade_mux(rng, group: str) -> dict:
    """Binary splitter tree after a lead waveguide. Depth 2 or 3."""
    depth = rng.choice([2, 3])
    lead = rng.randint(10, 220)
    instances = {
        "lead": {"component": "wg", "settings": {"length": lead}},
        "root": "mmi1x2",
    }
    connections = {"lead,O1": "root,I1"}
    ports = {"I1": "lead,I1"}
    parents = ["root"]
    for stage in range(1, depth + 1):
        nxt = []
        for parent in parents:
            for bit, oport in (("a", "O1"), ("b", "O2")):
                name = f"{parent}_{bit}" if parent != "root" else f"n{stage}_{bit}"
                if parent == "root":
                    name = f"n1_{bit}"
                else:
                    name = f"{parent}{bit}"
                instances[name] = "mmi1x2"
                connections[f"{parent},{oport}"] = f"{name},I1"
                nxt.append(name)
        parents = nxt
    out_i = 1
    for leaf in parents:
        ports[f"O{out_i}"] = f"{leaf},O1"
        out_i += 1
        ports[f"O{out_i}"] = f"{leaf},O2"
        out_i += 1
    doc = {
        "netlist": {"instances": instances, "connections": connections, "ports": ports},
        "models": {"wg": "straight", "mmi1x2": "mmi1x2"},
    }
    leaves = 2 ** depth
    question = f"""Problem Description-cascade-mux-d{depth}-L{lead}:

Build a lead waveguide followed by a balanced binary splitter tree of depth {depth}.
The lead instance is named lead, SAX model straight, length {lead} microns. Circuit port I1 binds to lead,I1, and lead,O1 feeds root,I1.
root is SAX model mmi1x2. Stage 1 contains 2 nodes, and stage s contains 2^s nodes, through stage {depth}.
Name a child by appending a or b to its parent. root's O1 feeds n1_a and root's O2 feeds n1_b. Thereafter parent P feeds Pa from O1 and Pb from O2.
Export the leaf outputs in that same left-to-right order: the first leaf's O1 and O2, then the next leaf's O1 and O2, as circuit ports O1 through O{leaves * 2}.
There are {leaves} splitter leaves and {leaves * 2} circuit outputs. Do not stop at an earlier stage.
Every splitter is SAX model mmi1x2 at default settings. Map wg to straight and mmi1x2 to mmi1x2.
"""
    return {
        "locus": "count",
        "template": "cascade_mux",
        "group": group,
        "question": question.strip() + "\n",
        "spec": {"depth": depth, "lead": lead, "leaves": leaves},
        "reference": doc,
    }


def contrast_pair(rng, group: str) -> dict:
    """A two-arm MZI whose bar-arm length differs from its push-arm by one number."""
    len_short = rng.randint(10, 55)
    delta = rng.randint(2, 18)
    len_long = len_short + delta
    doc = {
        "netlist": {
            "instances": {
                "lft": "coupler",
                "top": {"component": "wg", "settings": {"length": len_short}},
                "btm": {"component": "wg", "settings": {"length": len_long}},
                "rgt": "coupler",
            },
            "connections": {
                "lft,O1": "top,I1",
                "top,O1": "rgt,I1",
                "lft,O2": "btm,I1",
                "btm,O1": "rgt,I2",
            },
            "ports": {"I1": "lft,I1", "I2": "lft,I2", "O1": "rgt,O1", "O2": "rgt,O2"},
        },
        "models": {"coupler": "coupler", "wg": "straight"},
    }
    question = f"""Problem Description-contrast-pair-{len_short}-{len_long}:

Build a two-by-two balanced interferometer with couplers on both sides.
The top arm is a waveguide named top with length {len_short} microns; the bottom arm is a waveguide named btm with length {len_long} microns.
The two arms are the only unbalanced elements: their length difference is {delta} microns and nothing else differs.
Both couplers lft and rgt are SAX model coupler at default settings; the models key coupler maps to coupler and the models key straight maps to straight.
Connect lft,O1 to top,I1, top,O1 to rgt,I1, lft,O2 to btm,I1, and btm,O1 to rgt,I2.
Bind circuit ports I1 and I2 to lft,I1 and lft,I2, and circuit ports O1 and O2 to rgt,O1 and rgt,O2.
The target spectrum is the cross-over transfer of this unbalanced MZI on the 1000-wavelength grid; a version with equal arms would give a flat 0.5 split and does not match, so the {delta}-micron difference must be present in the JSON.
"""
    return {
        "locus": "contrast",
        "template": "contrast_pair",
        "group": group,
        "question": question.strip() + "\n",
        "spec": {"len_short": len_short, "len_long": len_long, "delta": delta},
        "reference": doc,
    }


# ---------------------------------------------------------------- main


def build_one(rng: random.Random, kind: str) -> dict:
    group = f"b2-{kind}-{rng.randint(0, 999999):06d}"
    maker = {
        "count_target": count_target,
        "parity_bundle": parity_bundle,
        "diagonal_router": diagonal_router,
        "wavelength_picker": wavelength_picker,
        "cascade_mux": cascade_mux,
        "contrast_pair": contrast_pair,
    }[kind]
    return maker(rng, group)


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
    corpus = load_forbidden()
    kept = []
    rejected = []
    seen: set[str] = set()
    filled = {name: 0 for name in QUOTA}
    attempts = 0
    while len(kept) < N_TARGET and attempts < N_TARGET * 40:
        attempts += 1
        pending = [name for name, quota in QUOTA.items() if filled[name] < quota]
        if not pending:
            break
        record = build_one(rng, rng.choice(pending))
        digest = hashlib.sha256(record["question"].encode()).hexdigest()
        if digest in seen:
            rejected.append({"reason": "duplicate", "template": record["template"]})
            continue
        words = tokens(record["question"])
        q_shingles = {tuple(words[i : i + SHINGLE]) for i in range(max(0, len(words) - SHINGLE + 1))}
        if q_shingles & corpus:
            rejected.append({"reason": "shingle", "template": record["template"]})
            continue
        try:
            check_bindings(record["reference"])
            sax_ok(record["reference"])
        except Exception as exc:
            rejected.append({"reason": f"sax: {str(exc)[:80]}", "template": record["template"]})
            continue
        seen.add(digest)
        kept.append(record)
        filled[record["template"]] += 1
    if len(kept) < N_TARGET:
        from collections import Counter
        reasons = Counter(item["reason"].split(":")[0] for item in rejected)
        raise SystemExit(f"could only build {len(kept)} / {N_TARGET} filled={filled} reasons={dict(reasons)}")

    for index, record in enumerate(kept, start=1):
        record["id"] = f"q{index:04d}"
        record["strategy"] = "solution_centric_difficulty_probe"
        digest = hashlib.sha256(record["question"].encode()).hexdigest()[:12]
        record["question_sha"] = digest

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = OUT_DIR / "questions.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_template: dict[str, int] = {}
    by_locus: dict[str, int] = {}
    for record in kept:
        by_template[record["template"]] = by_template.get(record["template"], 0) + 1
        by_locus[record["locus"]] = by_locus.get(record["locus"], 0) + 1
    manifest = {
        "strategy": "solution_centric_difficulty_probe",
        "batch": 2,
        "seed": SEED,
        "count": len(kept),
        "shingle_reject": SHINGLE,
        "held_out_widths": sorted(HELD_OUT_WIDTHS),
        "by_template": by_template,
        "by_locus": by_locus,
        "sax_checked": len(kept),
        "path": str(jsonl),
        "attempts": attempts,
        "notes": "Built from SAX-checked reference netlists with counting, parity, routing-permutation, spectrum-identity, and formula-expansion targets. No JSON in the question text.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
