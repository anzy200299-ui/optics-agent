"""Keep PICBench teacher trajectories whose visible netlist matches the reference.

The teacher is glm-5.3-shouyun on the 1000 weakness questions. A trajectory is
kept only when the visible <result> JSON builds in SAX and its transmission
spectra match the question's reference on 1000 wavelengths, rounded to 10
decimals, the same comparison PICBench uses for a functional pass.

The only edit allowed is an identity entry in models. If an instance uses a
built-in SAX name that was left out of models, that name is added and the
graph is not touched. The repaired JSON replaces the object inside <result>.
reasoning_content is archived on the trajectory and is not a training target.

ms-swift merges consecutive user turns, so the duplicated PICBench user
question is embedded as one user string that still renders as two user turns.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from pathlib import Path

from repo_paths import PICBENCH_TRAJ_DIR, PICBENCH_VISIBLE_JSONL, REPO

ROOT = REPO
PIC = ROOT / "third_party" / "PICBench" / "PICBench"
QUESTIONS = ROOT / "data" / "weakness_questions" / "questions.jsonl"
TRAJ_DIR = PICBENCH_TRAJ_DIR
PREFIX_SCORE = ROOT / "logs" / "sft" / "glm-5.3-shouyun-score.json"
DEFAULT_JSONL = PICBENCH_VISIBLE_JSONL
DEFAULT_REPORT = ROOT / "logs" / "sft" / "picbench_glm53_visible_filter.json"

WL_START = 1.51
WL_STOP = 1.59
N_WL = 1000
PRECISION = 10


def locate_result(content: str) -> tuple[dict, int, int] | None:
    marker = "<result>"
    at = content.find(marker)
    if at < 0:
        return None
    body_at = at + len(marker)
    close = content.find("</result>", body_at)
    search_end = close if close >= 0 else len(content)
    start = content.find("{", body_at, search_end)
    if start < 0:
        return None
    try:
        obj, end = json.JSONDecoder().raw_decode(content[start:])
    except json.JSONDecodeError:
        return None
    if end > search_end - start:
        return None
    if not isinstance(obj, dict) or "netlist" not in obj or "models" not in obj:
        return None
    if not isinstance(obj["netlist"], dict) or not isinstance(obj["models"], dict):
        return None
    return obj, start, start + end


def component_names(netlist: dict) -> list[str]:
    names = []
    for inst in netlist.get("instances", {}).values():
        if isinstance(inst, str):
            names.append(inst)
        elif isinstance(inst, dict) and isinstance(inst.get("component"), str):
            names.append(inst["component"])
    return names


def identity_repair(doc: dict, builtins: set[str]) -> tuple[dict, list[str]] | None:
    """Add missing built-in model names. Instances, connections, and ports stay put."""
    models = dict(doc["models"])
    added: list[str] = []
    for name in component_names(doc["netlist"]):
        if name in models or name in added:
            continue
        if name in builtins:
            added.append(name)
    if not added:
        return None
    for name in added:
        models[name] = name
    repaired = {"netlist": doc["netlist"], "models": models}
    return repaired, added


def replace_result(content: str, start: int, end: int, doc: dict) -> str:
    rendered = json.dumps(doc, ensure_ascii=False, indent=2)
    return content[:start] + rendered + content[end:]


def training_messages(messages: list[dict], answer: str) -> list[dict]:
    if [m.get("role") for m in messages] != ["system", "user", "assistant", "user", "user"]:
        raise ValueError("unexpected prompt roles")
    question = messages[3]["content"].strip()
    if messages[4]["content"].strip() != question:
        raise ValueError("the two user turns differ")
    # One swift user turn, two PICBench user turns after the chat template is applied.
    query = f"{question}<|im_end|>\n<|im_start|>user\n{question}"
    return [
        {"role": "system", "content": messages[0]["content"]},
        {"role": "user", "content": messages[1]["content"]},
        {"role": "assistant", "content": messages[2]["content"], "loss": False},
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer.strip()},
    ]


def _init_worker() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.chdir(PIC)
    sys.path.insert(0, str(PIC))
    import jax.numpy as jnp
    import numpy as np
    import sax
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

    global _SAX, _JNP, _NP, _MODELS, _BUILTINS
    _SAX = sax
    _JNP = jnp
    _NP = np
    _MODELS = {
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
    _BUILTINS = set(_MODELS)


def _curves(doc: dict):
    models = {}
    for key, value in doc["models"].items():
        if value not in _MODELS:
            raise KeyError(value)
        models[key] = _MODELS[value]
    circuit, _ = _SAX.circuit(netlist=doc["netlist"], models=models)
    wl = _JNP.linspace(WL_START, WL_STOP, N_WL)
    spectrum = circuit(wl=wl)
    ports = doc["netlist"]["ports"]
    inputs = [key for key in ports if str(key).startswith("I")]
    outputs = [key for key in ports if str(key).startswith("O")]
    if not inputs or not outputs:
        raise ValueError("no I/O ports")
    curves = []
    for src in inputs:
        for dst in outputs:
            power = _NP.asarray(abs(spectrum[src, dst]) ** 2).tolist()
            if not _NP.all(_NP.isfinite(power)):
                raise ValueError(f"non-finite S[{src},{dst}]")
            curves.append(tuple(_NP.round(power, PRECISION)))
    return frozenset(curves)


def _base(task: dict, **extra) -> dict:
    row = {
        "id": task["id"],
        "locus": task["locus"],
        "template": task["template"],
    }
    row.update(extra)
    return row


def _score(task: dict) -> dict:
    try:
        reference = _curves(task["reference"])
        located = locate_result(task["content"])
        if located is None:
            return _base(task, verdict="bad_json", detail="visible result is not a netlist object")
        doc, start, end = located
        try:
            got = _curves(doc)
        except Exception as exc:  # noqa: BLE001
            repaired = identity_repair(doc, _BUILTINS)
            if repaired is None:
                return _base(task, verdict="build_error", detail=f"{type(exc).__name__}: {exc}"[:400])
            new_doc, added = repaired
            if new_doc["netlist"] != doc["netlist"]:
                raise RuntimeError("repair changed the netlist")
            try:
                got = _curves(new_doc)
            except Exception as repaired_exc:  # noqa: BLE001
                return _base(
                    task,
                    verdict="build_error",
                    detail=f"after model-name repair: {type(repaired_exc).__name__}: {repaired_exc}"[:400],
                    added_models=added,
                )
            if got != reference:
                return _base(
                    task,
                    verdict="spectrum_mismatch",
                    detail="model-name repair builds, spectrum differs",
                    added_models=added,
                )
            answer = replace_result(task["content"], start, end, new_doc)
            return _base(
                task,
                verdict="func_repaired",
                added_models=added,
                messages=training_messages(task["messages"], answer),
            )
        if got != reference:
            return _base(task, verdict="spectrum_mismatch", detail="built, spectrum differs")
        return _base(task, verdict="func", messages=training_messages(task["messages"], task["content"]))
    except Exception as exc:  # noqa: BLE001
        return _base(task, verdict="error", detail=f"{type(exc).__name__}: {exc}"[:400])


def load_questions() -> dict[str, dict]:
    rows = {}
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["id"]] = row
    return rows


def prefix_check(rows: list[dict]) -> dict:
    if not PREFIX_SCORE.exists():
        return {"checked": False}
    previous = json.loads(PREFIX_SCORE.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in rows}
    mismatches = []
    rewritten = []
    for old in previous["rows"]:
        got = by_id.get(old["id"])
        if got is None:
            mismatches.append({"id": old["id"], "old": old["verdict"], "new": "missing"})
            continue
        if old["verdict"] == "func" and got["verdict"] != "func":
            mismatches.append({"id": old["id"], "old": "func", "new": got["verdict"], "detail": got.get("detail", "")})
        elif old["verdict"] == "syntax" and got["verdict"] != "func_repaired":
            mismatches.append(
                {"id": old["id"], "old": "syntax", "new": got["verdict"], "detail": got.get("detail", "")}
            )
        elif old["verdict"] == "bad_json" and got["verdict"] in {"func", "func_repaired"}:
            # q0108 was unparseable in the 175-file prefix. The finished rollout
            # rewrote that file; the copy on disk now spectrum-matches.
            rewritten.append({"id": old["id"], "old": "bad_json", "new": got["verdict"]})
        elif old["verdict"] == "bad_json" and got["verdict"] not in {"bad_json", "build_error"}:
            mismatches.append({"id": old["id"], "old": "bad_json", "new": got["verdict"]})
    return {
        "checked": True,
        "prefix_n": previous["n"],
        "mismatches": mismatches[:20],
        "n_mismatch": len(mismatches),
        "prefix_bad_json_now_matched": rewritten,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 scores every trajectory")
    parser.add_argument("--traj-dir", type=Path, default=TRAJ_DIR)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    os.environ["JAX_PLATFORMS"] = "cpu"
    questions = load_questions()
    paths = sorted(args.traj_dir.glob("q*.json"))
    if args.limit:
        paths = paths[: args.limit]
    payloads = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        qid = payload["id"]
        if qid not in questions:
            raise SystemExit(f"{qid} has no weakness question")
        payloads.append(payload)
    print(f"scoring {len(payloads)} trajectories", flush=True)
    tasks = []
    for payload in payloads:
        qid = payload["id"]
        tasks.append(
            {
                "id": qid,
                "content": payload.get("content") or "",
                "messages": payload["messages"],
                "locus": payload.get("locus"),
                "template": payload.get("template"),
                "reference": questions[qid]["reference"],
            }
        )

    if args.workers <= 1:
        scored = [_score(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        scored = []
        with ctx.Pool(args.workers, initializer=_init_worker) as pool:
            for index, row in enumerate(pool.imap_unordered(_score, tasks, chunksize=1), start=1):
                scored.append(row)
                if index % 50 == 0 or index == len(tasks):
                    print(f"teacher spectra {index}/{len(tasks)} {row['id']} {row['verdict']}", flush=True)

    scored.sort(key=lambda row: row["id"])
    counts = Counter(row["verdict"] for row in scored)
    by_locus = Counter()
    kept_rows = []
    report_rows = []
    for row in scored:
        report_rows.append(
            {
                "id": row["id"],
                "verdict": row["verdict"],
                "locus": row.get("locus"),
                "template": row.get("template"),
                "detail": row.get("detail", ""),
                "added_models": row.get("added_models", []),
            }
        )
        if row["verdict"] not in {"func", "func_repaired"}:
            continue
        by_locus[row["locus"]] += 1
        kept_rows.append({"messages": row["messages"]})

    agreement = prefix_check(scored) if not args.limit else {"checked": False, "reason": "limited run"}
    report = {
        "teacher": "glm-5.3-shouyun",
        "n_scored": len(scored),
        "n_kept": len(kept_rows),
        "counts": dict(counts),
        "kept_by_locus": dict(by_locus),
        "wavelengths": {"start_um": WL_START, "stop_um": WL_STOP, "n": N_WL, "round_decimals": PRECISION},
        "repair": "identity model names only; instances, connections, and ports are unchanged",
        "target": "visible content, not reasoning_content",
        "prefix_agreement": agreement,
        "rows": report_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("w", encoding="utf-8") as handle:
        for row in kept_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"n_scored": len(scored), "n_kept": len(kept_rows), "counts": dict(counts), "prefix": agreement}, indent=2), flush=True)
    if agreement.get("checked") and agreement["n_mismatch"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
