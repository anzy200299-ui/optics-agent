"""Score saved Qwen replies with PICBench's SAX evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import REPO

ROOT = REPO
PIC_DIR = ROOT / "third_party" / "PICBench" / "PICBench"
OUT = ROOT / "logs" / "picbench" / "qwen35-9b"
sys.path.insert(0, str(PIC_DIR))

import os

os.chdir(PIC_DIR)
from evaluation import evaluate  # noqa: E402


def pass_at(c: int, n: int, k: int) -> float:
    if n - c < k:
        return 1.0
    # unbiased estimator, n and k are small
    from math import comb

    return 1.0 - comb(n - c, k) / comb(n, k)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="qwen35-9b",
        help="directory name under logs/picbench to score",
    )
    args = parser.parse_args()
    OUT = ROOT / "logs" / "picbench" / args.model
    designs = sorted(path.name for path in OUT.iterdir() if path.is_dir())
    rows = []
    syntax_1 = []
    func_1 = []
    syntax_5 = []
    func_5 = []
    for name in designs:
        samples = sorted(name_path for name_path in (OUT / name).glob("s*.json"))
        syntax = 0
        func = 0
        details = []
        for sample in samples:
            payload = json.loads(sample.read_text())
            netlist = payload.get("netlist")
            if not netlist:
                verdict = "no result part"
            else:
                verdict = evaluate(netlist, name)
            if str(verdict).startswith("functional check passed"):
                func += 1
            elif str(verdict).startswith("functional error"):
                syntax += 1
            details.append({"sample": sample.name, "verdict": verdict})
            print(f"{name} {sample.name}: {str(verdict)[:160]}", flush=True)
        n = len(samples)
        syntax_c = syntax + func
        syntax_1.append(pass_at(syntax_c, n, 1))
        func_1.append(pass_at(func, n, 1))
        syntax_5.append(pass_at(syntax_c, n, min(5, n)))
        func_5.append(pass_at(func, n, min(5, n)))
        rows.append({"design": name, "n": n, "syntax": syntax, "func": func, "details": details})
    summary = {
        "n_designs": len(rows),
        "syntax_pass1": sum(syntax_1) / len(syntax_1) if rows else None,
        "func_pass1": sum(func_1) / len(func_1) if rows else None,
        "syntax_pass5": sum(syntax_5) / len(syntax_5) if rows else None,
        "func_pass5": sum(func_5) / len(func_5) if rows else None,
        "designs": rows,
    }
    dest = OUT / "summary.json"
    dest.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"designs {summary['n_designs']} "
        f"syntax@1 {summary['syntax_pass1']:.4f} func@1 {summary['func_pass1']:.4f} "
        f"syntax@5 {summary['syntax_pass5']:.4f} func@5 {summary['func_pass5']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
