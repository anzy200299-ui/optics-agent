"""NetlistBench, temperature 0, one shot, official prompts.

DeepSeek-V4-Pro uses the same dawnfm gateway as the PICBench run.
Qwen3.5-9B runs locally with thinking off and greedy decoding.
Family directories contain only the model text, so the official oracle
scripts can score them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from repo_paths import MODEL_9B, REPO

ROOT = REPO
NB = ROOT / "third_party" / "NetlistBench"
sys.path.insert(0, str(NB / "scripts"))

from run_all_families import _build_case_prompt, _discover_family_dirs  # noqa: E402
from run_recognition_on_cases import SYSTEM_PROMPT, discover_cases, render_user_prompt  # noqa: E402

EDIT = NB / "benchmark" / "Cases" / "v2_edit_raw"
REC = NB / "benchmark" / "Cases" / "v2_recognition_raw"
MODEL_PATH = MODEL_9B


def jobs() -> list[dict]:
    os.chdir(NB)
    rows: list[dict] = []
    for family in _discover_family_dirs(EDIT):
        for case_json in sorted((family / "cases").rglob("case.json")):
            case = json.loads(case_json.read_text(encoding="utf-8"))
            system, user, extension, _cache = _build_case_prompt(
                case, strip_hint_comments=True, representation="spice"
            )
            rows.append(
                {
                    "id": case["case_id"],
                    "system": system,
                    "user": user,
                    "rel": Path("edit") / family.name / f"{case['case_id']}{extension}",
                }
            )
    for split in ("flat", "subckt"):
        for case_dir in discover_cases(REC / split / "cases"):
            case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "id": case["case_id"],
                    "system": SYSTEM_PROMPT,
                    "user": render_user_prompt(case_dir, case, "spice"),
                    "rel": Path("recognition") / split / f"{case['case_id']}.txt",
                }
            )
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    body = text if text.endswith("\n") else text + "\n"
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def chat(base: str, key: str, model: str, system: str, user: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2400) as response:
        return json.loads(response.read().decode())


def visible_text(data: dict) -> tuple[str, str, dict]:
    choice = data["choices"][0]
    message = choice.get("message") or {}
    return (message.get("content") or "").strip(), choice.get("finish_reason") or "", data.get("usage") or {}


def one_api(base: str, key: str, model: str, out: Path, job: dict, max_tokens: int) -> str:
    dest = out / job["rel"]
    if dest.exists() and dest.stat().st_size > 0:
        return f"skip {job['id']}"
    last_error = "unknown"
    for attempt in range(4):
        try:
            data = chat(base, key, model, job["system"], job["user"], max_tokens)
            text, finish, usage = visible_text(data)
            if text:
                write_text(dest, text)
                return (
                    f"wrote {job['id']} finish {finish} chars {len(text)} "
                    f"completion {usage.get('completion_tokens')}"
                )
            last_error = f"empty finish {finish}"
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:240]
            last_error = f"HTTP {exc.code} {body}"
            if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                break
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        time.sleep(2 * (attempt + 1))
    return f"FAIL {job['id']} {last_error}"


def run_api(pending: list[dict], out: Path, workers: int, max_tokens: int, model: str) -> None:
    base = os.environ["GLM_BASE"]
    key = os.environ["GLM_API_KEY"]
    print(f"{model} jobs {len(pending)} workers {workers} max_tokens {max_tokens}", flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_api, base, key, model, out, job, max_tokens) for job in pending]
        for future in as_completed(futures):
            line = future.result()
            print(line, flush=True)
            if line.startswith("FAIL"):
                failures += 1
    print(f"done failures {failures}", flush=True)


def run_qwen(pending: list[dict], out: Path, max_new_tokens: int, model_path: str) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    print(f"qwen jobs {len(pending)} max_new_tokens {max_new_tokens} model {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    failures = 0
    for job in pending:
        dest = out / job["rel"]
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip {job['id']}", flush=True)
            continue
        messages = [
            {"role": "system", "content": job["system"]},
            {"role": "user", "content": job["user"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_prompt = int(inputs["input_ids"].shape[1])
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        elapsed = time.perf_counter() - started
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        n_new = int(new_tokens.shape[0])
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not text:
            failures += 1
            print(f"FAIL {job['id']} empty tokens {n_new} prompt {n_prompt} sec {elapsed:.1f}", flush=True)
            continue
        write_text(dest, text)
        print(
            f"wrote {job['id']} chars {len(text)} tokens {n_new} prompt {n_prompt} "
            f"sec {elapsed:.1f} hit_cap {n_new >= max_new_tokens}",
            flush=True,
        )
    print(f"done failures {failures}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["api", "qwen"], required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.out and not Path(args.out).is_absolute():
        args.out = str(ROOT / args.out)
    if args.backend == "qwen":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    rows = jobs()
    if args.limit:
        rows = rows[: args.limit]
    if args.backend == "qwen":
        rows = [row for index, row in enumerate(rows) if index % args.shards == args.gpu]
        out = Path(args.out) if args.out else ROOT / "logs" / "netlistbench" / "qwen35-9b"
        model_path = args.model_path or str(MODEL_PATH)
        run_qwen(rows, out, args.max_new_tokens, model_path)
    else:
        out = Path(args.out) if args.out else ROOT / "logs" / "netlistbench" / args.model
        run_api(rows, out, args.workers, args.max_tokens, args.model)


if __name__ == "__main__":
    main()
