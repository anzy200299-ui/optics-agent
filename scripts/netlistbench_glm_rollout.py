"""Roll out glm-5.3-shouyun on the 1000 synthetic NetlistBench questions.

Reads GLM_BASE and GLM_API_KEY. The question text is the user message.
The reference answer is not sent. Output is a trace, not a PICBench sample,
and it is not written into the 2342-case benchmark run directories.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from repo_paths import NETLIST_TRAJ_DIR, REPO

ROOT = REPO
QUESTIONS = ROOT / "data" / "netlistbench_questions" / "questions.jsonl"
SYSTEM = "Follow the user instruction exactly."


def messages_for(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question.strip()},
    ]


def chat(base: str, key: str, model: str, messages: list[dict[str, str]], max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    # Reasoning and the visible answer share this budget.
    with urllib.request.urlopen(req, timeout=2400) as resp:
        return json.loads(resp.read().decode())


def load_done(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool((payload.get("content") or "").strip())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def one(base: str, key: str, model: str, out: Path, record: dict, max_tokens: int) -> str:
    qid = record["id"]
    dest = out / f"{qid}.json"
    if load_done(dest):
        return f"skip {qid}"
    messages = messages_for(record["question"])
    last_error = "unknown"
    for attempt in range(4):
        try:
            data = chat(base, key, model, messages, max_tokens)
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            if not content.strip():
                last_error = f"empty finish {choice.get('finish_reason')}"
            else:
                usage = data.get("usage") or {}
                payload = {
                    "id": qid,
                    "locus": record.get("locus"),
                    "template": record.get("template"),
                    "model": model,
                    "max_tokens": max_tokens,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": usage,
                    "messages": messages,
                    "content": content,
                    "reasoning_content": message.get("reasoning_content"),
                }
                write_json(dest, payload)
                return (
                    f"wrote {qid} finish {choice.get('finish_reason')} "
                    f"chars {len(content)} completion {usage.get('completion_tokens')}"
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:400]
            last_error = f"HTTP {exc.code} {body}"
            if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                break
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{qid}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="glm-5.3-shouyun")
    parser.add_argument("--max-tokens", type=int, default=131072)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="0 means all questions")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    base = os.environ["GLM_BASE"]
    key = os.environ["GLM_API_KEY"]
    records = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]
    if args.out is not None:
        out = args.out
    elif args.model == "glm-5.3-shouyun":
        out = NETLIST_TRAJ_DIR
    else:
        out = ROOT / "logs" / "netlistbench" / args.model
    print(
        f"{args.model} jobs {len(records)} workers {args.workers} max_tokens {args.max_tokens}",
        flush=True,
    )
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(one, base, key, args.model, out, record, args.max_tokens) for record in records
        ]
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {exc}", flush=True)
    print(f"done failures {failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
