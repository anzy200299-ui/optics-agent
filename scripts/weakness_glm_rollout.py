"""Teacher rollout on the 1000 weakness questions for later SFT.

Reads GLM_BASE and GLM_API_KEY. Does not touch the 24 PICBench test items.
A sample is kept only when <result> is in the visible content and the JSON
parses as a netlist. Reasoning and the answer share max_tokens.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from picbench_qwen_zero import EXAMPLE_A, EXAMPLE_Q, PIC, PROBLEM_PROMPT, extract_netlist

ROOT = Path("/mnt/data/optics-agent")
QUESTIONS = ROOT / "data" / "weakness_questions" / "questions.jsonl"


def messages_for(question: str) -> list[dict[str, str]]:
    system_prompt = (PIC / "system_prompt.txt").read_text()
    body = question.split("Problem Description-", 1)[1]
    user = body + "\n" + PROBLEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": EXAMPLE_Q},
        {"role": "assistant", "content": EXAMPLE_A},
        {"role": "user", "content": user},
        {"role": "user", "content": user},
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
    with urllib.request.urlopen(req, timeout=2400) as resp:
        return json.loads(resp.read().decode())


def reply_text(message: dict) -> str:
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if "<result>" in content:
        return content
    if "<result>" in reasoning:
        return reasoning
    return content or reasoning


def netlist_ok(text: str | None) -> bool:
    if not text:
        return False
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and "netlist" in obj and "models" in obj


def load_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return netlist_ok(payload.get("netlist")) and "<result>" in (payload.get("content") or "")


def one(base: str, key: str, model: str, out: Path, record: dict, max_tokens: int) -> str:
    qid = record["id"]
    dest = out / f"{qid}.json"
    if load_done(dest):
        return f"skip {qid}"
    messages = messages_for(record["question"])
    last_error = None
    for attempt in range(4):
        try:
            data = chat(base, key, model, messages, max_tokens)
            choice = data["choices"][0]
            message = choice.get("message") or {}
            text = reply_text(message)
            netlist = extract_netlist(text)
            content = message.get("content") or ""
            payload = {
                "id": qid,
                "locus": record.get("locus"),
                "template": record.get("template"),
                "model": model,
                "max_tokens": max_tokens,
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage"),
                "result_in_content": "<result>" in content,
                "netlist_ok": netlist_ok(netlist),
                "messages": messages,
                "content": content,
                "reasoning_content": message.get("reasoning_content"),
                "response": text,
                "netlist": netlist,
            }
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            usage = data.get("usage") or {}
            return (
                f"wrote {qid} finish {choice.get('finish_reason')} "
                f"completion {usage.get('completion_tokens')} "
                f"result_in_content {payload['result_in_content']} "
                f"netlist_ok {payload['netlist_ok']}"
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:500]
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 means all questions")
    args = parser.parse_args()
    base = os.environ["GLM_BASE"]
    key = os.environ["GLM_API_KEY"]
    records = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]
    out = ROOT / "logs" / "sft" / args.model
    print(f"{args.model} jobs {len(records)} workers {args.workers} max_tokens {args.max_tokens}", flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, base, key, args.model, out, record, args.max_tokens) for record in records]
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
