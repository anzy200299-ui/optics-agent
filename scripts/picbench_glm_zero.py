"""Zero-shot API rollout on PICBench, same prompt construction as the local 9B run.

No restrictions, no error feedback, n=5, temperature 1. The user question is
appended twice, matching gen_data.py. The key is read from GLM_API_KEY.
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

from picbench_qwen_zero import (
    EXAMPLE_A,
    EXAMPLE_Q,
    PIC,
    designs,
    extract_netlist,
    question_for,
)

from repo_paths import REPO

ROOT = REPO


def messages_for(name: str) -> list[dict[str, str]]:
    system_prompt = (PIC / "system_prompt.txt").read_text()
    question = question_for(name)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": EXAMPLE_Q},
        {"role": "assistant", "content": EXAMPLE_A},
        {"role": "user", "content": question},
        {"role": "user", "content": question},
    ]


def load_api_env() -> dict[str, str]:
    path = ROOT / "configs" / "api.env"
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def credentials_for(model: str) -> tuple[str, str, str]:
    """Base URL, key, and the model id the gateway expects.

    Output directories stay on --model. The gateway id can differ: the
    unfinished PICBench glm-5.3 files live in logs/picbench/glm-5.3, while
    the configured gateway calls that model glm-5.3-shouyun.
    """
    file_vals = load_api_env()
    if model == "deepseek-v4-pro":
        base = os.environ.get("DEEPSEEK_V4_PRO_BASE") or file_vals.get("DEEPSEEK_V4_PRO_BASE")
        key = os.environ.get("DEEPSEEK_V4_PRO_API_KEY") or file_vals.get("DEEPSEEK_V4_PRO_API_KEY")
        api_model = os.environ.get("DEEPSEEK_V4_PRO_MODEL") or file_vals.get("DEEPSEEK_V4_PRO_MODEL") or model
    else:
        base = os.environ.get("GLM_5_3_SHOUYUN_BASE") or file_vals.get("GLM_5_3_SHOUYUN_BASE")
        key = os.environ.get("GLM_5_3_SHOUYUN_API_KEY") or file_vals.get("GLM_5_3_SHOUYUN_API_KEY")
        api_model = os.environ.get("GLM_5_3_SHOUYUN_MODEL") or file_vals.get("GLM_5_3_SHOUYUN_MODEL") or model
    # Explicit process environment still wins, matching the older scripts.
    base = os.environ.get("GLM_BASE") or base
    key = os.environ.get("GLM_API_KEY") or key
    api_model = os.environ.get("GLM_API_MODEL") or api_model
    if not base or not key:
        raise SystemExit("missing API base or key in the environment or configs/api.env")
    return base, key, api_model


def _read_sse(resp, first_line: str | None = None) -> dict:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish = None
    usage = None

    def handle(line: str) -> bool:
        nonlocal finish, usage
        line = line.strip()
        if not line.startswith("data:"):
            return False
        data = line[5:].strip()
        if data == "[DONE]":
            return True
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return False
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            return False
        choice = choices[0]
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
        delta = choice.get("delta") or choice.get("message") or {}
        piece = delta.get("content")
        if piece:
            content_parts.append(piece)
        reasoning = delta.get("reasoning_content")
        if reasoning:
            reasoning_parts.append(reasoning)
        return False

    pending = [first_line] if first_line is not None else []
    while True:
        if pending:
            line = pending.pop(0)
        else:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace")
        if handle(line):
            break
    return {
        "choices": [
            {
                "finish_reason": finish,
                "message": {
                    "content": "".join(content_parts),
                    "reasoning_content": "".join(reasoning_parts) or None,
                },
            }
        ],
        "usage": usage,
    }


def _post_chat(base: str, key: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    # Socket timeout is idle time. Streaming keeps a long 8x8 completion
    # from looking idle to the gateway.
    with urllib.request.urlopen(req, timeout=2400) as resp:
        first = resp.readline()
        if not first:
            raise RuntimeError("empty response")
        text = first.decode("utf-8", errors="replace")
        if text.lstrip().startswith("data:"):
            return _read_sse(resp, first_line=text)
        return json.loads(text + resp.read().decode("utf-8", errors="replace"))


def chat(base: str, key: str, model: str, messages: list[dict[str, str]], max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 1,
        "max_tokens": max_tokens,
    }
    # Reasoning and the visible answer share this budget. A 32k cap was
    # consumed before <result>; allow a long completion to finish.
    # Stream so nginx does not 504 while an 8x8 design is still generating.
    streamed = dict(payload)
    streamed["stream"] = True
    streamed["stream_options"] = {"include_usage": True}
    try:
        return _post_chat(base, key, streamed)
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
        exc.read()
    streamed.pop("stream_options", None)
    try:
        return _post_chat(base, key, streamed)
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
        exc.read()
    return _post_chat(base, key, payload)


def reply_text(message: dict) -> str:
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if "<result>" in content:
        return content
    if "<result>" in reasoning:
        return reasoning
    return content or reasoning


def result_saved(dest: Path) -> bool:
    """A sample is done only when the visible result parses as a netlist."""
    try:
        payload = json.loads(dest.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    netlist = payload.get("netlist")
    if not netlist:
        return False
    try:
        obj = json.loads(netlist)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and "netlist" in obj and "models" in obj


def one(
    base: str,
    key: str,
    model: str,
    api_model: str,
    out: Path,
    name: str,
    sample: int,
    max_tokens: int,
) -> str:
    dest = out / name / f"s{sample}.json"
    if dest.exists() and result_saved(dest):
        return f"skip {name} s{sample}"
    last_error = None
    for attempt in range(4):
        try:
            data = chat(base, key, api_model, messages_for(name), max_tokens)
            choice = data["choices"][0]
            message = choice.get("message") or {}
            text = reply_text(message)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                json.dumps(
                    {
                        "design": name,
                        "sample": sample,
                        "model": model,
                        "api_model": api_model,
                        "max_tokens": max_tokens,
                        "finish_reason": choice.get("finish_reason"),
                        "usage": data.get("usage"),
                        "content": message.get("content"),
                        "reasoning_content": message.get("reasoning_content"),
                        "response": text,
                        "netlist": extract_netlist(text),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            usage = data.get("usage") or {}
            return (
                f"wrote {name} s{sample} chars {len(text)} "
                f"finish {choice.get('finish_reason')} "
                f"completion {usage.get('completion_tokens')}"
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:500]
            last_error = f"HTTP {exc.code} {body}"
            if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                break
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{name} s{sample}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    base, key, api_model = credentials_for(args.model)
    out = ROOT / "logs" / "picbench" / args.model
    jobs = [(name, sample) for name in designs() for sample in range(args.samples)]
    print(
        f"{args.model} api_model {api_model} jobs {len(jobs)} workers {args.workers}",
        flush=True,
    )
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(one, base, key, args.model, api_model, out, name, sample, args.max_tokens)
            for name, sample in jobs
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
