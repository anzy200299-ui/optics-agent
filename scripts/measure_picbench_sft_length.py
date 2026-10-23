"""Measure the trained token length of every kept PICBench trajectory.

The length is the ms-swift training encoding: last-round loss, thinking off,
and the empty <think> prefix masked. max_length must be at least this maximum,
because a shorter cap deletes trajectories under truncation_strategy=delete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import MODEL_9B, PICBENCH_VISIBLE_JSONL, REPO

MODEL = str(MODEL_9B)
DEFAULT_JSONL = PICBENCH_VISIBLE_JSONL
DEFAULT_LENGTH = REPO / "data" / "sft" / "picbench_glm53_visible.max_length"
BOUNDARY = "<|im_end|>\n<|im_start|>user\n"
LOSS_SCALE = "last_round+ignore_empty_think"


def expected_text(messages: list[dict]) -> str:
    system = messages[0]["content"].strip()
    example_q = messages[1]["content"].strip()
    example_a = messages[2]["content"].strip()
    left, sep, right = messages[3]["content"].partition(BOUNDARY)
    if not sep or left != right:
        raise ValueError("user turn does not contain two identical PICBench questions")
    answer = messages[4]["content"].strip()
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{example_q}<|im_end|>\n"
        f"<|im_start|>assistant\n{example_a}<|im_end|>\n"
        f"<|im_start|>user\n{left}<|im_end|>\n"
        f"<|im_start|>user\n{left}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>\n"
    )


def supervised_text(tokenizer, input_ids: list[int], labels: list[int]) -> str:
    kept = [token for token, label in zip(input_ids, labels) if label != -100]
    return tokenizer.decode(kept, skip_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--length-file", type=Path, default=DEFAULT_LENGTH)
    parser.add_argument("--check-rows", type=int, default=3)
    args = parser.parse_args()
    from swift.model import get_processor
    from swift.template import get_template

    rows = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("no training rows")
    processor = get_processor(MODEL)
    template = get_template(
        processor,
        max_length=262144,
        truncation_strategy="raise",
        loss_scale=LOSS_SCALE,
        is_binary_loss_scale=True,
        enable_thinking=False,
        add_non_thinking_prefix=True,
        padding_free=False,
        sequence_parallel_size=1,
        use_chat_template=True,
    )
    template.set_mode("train")
    tokenizer = processor.tokenizer
    lengths = []
    for index, row in enumerate(rows):
        messages = row["messages"]
        if messages[2].get("loss") is not False:
            raise SystemExit(f"row {index}: few-shot assistant must set loss false")
        encoded = template.encode({"messages": messages}, return_length=True)
        length = max(encoded["lengths"])
        if length != len(encoded["input_ids"]):
            raise SystemExit(f"row {index}: length {length} != input_ids {len(encoded['input_ids'])}")
        lengths.append(length)
        if index < args.check_rows:
            decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
            target = expected_text(messages)
            if decoded != target:
                print("template mismatch", flush=True)
                print("--- decoded ---", flush=True)
                print(decoded[:1500], flush=True)
                print("--- expected ---", flush=True)
                print(target[:1500], flush=True)
                raise SystemExit(2)
            answer = messages[4]["content"].strip()
            learned = supervised_text(tokenizer, encoded["input_ids"], encoded["labels"])
            if answer not in learned:
                raise SystemExit(f"row {index}: supervised tokens do not contain the visible answer")
            if "<|im_start|>user" in learned:
                raise SystemExit(f"row {index}: loss includes the prompt")
            if not learned.startswith(answer):
                raise SystemExit(f"row {index}: supervised text does not start with the visible answer")
        if (index + 1) % 200 == 0 or index + 1 == len(rows):
            print(f"encoded {index + 1}/{len(rows)}", flush=True)
    longest = max(lengths)
    args.length_file.write_text(f"{longest}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "n": len(lengths),
                "min": min(lengths),
                "max": longest,
                "length_file": str(args.length_file),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
