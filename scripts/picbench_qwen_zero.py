"""Zero-shot Qwen3.5-9B on PICBench, same prompt construction as gen_data.py.

No restrictions, no error feedback, n=5. Thinking is off so the reply stays in
the <analysis>/<result> format the benchmark parses. Temperature is 1, matching
the OpenAI default used by the published runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from repo_paths import MODEL_9B, REPO

ROOT = REPO
PIC = ROOT / "third_party" / "PICBench" / "PICBench"
CASES = ROOT / "third_party" / "PICBench" / "testcases"
OUT = ROOT / "logs" / "picbench" / "qwen35-9b"
MODEL = str(MODEL_9B)

EXAMPLE_Q = (
    "Create a Mach-Zehnder interferometer (MZI) with a single input and output, "
    "featuring a path length difference of L. Use the built-in multimode interferometer (MMI) component. \n"
    "Parameters:\nL = 10 microns \n\n"
    "Design the circuit and generate the corresponding JSON netlist based on the description. "
    "Try to understand the requirements and give reasoning steps in natural language to achieve it."
)
EXAMPLE_A = """<analysis>
To create a Mach-Zehnder Interferometer (MZI) with one input and one output using the built-in MMI component, we need the following steps:

1. **Components Needed**:
   - Two MMIs (acting as splitters and combiners).
   - Two waveguides (one with default length, one with an additional 10 microns to create the length difference).

2. **Instances**:
   - `mmi1`: First MMI splitter.
   - `waveguide_top`: Waveguide with length = default length + 10 microns.
   - `waveguide_bottom`: Waveguide with default length.
   - `mmi2`: Second MMI combiner.

3. **Connections**:
   - Connect output ports of `mmi1` to inputs of the two waveguides.
   - Connect outputs of the waveguides to inputs of `mmi2`.

4. **Ports**:
   - Define the input port connected to `mmi1`.
   - Define the output port from `mmi2`.

5. **Length Difference**:
   - Set `waveguide_top` length to default length + 10 microns.

<result>
{
  "netlist":{
    "instances": {
      "mmi1": "mmi",
      "waveguide_top": {"component": "waveguide", "settings": {"length": 20}},
      "waveguide_bottom": "waveguide",
      "mmi2": "mmi"
    },
    "connections": {
      "mmi1,O1": "waveguide_bottom,I1",
      "waveguide_bottom,O1": "mmi2,I1",
      "mmi1,O2": "waveguide_top,I1",
      "waveguide_top,O1": "mmi2,I1"
    },
    "ports": {
      "I1": "mmi1,I1",
      "O1": "mmi2,O1"
    }
  },
  "models":{
    "mmi": "mmi1x2",
    "waveguide": "straight"
  }
}"""

PROBLEM_PROMPT = (
    "Design the circuit and generate the corresponding JSON netlist based on the description. "
    "Try to understand the requirements and give reasoning steps in natural language to achieve it. "
    "In addition, try to give advice to avoid syntax error based on the restrictions."
)


def designs() -> list[str]:
    return sorted(path.name for path in CASES.iterdir() if path.is_dir())


def question_for(name: str) -> str:
    text = (CASES / name / f"{name}_problem_description.txt").read_text(encoding="utf-8")
    body = text.split("Problem Description-")[1]
    return body + "\n" + PROBLEM_PROMPT


def extract_netlist(response: str) -> str | None:
    if "<result>" not in response:
        return None
    netlist = response.split("<result>", 1)[1]
    netlist = netlist.replace("</result>", "")
    lines = netlist.split("\n")
    if lines and lines[0].strip() == "```json":
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    text = "\n".join(lines).strip()
    return text or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard", type=int, default=None, help="design index modulo nshards; default is --gpu")
    parser.add_argument("--nshards", type=int, default=4)
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--samples", type=int, default=5)
    # 4096 stopped inside <analysis> before <result>. The model context is
    # 262144. 32768 covers a long analysis plus the largest golden netlist.
    # Generation stops once </result> is emitted, so finished netlists do not
    # run out to this cap.
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    shard = args.gpu if args.shard is None else args.shard

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    system_prompt = (PIC / "system_prompt.txt").read_text()
    names = [name for index, name in enumerate(designs()) if index % args.nshards == shard]
    print(f"gpu {args.gpu} shard {shard}/{args.nshards} model {args.model}", flush=True)
    print(f"gpu {args.gpu} out {args.out}", flush=True)
    print(f"gpu {args.gpu} tasks {names}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    for name in names:
        question = question_for(name)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": EXAMPLE_Q},
            {"role": "assistant", "content": EXAMPLE_A},
            {"role": "user", "content": question},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        for sample in range(args.samples):
            dest = args.out / name / f"s{sample}.json"
            if dest.exists():
                print(f"skip {name} s{sample}", flush=True)
                continue
            torch.manual_seed(sample)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    stop_strings=["</result>"],
                    tokenizer=tokenizer,
                )
            new_tokens = output[0, inputs["input_ids"].shape[1] :]
            n_new = int(new_tokens.shape[0])
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                json.dumps(
                    {
                        "design": name,
                        "sample": sample,
                        "model": str(args.model),
                        "max_new_tokens": args.max_new_tokens,
                        "n_new_tokens": n_new,
                        "hit_cap": n_new >= args.max_new_tokens,
                        "response": text,
                        "netlist": extract_netlist(text),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print(
                f"wrote {name} s{sample} chars {len(text)} tokens {n_new} "
                f"hit_cap {n_new >= args.max_new_tokens}",
                flush=True,
            )


if __name__ == "__main__":
    main()
