#!/usr/bin/env bash
# Score saved epochs 1-7 of both students on PICBench Without EF.
# Epoch 8 is already scored. BEST stays epoch 8, chosen by the training-loss
# rule before these scores exist. This script does not read the scores to
# pick a checkpoint, and it does not move the 10/120 or 2/24 bars.
set -euo pipefail

ROOT="/mnt/oss/yanzhengan/optics-agent"
PY="/mnt/data/optics-agent/conda-envs/router/bin/python"
SCORE="/mnt/data/optics-agent/.venv-sax/bin/python"
cd "$ROOT"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu
export TMPDIR=/dev/shm/picbench_sft
mkdir -p "$TMPDIR" /tmp/picbench-epochs logs/picbench

# The finished Reference epoch-8 copy is no longer needed for generation.
rm -rf /dev/shm/sft-reference-epoch8

note() {
  printf '%s %s\n' "$(date '+%F %T')" "$*"
}

score_one() {
  local name="$1"
  local summary="logs/picbench/${name}/summary.json"
  if [[ -f "$summary" ]]; then
    note "score exists ${name}"
    return 0
  fi
  local nfiles
  nfiles="$(find "logs/picbench/${name}" -name 's*.json' | wc -l)"
  if [[ "$nfiles" -ne 120 ]]; then
    echo "ERROR: ${name} has ${nfiles} sample files, expected 120" >&2
    exit 1
  fi
  "$SCORE" -u scripts/picbench_score.py --model "$name"
}

for arm in visible reference; do
  for epoch in 1 2 3 4 5 6 7; do
    name="qwen35-9b-${arm}-sft-epoch${epoch}"
    out="logs/picbench/${name}"
    if [[ -f "${out}/summary.json" ]]; then
      note "already scored ${name}"
      continue
    fi
    src="sft/picbench_${arm}/epoch${epoch}"
    if [[ ! -f "${src}/model.safetensors.index.json" ]]; then
      echo "ERROR: missing checkpoint ${src}" >&2
      exit 1
    fi
    dst="/dev/shm/sft-eval-epoch"
    rm -rf "$dst"
    mkdir -p "$dst"
    note "copy ${src} -> ${dst}"
    cp -a "${src}/." "${dst}/"
    pids=()
    for gpu in 0 1 2 3; do
      "$PY" -u scripts/picbench_qwen_zero.py \
        --gpu "$gpu" --nshards 4 --samples 5 --max-new-tokens 32768 \
        --model "$dst" --out "$ROOT/$out" \
        > "/tmp/picbench-epochs/${name}-gpu${gpu}.log" 2>&1 &
      pids+=("$!")
    done
    fail=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        fail=1
      fi
    done
    cp -f /tmp/picbench-epochs/"${name}"-gpu*.log "logs/picbench/" || true
    if [[ "$fail" -ne 0 ]]; then
      echo "ERROR: generation failed ${name}" >&2
      exit 1
    fi
    score_one "$name"
    rm -rf "$dst"
    note "finished ${name}"
  done
done

"$PY" - << 'PY'
import json
from pathlib import Path
root = Path("/mnt/oss/yanzhengan/optics-agent/logs/picbench")
rows = []
for arm in ("visible", "reference"):
    for epoch in range(1, 9):
        path = root / f"qwen35-9b-{arm}-sft-epoch{epoch}" / "summary.json"
        if not path.exists():
            rows.append({"arm": arm, "epoch": epoch, "status": "missing"})
            continue
        summary = json.loads(path.read_text())
        func = sum(row["func"] for row in summary["designs"])
        syntax_only = sum(row["syntax"] for row in summary["designs"])
        rows.append({
            "arm": arm,
            "epoch": epoch,
            "n_designs": summary["n_designs"],
            "functional_matches": func,
            "simulable": syntax_only + func,
            "syntax_pass1": summary["syntax_pass1"],
            "func_pass1": summary["func_pass1"],
            "syntax_pass5": summary["syntax_pass5"],
            "func_pass5": summary["func_pass5"],
            "selected": epoch == 8,
        })
out = {
    "protocol": "PICBench Without EF, temperature 1, n=5, thinking off, max_new_tokens 32768, same prompt as epoch 8",
    "selection": "BEST remains epoch 8 on both arms. It was chosen by the 2% training-loss rule and the epoch cap. These scores are not a selection.",
    "bars": "10/120 and 2/24 are unchanged.",
    "rows": rows,
}
dest = root / "earlier_epochs.json"
dest.write_text(json.dumps(out, indent=2) + "\n")
print("wrote", dest)
PY
note "earlier-epoch eval done"
