#!/usr/bin/env bash
# Train the follow-up students after the earlier-epoch eval has released the GPUs.
# Each run uses the published 2% loss rule. Only the loss-selected checkpoint is
# scored. The 10/120 and 2/24 bars are not changed, and these rows are not
# merged into the published 985.
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

if [[ ! -f logs/picbench/earlier_epochs.json ]]; then
  echo "ERROR: earlier-epoch scores are not on disk; not training" >&2
  exit 1
fi

eval_best() {
  local name="$1"
  local ckpt="$2"
  local out="logs/picbench/${name}"
  if [[ -f "${out}/summary.json" ]]; then
    echo "score exists ${name}"
    return 0
  fi
  local dst="/dev/shm/sft-eval-epoch"
  rm -rf "$dst"
  mkdir -p "$dst"
  cp -a "${ckpt}/." "${dst}/"
  local pids=()
  local gpu
  for gpu in 0 1 2 3; do
    "$PY" -u scripts/picbench_qwen_zero.py \
      --gpu "$gpu" --nshards 4 --samples 5 --max-new-tokens 32768 \
      --model "$dst" --out "$ROOT/$out" \
      > "/tmp/picbench-epochs/${name}-gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  local fail=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  rm -rf "$dst"
  if [[ "$fail" -ne 0 ]]; then
    echo "ERROR: generation failed ${name}" >&2
    exit 1
  fi
  local nfiles
  nfiles="$(find "$out" -name 's*.json' | wc -l)"
  if [[ "$nfiles" -ne 120 ]]; then
    echo "ERROR: ${name} has ${nfiles} samples" >&2
    exit 1
  fi
  "$SCORE" -u scripts/picbench_score.py --model "$name"
}

train_one() {
  local name="$1"
  local data="$2"
  local expect="$3"
  local skip="$4"
  local label="$5"
  local oss="${ROOT}/sft/ablation/${name}"
  local local_root="/tmp/ablation-${name}"
  DATA_PATH="$data" \
  OSS_ROOT="$oss" \
  LOCAL_ROOT="$local_root" \
  EXPECT_N="$expect" \
  SKIP_MEASURE="$skip" \
  RUN_LABEL="$label" \
  LOCK_OVERRIDE="/tmp/picbench_ablation.converge.lock" \
  bash "${ROOT}/scripts/run_ablation_converge.sh"
  local best
  best="$(tr -d '[:space:]' < "${oss}/BEST")"
  eval_best "ablation-${name}" "${oss}/best"
  echo "${name} best ${best}" >> logs/picbench/ablation_progress.txt
}

# Visible-reply subsets. skip length measurement; they are rows of the 985.
train_one exact901 data/ablation/exact901.visible.jsonl 901 1 "901 unrepaired teacher matches"
train_one matched901 data/ablation/matched901.visible.jsonl 901 1 "901-row draw from the 985, seed 20261023"

python3 - << 'PY' > /tmp/ablation-queue.txt
import json
from pathlib import Path
manifest = json.loads(Path("data/ablation/manifest.json").read_text())
# exact901 and matched901 are launched above; queue the locus pairs from the manifest.
for job in manifest["jobs"]:
    name = job["name"]
    if name in {"exact901", "matched901"}:
        continue
    print(f"{name}\t{job['visible_jsonl']}\t{job['n']}\t1\tvisible subset {name}")
PY

while IFS=$'\t' read -r name data expect skip label; do
  train_one "$name" "$data" "$expect" "$skip" "$label"
done < /tmp/ablation-queue.txt

reweight_n="$(wc -l < data/ablation/reweight.reference.jsonl)"
random_n="$(wc -l < data/ablation/random_legal.reference.jsonl)"
noname_n="$(wc -l < data/ablation/noname.reference.jsonl)"
train_one reweight data/ablation/reweight.reference.jsonl "$reweight_n" 0 "reference JSON, re-audit weights"
train_one random-legal data/ablation/random_legal.reference.jsonl "$random_n" 0 "reference JSON, uniform legal circuits"
train_one noname data/ablation/noname.reference.jsonl "$noname_n" 0 "reference JSON, prompts without SAX model ids"

echo "follow-up training done"
