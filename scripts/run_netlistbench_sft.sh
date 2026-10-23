#!/bin/bash
# Official NetlistBench for the two epoch-8 students.
# Same protocol as the Qwen3.5-9B zero-shot run: greedy, thinking off, 4096 new tokens.
set -euo pipefail
cd /mnt/oss/yanzhengan/optics-agent
PY=/mnt/data/envs/ms-swift/bin/python
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/dev/shm/picbench_sft
mkdir -p "$TMPDIR" logs/netlistbench /tmp/netlistbench-sft

copy_ckpt() {
  local src="$1"
  local dst="$2"
  if [[ -f "$dst/model.safetensors.index.json" ]]; then
    echo "ckpt present $dst"
    return 0
  fi
  mkdir -p "$dst"
  echo "copy $src -> $dst"
  cp -a "$src/." "$dst/"
}

copy_ckpt sft/picbench_visible/epoch8 /dev/shm/sft-visible-epoch8
copy_ckpt sft/picbench_reference/epoch8 /dev/shm/sft-reference-epoch8

run_one() {
  local name="$1"
  local model="$2"
  local out="/mnt/oss/yanzhengan/optics-agent/logs/netlistbench/$name"
  local gpu pid fail
  local pids=()
  echo "generate $name from $model"
  for gpu in 0 1 2 3; do
    "$PY" -u scripts/netlistbench_run.py \
      --backend qwen --gpu "$gpu" --shards 4 --max-new-tokens 4096 \
      --model-path "$model" --out "$out" \
      > "/tmp/netlistbench-sft/${name}-gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  cp -f /tmp/netlistbench-sft/"${name}"-gpu*.log logs/netlistbench/ || true
  if [[ "$fail" -ne 0 ]]; then
    echo "generation failed $name" >&2
    exit 1
  fi
  python3 -u scripts/netlistbench_score.py --run "$out" --name "$name" --checkpoint "$model"
}

run_one qwen35-9b-visible-sft-epoch8 /dev/shm/sft-visible-epoch8
run_one qwen35-9b-reference-sft-epoch8 /dev/shm/sft-reference-epoch8
echo "netlistbench sft eval done"
