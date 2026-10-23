#!/bin/bash
# Official NetlistBench for Reference-JSON epoch 8 only.
# Same protocol as the zero-shot 9B run and the visible-reply epoch-8 run:
# greedy, thinking off, 4096 new tokens, official prompts, one completion.
# The visible-reply arm already has logs/netlistbench/qwen35-9b-visible-sft-epoch8-score.json.
set -euo pipefail
cd /mnt/oss/yanzhengan/optics-agent
PY=/mnt/data/envs/ms-swift/bin/python
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/dev/shm/picbench_sft
mkdir -p "$TMPDIR" logs/netlistbench /tmp/netlistbench-sft

src=sft/picbench_reference/epoch8
dst=/dev/shm/sft-reference-epoch8
if [[ ! -f "$dst/model.safetensors.index.json" ]]; then
  mkdir -p "$dst"
  echo "copy $src -> $dst"
  cp -a "$src/." "$dst/"
else
  echo "ckpt present $dst"
fi

name=qwen35-9b-reference-sft-epoch8
out="/mnt/oss/yanzhengan/optics-agent/logs/netlistbench/$name"
pids=()
for gpu in 0 1 2 3; do
  "$PY" -u scripts/netlistbench_run.py \
    --backend qwen --gpu "$gpu" --shards 4 --max-new-tokens 4096 \
    --model-path "$dst" --out "$out" \
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
python3 -u scripts/netlistbench_score.py --run "$out" --name "$name" --checkpoint "$dst"
echo "reference netlistbench done"
