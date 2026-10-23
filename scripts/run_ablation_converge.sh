#!/usr/bin/env bash
# One ablation student. Same optimizer, length cap, and 2% loss rule as the
# published runs. The epoch is not chosen from the 24 test problems.
# DATA_PATH, OSS_ROOT, LOCAL_ROOT, EXPECT_N, and SKIP_MEASURE are required.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${DATA_PATH:?}"
: "${OSS_ROOT:?}"
: "${LOCAL_ROOT:?}"
: "${EXPECT_N:?}"
SKIP_MEASURE="${SKIP_MEASURE:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-8}"
REL_DROP="${REL_DROP:-0.02}"
LR="${LEARNING_RATE:-5e-6}"
LOCK="${LOCK_OVERRIDE:-/tmp/picbench_ablation.converge.lock}"
RUN_LABEL="${RUN_LABEL:-ablation}"
MAX_LENGTH_FIXED="${MAX_LENGTH_FIXED:-5583}"

TRAIN_SH="${ROOT}/scripts/train_qwen35_picbench_visible_sft.sh"
COPY_PY="${ROOT}/scripts/copy_hf_checkpoint.py"
LOG_DIR="${OSS_ROOT}/logs"
BASE_MODEL="${ROOT}/models/Qwen3.5-9B"

mkdir -p "${OSS_ROOT}" "${LOG_DIR}" "${LOCAL_ROOT}"
exec 9>"${LOCK}"
flock -n 9 || { echo "ERROR: ablation lock held: ${LOCK}" >&2; exit 1; }

if [[ -f "${OSS_ROOT}/DONE" ]]; then
  echo "already done ${OSS_ROOT}"
  exit 0
fi
if pgrep -f 'swift/cli/sft.py' >/dev/null 2>&1; then
  echo "ERROR: swift sft is already running" >&2
  exit 1
fi

export PATH="/mnt/data/envs/ms-swift/bin:${PATH}"
python - << PY
import json, torch
from pathlib import Path
data = Path("${DATA_PATH}")
n = sum(1 for line in data.open() if line.strip())
row = json.loads(data.open().readline())
assert n == int("${EXPECT_N}"), (n, "${EXPECT_N}")
assert "messages" in row, row.keys()
assert torch.cuda.device_count() >= 4, torch.cuda.device_count()
print(f"preflight rows={n} gpus={torch.cuda.device_count()} label=${RUN_LABEL}")
PY

python3 - "${OSS_ROOT}/RUN.txt" "${LR}" "${REL_DROP}" "${DATA_PATH}" "${RUN_LABEL}" "${EXPECT_N}" << 'PY'
import sys
from pathlib import Path
dest, lr, drop, data_path, label, expect = sys.argv[1:]
Path(dest).write_text(
    "Qwen3.5-9B full-parameter SFT, started from the base model.\n"
    f"Data: {data_path}\n"
    f"Rows: {expect}\n"
    f"Learning rate: constant {lr} after warmup. Epoch 1 warmup ratio 0.05; later epochs warmup 0.\n"
    f"Stop when a new epoch last-40 mean loss is not at least {drop} below the previous epoch.\n"
    "The epoch is not chosen from the 24 PICBench problems.\n"
    f"This directory is the {label}.\n"
)
print("wrote", dest)
PY

mean_last40() {
  python3 - "$1" << 'PY'
import json, sys
path = sys.argv[1]
losses = []
for line in open(path):
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)
    if "loss" not in obj:
        continue
    losses.append(float(obj["loss"]))
if len(losses) < 40:
    raise SystemExit(f"only {len(losses)} loss rows in {path}")
print(f"{sum(losses[-40:]) / 40:.8f}")
PY
}

echo "oss ${OSS_ROOT}"
echo "work ${LOCAL_ROOT}"
echo "data ${DATA_PATH}"
echo "rows ${EXPECT_N} lr ${LR} max_epochs ${MAX_EPOCHS} rel_drop ${REL_DROP}"
python3 -c 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("")' "${OSS_ROOT}/loss_trace.txt"

PREV_MEAN=""
MODEL="${BASE_MODEL}"
BEST_TAG=""
BEST_LOCAL=""

for i in $(seq 1 "${MAX_EPOCHS}"); do
  TAG="epoch${i}"
  OUT="${LOCAL_ROOT}/${TAG}"
  if [[ "${i}" -eq 1 ]]; then
    WARMUP="0.05"
    MEASURE_FLAG="${SKIP_MEASURE}"
  else
    WARMUP="0"
    MEASURE_FLAG=1
  fi
  echo "=== start ${TAG} model=${MODEL} lr=${LR} warmup=${WARMUP} ==="
  if [[ "${MEASURE_FLAG}" == 1 ]]; then
    REBUILD=0 SKIP_MEASURE=1 MAX_LENGTH="${MAX_LENGTH_FIXED}" \
      EPOCHS=1 LEARNING_RATE="${LR}" LR_SCHEDULER=constant WARMUP_RATIO="${WARMUP}" \
      REPORT_TO=none DATA_PATH="${DATA_PATH}" MODEL_PATH="${MODEL}" \
      OUTPUT_DIR="${OUT}" LOGGING_DIR="${LOG_DIR}/tensorboard-${TAG}" \
      MASTER_PORT=$((29720 + i)) SEED=$((41 + i)) SAVE_TOTAL_LIMIT=1 \
      TRAIN_LOG="${LOG_DIR}/${TAG}.log" LENGTH_FILE="${OSS_ROOT}/max_length" \
      bash "${TRAIN_SH}"
  else
    REBUILD=0 SKIP_MEASURE=0 \
      EPOCHS=1 LEARNING_RATE="${LR}" LR_SCHEDULER=constant WARMUP_RATIO="${WARMUP}" \
      REPORT_TO=none DATA_PATH="${DATA_PATH}" MODEL_PATH="${MODEL}" \
      OUTPUT_DIR="${OUT}" LOGGING_DIR="${LOG_DIR}/tensorboard-${TAG}" \
      MASTER_PORT=$((29720 + i)) SEED=$((41 + i)) SAVE_TOTAL_LIMIT=1 \
      TRAIN_LOG="${LOG_DIR}/${TAG}.log" LENGTH_FILE="${OSS_ROOT}/max_length" \
      bash "${TRAIN_SH}"
    MAX_LENGTH_FIXED="$(tr -d '[:space:]' < "${OSS_ROOT}/max_length")"
    echo "measured max_length ${MAX_LENGTH_FIXED}"
  fi
  CKPT="$(find "${OUT}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  if [[ -z "${CKPT}" || ! -f "${CKPT}/model.safetensors.index.json" ]]; then
    echo "ERROR: ${TAG} finished without a model checkpoint" >&2
    exit 3
  fi
  NEW_MEAN="$(mean_last40 "${OUT}/logging.jsonl")"
  echo "${TAG} last40 ${NEW_MEAN} previous ${PREV_MEAN:-none}"
  python3 -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); p.write_text((p.read_text() if p.exists() else "") + sys.argv[2] + "\n")' "${OSS_ROOT}/loss_trace.txt" "${TAG} ${NEW_MEAN}"
  python3 "${COPY_PY}" "${CKPT}" "${OSS_ROOT}/${TAG}"
  if [[ -z "${PREV_MEAN}" ]]; then
    PREV_MEAN="${NEW_MEAN}"
    MODEL="${CKPT}"
    BEST_TAG="${TAG}"
    BEST_LOCAL="${CKPT}"
    python3 -c 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2] + "\n")' "${OSS_ROOT}/BEST" "${BEST_TAG}"
    echo "keep ${TAG}"
    continue
  fi
  LIMIT="$(python3 - "${PREV_MEAN}" "${REL_DROP}" << 'PY'
import sys
prev, drop = float(sys.argv[1]), float(sys.argv[2])
print(f"{prev * (1.0 - drop):.8f}")
PY
)"
  echo "continue if ${NEW_MEAN} <= ${LIMIT}"
  if python3 - "${NEW_MEAN}" "${LIMIT}" << 'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)
PY
  then
    PREV_MEAN="${NEW_MEAN}"
    MODEL="${CKPT}"
    BEST_TAG="${TAG}"
    BEST_LOCAL="${CKPT}"
    python3 -c 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2] + "\n")' "${OSS_ROOT}/BEST" "${BEST_TAG}"
    echo "keep ${TAG}"
    if [[ "${i}" -eq "${MAX_EPOCHS}" ]]; then
      echo "stop: reached the epoch cap (${MAX_EPOCHS}) while loss was still falling"
    fi
  else
    echo "stop: ${TAG} last-40 mean did not fall by ${REL_DROP}; keeping ${BEST_TAG}"
    break
  fi
done

python3 "${COPY_PY}" "${BEST_LOCAL}" "${OSS_ROOT}/best"
python3 -c 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2] + "\n")' "${OSS_ROOT}/BEST" "${BEST_TAG}"
rm -rf "${LOCAL_ROOT}"
date -u '+%F %T' > "${OSS_ROOT}/DONE"
echo "done best=${BEST_TAG} oss=${OSS_ROOT}/best"
