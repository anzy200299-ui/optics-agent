#!/usr/bin/env bash
# Fresh Qwen3.5-9B PICBench visible-reply SFT from the base model.
# Stop when a new epoch's last 40-step mean loss is not 2% below the previous epoch.
# Checkpoints, logs, and caches stay under this repo. The lock stays on local /tmp.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

eval "$(python3 - "${SCRIPT_DIR}" << 'PY'
import sys
sys.path.insert(0, sys.argv[1])
from repo_paths import MODEL_9B, PICBENCH_VISIBLE_JSONL, SFT_STORE, SFT_WORK
def sh(value) -> str:
    return str(value).replace("'", "'\\''")
print(f"BASE_MODEL='{sh(MODEL_9B)}'")
print(f"DATA_PATH='{sh(PICBENCH_VISIBLE_JSONL)}'")
print(f"OSS_ROOT='{sh(SFT_STORE)}'")
print(f"LOCAL_ROOT='{sh(SFT_WORK)}'")
PY
)"
DATA_PATH="${DATA_PATH_OVERRIDE:-${DATA_PATH}}"
OSS_ROOT="${OSS_ROOT_OVERRIDE:-${OSS_ROOT}}"
LOCAL_ROOT="${LOCAL_ROOT_OVERRIDE:-${LOCAL_ROOT}}"
LOCK="${LOCK_OVERRIDE:-/tmp/picbench_visible_sft.converge.lock}"
MAX_LENGTH_FIXED="${MAX_LENGTH_FIXED:-5583}"

TRAIN_SH="${ROOT}/scripts/train_qwen35_picbench_visible_sft.sh"
COPY_PY="${ROOT}/scripts/copy_hf_checkpoint.py"
LOG_DIR="${OSS_ROOT}/logs"
MAX_EPOCHS="${MAX_EPOCHS:-8}"
REL_DROP="${REL_DROP:-0.02}"
LR="${LEARNING_RATE:-5e-6}"
mkdir -p "${OSS_ROOT}" "${LOG_DIR}" "${LOCAL_ROOT}"
exec 9>"${LOCK}"
flock -n 9 || { echo "ERROR: a convergence run already holds ${LOCK}" >&2; exit 1; }

if pgrep -f 'swift/cli/sft.py' >/dev/null 2>&1; then
  echo "ERROR: swift sft is already running" >&2
  exit 1
fi

export PATH="/mnt/data/envs/ms-swift/bin:${PATH}"
python - << PY
import json, torch
from pathlib import Path
data = Path("${DATA_PATH}")
n = sum(1 for _ in data.open())
row = json.loads(data.open().readline())
assert n == 985, n
assert "messages" in row, row.keys()
assert torch.cuda.device_count() >= 4, torch.cuda.device_count()
print(f"preflight rows={n} gpus={torch.cuda.device_count()}")
PY

python3 - "${OSS_ROOT}/RUN.txt" "${LR}" "${REL_DROP}" "${DATA_PATH}" "${RUN_LABEL:-PICBench visible-reply run}" << 'PY'
import sys
from pathlib import Path
dest, lr, drop, data_path, label = sys.argv[1:]
Path(dest).write_text(
    "Qwen3.5-9B full-parameter SFT, started from the base model.\n"
    f"Data: {data_path}\n"
    f"Learning rate: constant {lr} after warmup. Epoch 1 warmup ratio 0.05; later epochs warmup 0.\n"
    f"Stop when a new epoch last-40 mean loss is not at least {drop} below the previous epoch.\n"
    "Load the directory named in BEST, or the copy in best/.\n"
    f"This directory is the {label}.\n"
    "Launch scripts live in the optics-agent repo under scripts/.\n"
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

echo "repo ${ROOT}"
echo "oss ${OSS_ROOT}"
echo "work ${LOCAL_ROOT}"
echo "base ${BASE_MODEL}"
echo "lr ${LR} max_epochs ${MAX_EPOCHS} rel_drop ${REL_DROP}"
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
  else
    WARMUP="0"
  fi
  echo "=== start ${TAG} model=${MODEL} lr=${LR} warmup=${WARMUP} ==="
  REBUILD=0 \
  SKIP_MEASURE=1 \
  MAX_LENGTH="${MAX_LENGTH_FIXED}" \
  EPOCHS=1 \
  LEARNING_RATE="${LR}" \
  LR_SCHEDULER=constant \
  WARMUP_RATIO="${WARMUP}" \
  REPORT_TO=none \
  DATA_PATH="${DATA_PATH}" \
  MODEL_PATH="${MODEL}" \
  OUTPUT_DIR="${OUT}" \
  LOGGING_DIR="${LOG_DIR}/tensorboard-${TAG}" \
  MASTER_PORT=$((29540 + i)) \
  SEED=$((41 + i)) \
  SAVE_TOTAL_LIMIT=1 \
  TRAIN_LOG="${LOG_DIR}/${TAG}.log" \
  bash "${TRAIN_SH}"
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
echo "done best=${BEST_TAG} oss=${OSS_ROOT}/best"
