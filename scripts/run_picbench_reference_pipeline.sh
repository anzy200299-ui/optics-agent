#!/usr/bin/env bash
# Reference-JSON SFT, then one PICBench Without-EF decode of the loss-rule checkpoint.
# Same 985 questions, same Qwen3.5-9B base, same chat wrapper, same 2% / 8-epoch rule
# as the visible-reply run. The stopping rule is fixed here, before any PICBench score.
# Live checkpoints stay on local /tmp. Finished copies go under sft/picbench_reference.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

eval "$(python3 - "${SCRIPT_DIR}" << 'PY'
import sys
sys.path.insert(0, sys.argv[1])
from repo_paths import (
    MODEL_9B,
    PICBENCH_REFERENCE_JSONL,
    ROUTER_PYTHON,
    SAX_PYTHON,
    SFT_REFERENCE_STORE,
    SFT_REFERENCE_WORK,
    SWIFT_BIN,
)
def sh(value) -> str:
    return str(value).replace("'", "'\\''")
print(f"MODEL_9B='{sh(MODEL_9B)}'")
print(f"DATA_PATH='{sh(PICBENCH_REFERENCE_JSONL)}'")
print(f"ROUTER_PYTHON='{sh(ROUTER_PYTHON)}'")
print(f"SAX_PYTHON='{sh(SAX_PYTHON)}'")
print(f"OSS_ROOT='{sh(SFT_REFERENCE_STORE)}'")
print(f"LOCAL_ROOT='{sh(SFT_REFERENCE_WORK)}'")
print(f"SWIFT_PYTHON='{sh(SWIFT_BIN / 'python')}'")
PY
)"

STATUS="${OSS_ROOT}/status.txt"
LOG_DIR="${OSS_ROOT}/logs"
LENGTH_FILE="${ROOT}/data/sft/picbench_reference.max_length"
mkdir -p "${OSS_ROOT}" "${LOG_DIR}" "${LOCAL_ROOT}" "$(dirname "${LENGTH_FILE}")"

write_status() {
  python3 -c 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2] + "\n")' "${STATUS}" "$1"
  echo "$1"
}

write_status "build reference jsonl"
python3 -u "${ROOT}/scripts/build_picbench_reference_jsonl.py"

write_status "measure token length"
"${SWIFT_PYTHON}" -u "${ROOT}/scripts/measure_picbench_sft_length.py" \
  --jsonl "${DATA_PATH}" \
  --length-file "${LENGTH_FILE}"
MEASURED="$(tr -d '[:space:]' < "${LENGTH_FILE}")"
if [[ -z "${MEASURED}" ]]; then
  echo "ERROR: length file is empty" >&2
  exit 2
fi
if (( MEASURED > 5583 )); then
  MAX_LENGTH_FIXED="${MEASURED}"
else
  MAX_LENGTH_FIXED=5583
fi
write_status "length measured=${MEASURED} max_length=${MAX_LENGTH_FIXED}; start 2% rule, cap 8 epochs"

export DATA_PATH_OVERRIDE="${DATA_PATH}"
export OSS_ROOT_OVERRIDE="${OSS_ROOT}"
export LOCAL_ROOT_OVERRIDE="${LOCAL_ROOT}"
export LOCK_OVERRIDE="/tmp/picbench_reference_sft.converge.lock"
export MAX_LENGTH_FIXED
export RUN_LABEL="PICBench reference-JSON run"
bash "${ROOT}/scripts/run_until_converge.sh"

BEST_TAG="$(tr -d '[:space:]' < "${OSS_ROOT}/BEST")"
CKPT="$(find "${LOCAL_ROOT}/${BEST_TAG}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
if [[ -z "${CKPT}" || ! -f "${CKPT}/model.safetensors.index.json" ]]; then
  CKPT="${OSS_ROOT}/best"
fi
if [[ ! -f "${CKPT}/model.safetensors.index.json" ]]; then
  echo "ERROR: no checkpoint for ${BEST_TAG}" >&2
  exit 3
fi

EVAL_NAME="qwen35-9b-reference-sft-${BEST_TAG}"
EVAL_OUT="${ROOT}/logs/picbench/${EVAL_NAME}"
EVAL_LOG="${LOCAL_ROOT}/eval-logs"
mkdir -p "${EVAL_OUT}" "${EVAL_LOG}"
write_status "train done best=${BEST_TAG}; PICBench Without EF from ${CKPT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${LOCAL_ROOT}/hf"
export TMPDIR="${LOCAL_ROOT}/tmp"
mkdir -p "${HF_HOME}" "${TMPDIR}"

pids=()
for gpu in 0 1 2 3; do
  "${ROUTER_PYTHON}" -u "${ROOT}/scripts/picbench_qwen_zero.py" \
    --gpu "${gpu}" \
    --model "${CKPT}" \
    --out "${EVAL_OUT}" \
    > "${EVAL_LOG}/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done
if [[ "${fail}" -ne 0 ]]; then
  echo "ERROR: PICBench generation failed; see ${EVAL_LOG}" >&2
  exit 4
fi

"${SAX_PYTHON}" -u "${ROOT}/scripts/picbench_score.py" --model "${EVAL_NAME}" \
  | tee "${EVAL_LOG}/score.log"
cp -a "${EVAL_LOG}" "${ROOT}/logs/picbench/${EVAL_NAME}-logs"
python3 -c 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2] + "\n")' \
  "${ROOT}/logs/picbench/${EVAL_NAME}-WHICH.txt" \
  "checkpoint ${CKPT}"$'\n'"best ${BEST_TAG}"$'\n'"max_length ${MAX_LENGTH_FIXED}"$'\n'"measured ${MEASURED}"
write_status "scored ${EVAL_NAME} from ${BEST_TAG}"
echo "pipeline_done ${EVAL_NAME}"
