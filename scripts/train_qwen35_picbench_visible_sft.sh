#!/usr/bin/env bash
# Visible-reply SFT of Qwen3.5-9B on SAX-matched glm-5.3-shouyun PICBench trajectories.
# Four A100-80GB, full parameters, ZeRO-3. max_length covers every kept trajectory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

eval "$(python3 - "${SCRIPT_DIR}" << 'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from repo_paths import (
    CUDA_HOME,
    MODEL_9B,
    PICBENCH_VISIBLE_JSONL,
    SAX_PYTHON,
    SFT_CACHE,
    SFT_WORK,
    SWIFT_BIN,
)
def sh(value: Path) -> str:
    return str(value).replace("'", "'\\''")
print(f"MODEL_PATH_DEFAULT='{sh(MODEL_9B)}'")
print(f"DATA_PATH_DEFAULT='{sh(PICBENCH_VISIBLE_JSONL)}'")
print(f"SAX_PYTHON='{sh(SAX_PYTHON)}'")
print(f"SFT_CACHE='{sh(SFT_CACHE)}'")
print(f"SFT_WORK='{sh(SFT_WORK)}'")
print(f"SWIFT_BIN='{sh(SWIFT_BIN)}'")
print(f"CUDA_HOME_DEFAULT='{sh(CUDA_HOME)}'")
PY
)"

export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_DEFAULT}}"
export PATH="${CUDA_HOME}/bin:${SWIFT_BIN}:${PATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export JAX_PLATFORMS=cpu
# ossfs rejects the small files compilers and Triton write, reporting ENOSPC.
# Weights and logs stay under the repo. These caches stay on local tmpfs.
LOCAL_CACHE="/dev/shm/picbench_sft"
export MODELSCOPE_CACHE="${LOCAL_CACHE}/modelscope"
export HF_HOME="${LOCAL_CACHE}/hf"
export HF_DATASETS_CACHE="${LOCAL_CACHE}/datasets"
export TRANSFORMERS_CACHE="${LOCAL_CACHE}/hf"
export XDG_CACHE_HOME="${LOCAL_CACHE}/xdg"
export TRITON_CACHE_DIR="${LOCAL_CACHE}/triton"
export TMPDIR="${LOCAL_CACHE}/tmp"
mkdir -p "${MODELSCOPE_CACHE}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}" "${TMPDIR}"

MODEL_PATH="${MODEL_PATH:-${MODEL_PATH_DEFAULT}}"
DATA_PATH="${DATA_PATH:-${DATA_PATH_DEFAULT}}"
REPORT_PATH="${REPORT_PATH:-${ROOT}/logs/sft/picbench_glm53_visible_filter.json}"
LENGTH_FILE="${LENGTH_FILE:-${ROOT}/data/sft/picbench_glm53_visible.max_length}"
OUTPUT_DIR="${OUTPUT_DIR:-${SFT_WORK}/default}"
LOGGING_DIR="${LOGGING_DIR:-${OUTPUT_DIR}/tensorboard}"

EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
REPORT_TO="${REPORT_TO:-none}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SEED="${SEED:-42}"
FILTER_WORKERS="${FILTER_WORKERS:-8}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-29521}"

if [[ "${REBUILD:-0}" == 1 ]]; then
  "${SAX_PYTHON}" -u "${ROOT}/scripts/filter_picbench_teacher.py" --workers "${FILTER_WORKERS}"
fi

if [[ "${SKIP_MEASURE:-0}" == 1 ]]; then
  if [[ -z "${MAX_LENGTH:-}" ]]; then
    echo "ERROR: SKIP_MEASURE=1 requires MAX_LENGTH." >&2
    exit 2
  fi
  MEASURED="${MAX_LENGTH}"
  echo "skip length measure; MAX_LENGTH=${MAX_LENGTH}"
else
  "${SWIFT_BIN}/python" -u "${ROOT}/scripts/measure_picbench_sft_length.py" \
    --jsonl "${DATA_PATH}" \
    --length-file "${LENGTH_FILE}"
  MEASURED="$(tr -d '[:space:]' < "${LENGTH_FILE}")"
  if [[ -z "${MAX_LENGTH:-}" ]]; then
    MAX_LENGTH="${MEASURED}"
  fi
fi
if (( MAX_LENGTH < MEASURED )); then
  echo "ERROR: MAX_LENGTH=${MAX_LENGTH} is shorter than the longest kept trajectory (${MEASURED})." >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_IDS[@]}" -ne "${NPROC_PER_NODE}" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES has ${#GPU_IDS[@]} ids but NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${LOGGING_DIR}"
TRAIN_LOG="${TRAIN_LOG:-${OUTPUT_DIR}/train.log}"
echo "repo ${ROOT}"
echo "max_length ${MAX_LENGTH} (longest kept trajectory ${MEASURED})"
echo "model ${MODEL_PATH}"
echo "data ${DATA_PATH}"
echo "report ${REPORT_PATH}"
echo "output ${OUTPUT_DIR}"
echo "epochs ${EPOCHS} lr ${LEARNING_RATE} scheduler ${LR_SCHEDULER} warmup ${WARMUP_RATIO}"

swift sft \
  --model "${MODEL_PATH}" \
  --check_model false \
  --tuner_type full \
  --freeze_llm false \
  --freeze_vit true \
  --freeze_aligner true \
  --dataset "${DATA_PATH}" \
  --split_dataset_ratio 0 \
  --dataset_shuffle true \
  --train_dataloader_shuffle true \
  --group_by_length true \
  --loss_scale last_round \
  --use_liger_kernel true \
  --is_binary_loss_scale true \
  --use_chat_template true \
  --enable_thinking false \
  --add_non_thinking_prefix true \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --use_logits_to_keep false \
  --padding_free false \
  --packing false \
  --sequence_parallel_size 1 \
  --deepspeed zero3 \
  --gradient_checkpointing true \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --max_length "${MAX_LENGTH}" \
  --truncation_strategy delete \
  --learning_rate "${LEARNING_RATE}" \
  --optim adamw_torch \
  --lr_scheduler_type "${LR_SCHEDULER}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_train_epochs "${EPOCHS}" \
  --eval_strategy no \
  --save_strategy epoch \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --save_only_model true \
  --logging_steps "${LOGGING_STEPS}" \
  --logging_first_step true \
  --logging_dir "${LOGGING_DIR}" \
  --report_to "${REPORT_TO}" \
  --dataset_num_proc 4 \
  --dataloader_num_workers 2 \
  --seed "${SEED}" \
  --data_seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  --add_version false \
  2>&1 | tee -a "${TRAIN_LOG}"

echo "Training finished. max_length=${MAX_LENGTH}"
