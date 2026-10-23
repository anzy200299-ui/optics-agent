"""Paths for this checkout.

Project files resolve from the repository that contains this script.
Teacher trajectories were moved beside the repo. The ms-swift environment
and the SAX/router interpreters stayed on /mnt/data: the copies under this
repo do not have working Python binaries. SFT loads the 9B weights from
this repo; /mnt/data is full.
Logs and the copied checkpoints stay under this repo. The trainer's
safetensors save cannot target ossfs (EOPNOTSUPP), so the live checkpoint
is written on the local root disk and then copied here.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PICBENCH_TRAJ_ROOT = Path("/mnt/oss/yanzhengan/picbench_glm53_trajectories")
PICBENCH_TRAJ_DIR = PICBENCH_TRAJ_ROOT / "raw"
PICBENCH_VISIBLE_JSONL = PICBENCH_TRAJ_ROOT / "picbench_glm53_visible.jsonl"
PICBENCH_REFERENCE_JSONL = PICBENCH_TRAJ_ROOT / "picbench_reference.jsonl"

SFT_REFERENCE_STORE = REPO / "sft" / "picbench_reference"
SFT_REFERENCE_WORK = Path("/tmp/picbench_reference_sft")

NETLIST_TRAJ_DIR = Path("/mnt/oss/yanzhengan/netlistbench_glm53_trajectories")

SFT_STORE = REPO / "sft" / "picbench_visible"
SFT_WORK = Path("/tmp/picbench_visible_sft")
SFT_CACHE = Path("/dev/shm/picbench_sft")

MODEL_9B = REPO / "models" / "Qwen3.5-9B"
CUDA_HOME = Path("/mnt/data/cuda-12.8/usr/local/cuda-12.8")
SWIFT_BIN = Path("/mnt/data/envs/ms-swift/bin")
SAX_PYTHON = Path("/mnt/data/optics-agent/.venv-sax/bin/python")
ROUTER_PYTHON = Path("/mnt/data/optics-agent/conda-envs/router/bin/python")
