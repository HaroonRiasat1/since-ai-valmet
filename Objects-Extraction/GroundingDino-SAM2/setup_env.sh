#!/usr/bin/env bash
# One-shot environment setup for Apple Silicon + Metal.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uv &>/dev/null; then
  echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

uv venv .venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install --upgrade pip
uv pip install -r requirements.txt

# Clone Grounded-SAM-2 (SAM2 + helpers) if missing
if [[ ! -d third_party/Grounded-SAM-2/sam2 ]]; then
  mkdir -p third_party
  git clone --depth 1 https://github.com/IDEA-Research/Grounded-SAM-2 third_party/Grounded-SAM-2
fi

export SAM2_BUILD_CUDA=0
export SAM2_BUILD_ALLOW_ERRORS=1
uv pip install --no-build-isolation -e third_party/Grounded-SAM-2

mkdir -p checkpoints
SAM_CKPT="checkpoints/sam2.1_hiera_large.pt"
if [[ ! -f "$SAM_CKPT" ]]; then
  echo "Downloading SAM 2.1 Hiera Large checkpoint..."
  curl -L --fail -o "$SAM_CKPT" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
fi

# Prefetch HF models (Grounding DINO) into cache
python - <<'PY'
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
pid = "IDEA-Research/grounding-dino-base"
AutoProcessor.from_pretrained(pid)
AutoModelForZeroShotObjectDetection.from_pretrained(pid)
print("HF prefetch ok:", pid)
PY

uv pip freeze --exclude-editable > requirements.lock.txt

python - <<'PY'
import torch
print("torch", torch.__version__, "mps", torch.backends.mps.is_available())
import mlx3d
print("mlx3d", getattr(mlx3d, "__version__", "?"))
from sam2.build_sam import build_sam2
print("sam2 build_sam2 import ok")
PY

echo "Done. Activate with: source .venv/bin/activate"
