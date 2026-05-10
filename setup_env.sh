#!/bin/bash
# FreeOrbit4D environment setup script
# Default (validated) stack: Python 3.10 CPython + PyTorch 2.4.0 + CUDA 12.1
#
# Usage:
#   bash setup_env.sh
#   bash setup_env.sh --recreate   # Remove an existing freeorbit4d env and reinstall
#
# Activate after installation:
#   conda activate freeorbit4d
#
# Common overrides (export before running):
#   ENV_NAME=my_env                # Use a different conda environment name
#   PYTHON_VERSION=3.10            # Major.minor (default) or full patch (e.g. 3.10.20)
#   MAX_JOBS=4                     # Parallel jobs for the PyTorch3D build (auto by default)

set -euo pipefail

ENV_NAME="${ENV_NAME:-freeorbit4d}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.4.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.0}"
CUDA_TAG="${CUDA_TAG:-cu121}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${CUDA_TAG}}"
XFORMERS_VERSION="${XFORMERS_VERSION:-0.0.27.post1}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.57.1}"
SAM2_REF="${SAM2_REF:-aa9b8722d0585b661ded4b3dff1bd103540554ae}"
DATAPIPELINES_REF="${DATAPIPELINES_REF:-8bce77d147033b3a5285b6d45ee85f33866964fc}"
CLIP_REF="${CLIP_REF:-d05afc436d78f1c48dc0dbf8e5980a9d471f35f6}"
PYTORCH3D_REF="${PYTORCH3D_REF:-75ebeeaea0908c5527e7b1e305fbc7681382db47}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECREATE=0

# Only cu121 is validated. To attempt a different CUDA wheel at your own risk,
# export CUDA_TAG together with TORCH_CUDA_VERSION and CUDA_TOOLKIT_VERSION so
# the conda cuda-toolkit label matches the wheel you ask pip for.
if [[ "${CUDA_TAG}" != "cu121" ]]; then
    if [[ -z "${TORCH_CUDA_VERSION:-}" || -z "${CUDA_TOOLKIT_VERSION:-}" ]]; then
        echo "Unsupported CUDA_TAG='${CUDA_TAG}'. FreeOrbit4D's release setup is validated only on cu121."
        echo "If you really want to try another CUDA wheel, also export both:"
        echo "  TORCH_CUDA_VERSION   (e.g. 12.4)"
        echo "  CUDA_TOOLKIT_VERSION (e.g. 12.4.1, must exist as nvidia/label/cuda-<value>)"
        exit 1
    fi
    echo "Warning: CUDA_TAG='${CUDA_TAG}' is not validated. Proceeding because TORCH_CUDA_VERSION and CUDA_TOOLKIT_VERSION are set."
fi
TORCH_CUDA_VERSION="${TORCH_CUDA_VERSION:-12.1}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12.1.0}"

# PyTorch3D's build can spike to ~3 GB resident per worker. Cap MAX_JOBS so a
# small box does not OOM-kill the compile partway through.
if [[ -z "${MAX_JOBS:-}" ]]; then
    _cpu_jobs="$(nproc 2>/dev/null || echo 4)"
    if (( _cpu_jobs > 8 )); then
        MAX_JOBS=8
    else
        MAX_JOBS="${_cpu_jobs}"
    fi
fi
export MAX_JOBS

if [[ "${1:-}" == "--recreate" ]]; then
    RECREATE=1
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '1,16p' "$0"
    exit 0
elif [[ $# -gt 0 ]]; then
    echo "Unknown argument: $1"
    echo "Usage: bash setup_env.sh [--recreate]"
    exit 1
fi

ensure_cpython() {
    python - <<'PY'
import platform
assert platform.python_implementation() == "CPython", platform.python_implementation()
print(f"  Python:     {platform.python_version()} ({platform.python_implementation()})")
PY
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing command: $1"
        exit 1
    fi
}

resolve_cuda_home() {
    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        :
    elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
        CUDA_HOME="${CONDA_PREFIX}"
    elif [[ -x "/usr/local/cuda-${TORCH_CUDA_VERSION}/bin/nvcc" ]]; then
        CUDA_HOME="/usr/local/cuda-${TORCH_CUDA_VERSION}"
    elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
        CUDA_HOME="/usr/local/cuda"
    elif command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    else
        echo "nvcc was not found. PyTorch3D needs the CUDA toolkit for compilation."
        echo "The installer normally provides nvcc via conda cuda-toolkit ${CUDA_TOOLKIT_VERSION}."
        echo "If you manage CUDA yourself, set CUDA_HOME to a directory containing bin/nvcc."
        exit 1
    fi
    export CUDA_HOME
    echo "  CUDA_HOME: ${CUDA_HOME}"
    "${CUDA_HOME}/bin/nvcc" --version | sed -n '1,4p'
}

require_command conda
require_command git
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  GPU: /'
else
    echo "Warning: nvidia-smi was not found; installation can continue, but the pipeline needs an NVIDIA GPU."
fi

echo "============================================================"
echo " FreeOrbit4D environment setup"
echo " Python ${PYTHON_VERSION} | PyTorch ${PYTORCH_VERSION} | CUDA wheel ${CUDA_TAG}"
echo " CUDA toolkit for builds: ${CUDA_TOOLKIT_VERSION}"
echo "============================================================"

echo -e "\n[1/8] Creating conda environment ${ENV_NAME} ..."
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [[ "$RECREATE" == "1" ]]; then
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "Environment ${ENV_NAME} already exists. To reinstall, run: bash setup_env.sh --recreate"
        exit 1
    fi
fi

conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
ensure_cpython

echo -e "\n[2/8] Installing CUDA ${CUDA_TOOLKIT_VERSION} toolkit ..."
conda install -c "nvidia/label/cuda-${CUDA_TOOLKIT_VERSION}" cuda-toolkit -y

echo -e "\n[3/8] Installing PyTorch ${PYTORCH_VERSION} from ${TORCH_INDEX_URL} ..."
pip install -U pip "setuptools<81" wheel
pip install --index-url "${TORCH_INDEX_URL}" \
    "torch==${PYTORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${PYTORCH_VERSION}"

echo -e "\n[4/8] Installing xformers ..."
pip install --index-url "${TORCH_INDEX_URL}" \
    "xformers==${XFORMERS_VERSION}" --no-deps

echo -e "\n[5/8] Checking ffmpeg ..."
if command -v ffmpeg >/dev/null 2>&1; then
    echo "  Using existing ffmpeg: $(command -v ffmpeg)"
else
    echo "  System ffmpeg was not found; Python dependencies will be installed first, then an imageio-ffmpeg shim will be created in the environment"
fi

echo -e "\n[6/8] Installing Python dependencies ..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# Some transitive dependencies in requirements may try to pull the newest
# transformers. Pin it to the version tested with torch 2.4, SGM/SV4D, and
# Qwen3-VL.
pip install "transformers==${TRANSFORMERS_VERSION}" "huggingface-hub<1.0,>=0.34.0"

if ! command -v ffmpeg >/dev/null 2>&1; then
    python - <<'PY'
import imageio_ffmpeg
import os
import stat
from pathlib import Path

target = Path(os.environ["CONDA_PREFIX"]) / "bin" / "ffmpeg"
exe = imageio_ffmpeg.get_ffmpeg_exe()
target.write_text(f'#!/usr/bin/env bash\nexec "{exe}" "$@"\n')
target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
print(f"  Created ffmpeg shim: {target} -> {exe}")
PY
fi
if ! command -v ffprobe >/dev/null 2>&1; then
    echo "  Warning: ffprobe was not found. The main pipeline does not need it, but codec detection for interactive video import will fall back."
fi

echo -e "\n[7/8] Installing subprojects ..."
cd "${SCRIPT_DIR}"

# Initialize git submodules. If the user installed from a tarball (no .git),
# expect the submodule directories to already contain source.
if [[ -d .git ]]; then
    git submodule sync --recursive
    git submodule update --init --recursive --checkout
else
    echo "  .git not found; assuming submodule sources are already vendored."
    for sub in DiffSynth-Studio generative-models; do
        if [[ ! -f "${sub}/pyproject.toml" && ! -f "${sub}/setup.py" ]]; then
            echo "  Missing submodule source: ${sub}/"
            echo "  Re-clone with submodules:  git clone --recursive https://github.com/VVeiCao/FreeOrbit4D.git"
            exit 1
        fi
    done
    for sub in page-4d/vggt page-4d/vggt_t_mask_mlp_fin10; do
        if [[ ! -d "${sub}/models" ]]; then
            echo "  Missing PAGE-4D source: ${sub}/"
            echo "  Use the full release archive or clone the repository with bundled PAGE-4D files."
            exit 1
        fi
    done
fi

# Local subprojects. Use --no-deps to avoid overwriting the packages installed above.
pip install -e DiffSynth-Studio --no-deps
pip install -e generative-models --no-deps

# Packages installed from pinned Git commits. accelerate is intentionally not listed here:
# it arrives transitively via peft (in requirements.txt) and is then used by
# diffsynth.
pip install -e "git+https://github.com/Stability-AI/datapipelines.git@${DATAPIPELINES_REF}#egg=sdata"
pip install "git+https://github.com/openai/CLIP.git@${CLIP_REF}"

# SAM2 for interactive mask annotation. Pin to a sam2.1 commit compatible with
# torch 2.4; the current main branch requires torch>=2.5.1. Use --no-deps to
# avoid replacing torch.
pip install "hydra-core==1.3.2"
pip install --no-deps "git+https://github.com/facebookresearch/sam2.git@${SAM2_REF}"

# PyTorch3D is built from source and needs ninja.
echo -e "\n    Building and installing PyTorch3D (MAX_JOBS=${MAX_JOBS}) ..."
pip install -U iopath fvcore
resolve_cuda_home
export FORCE_CUDA="${FORCE_CUDA:-1}"
pip install --no-deps "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_REF}" \
    --no-build-isolation

echo -e "\n[8/8] Verifying installation ..."
export FREEORBIT_EXPECTED_TORCH="${PYTORCH_VERSION}"
export FREEORBIT_EXPECTED_CUDA="${TORCH_CUDA_VERSION}"
export FREEORBIT_EXPECTED_TRANSFORMERS="${TRANSFORMERS_VERSION}"
export FREEORBIT_EXPECTED_SAM2_COMMIT="${SAM2_REF}"
python "${SCRIPT_DIR}/scripts/check_install.py"
python -m pip check

echo -e "\n============================================================"
echo " Installation complete."
echo "============================================================"
echo " Activate environment:  conda activate ${ENV_NAME}"
echo " Download weights:      bash download_checkpoints.sh required"
echo " Run demo:              python run_pipeline.py full --config configs/scenes/camel.yaml"
echo ""
