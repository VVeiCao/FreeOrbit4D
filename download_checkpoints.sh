#!/bin/bash
# FreeOrbit4D checkpoint download script
#
# Usage:
#   bash download_checkpoints.sh          # Download checkpoints required for end-to-end runs
#   bash download_checkpoints.sh sv4d     # Download only SV4D  (Stage 0, ~24GB)
#   bash download_checkpoints.sh dpg      # Download only DPG   (Stage 1, ~6.4GB)
#   bash download_checkpoints.sh wan      # Download only Wan2.2 (Stage 2, ~76GB)
#   bash download_checkpoints.sh vggt     # Pre-download VGGT  (Stage 1, ~4GB, HF cache)
#   bash download_checkpoints.sh qwen     # Pre-download Qwen3-VL (Stage 2 caption, ~4GB, HF cache)
#   bash download_checkpoints.sh sam2     # Download only SAM2  (interactive annotation, ~900MB)
#   bash download_checkpoints.sh all      # Download everything

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/run_pipeline.py" ]; then
    PROJECT_ROOT="${SCRIPT_DIR}"
else
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi
CKPT_DIR="${PROJECT_ROOT}/checkpoints"

TARGET="${1:-required}"

echo "============================================================"
echo " FreeOrbit4D checkpoint download"
echo "============================================================"
echo " Target directory: ${CKPT_DIR}"
echo " Download target: ${TARGET}"
echo "============================================================"

# ---- SV4D (Stage 0): stabilityai/sv4d2.0 ----
download_sv4d() {
    echo -e "\n[SV4D] Downloading SV4D multiview generation checkpoints (~12GB x2) ..."
    mkdir -p "${CKPT_DIR}/sv4d"

    if [ -f "${CKPT_DIR}/sv4d/sv4d2.safetensors" ]; then
        echo "  sv4d2.safetensors already exists; skipping"
    else
        huggingface-cli download stabilityai/sv4d2.0 sv4d2.safetensors \
            --local-dir "${CKPT_DIR}/sv4d"
    fi

    if [ -f "${CKPT_DIR}/sv4d/sv4d2_8views.safetensors" ]; then
        echo "  sv4d2_8views.safetensors already exists; skipping"
    else
        huggingface-cli download stabilityai/sv4d2.0 sv4d2_8views.safetensors \
            --local-dir "${CKPT_DIR}/sv4d"
    fi

    echo "[SV4D] Done"
}

# ---- DPG (Stage 1 background): PAGE-4D ----
download_dpg() {
    echo -e "\n[DPG] Downloading DPG background point-cloud checkpoint (~6.4GB) ..."
    mkdir -p "${CKPT_DIR}/dpg"
    local dpg_file="${CKPT_DIR}/dpg/checkpoint_150.pt"

    fix_dpg_zip_wrapper() {
        python - "$dpg_file" <<'PY'
from pathlib import Path
import os
import sys
import tempfile
import zipfile

path = Path(sys.argv[1])
if not zipfile.is_zipfile(path):
    raise SystemExit(0)

with zipfile.ZipFile(path) as zf:
    names = set(zf.namelist())
    # Google Drive may return a zip named checkpoint_150.pt containing the real
    # checkpoint under weights/checkpoint_150.pt. PyTorch then sees a zip
    # archive without the metadata expected from torch.save and fails at load
    # time. Rewrite the downloaded file in-place with the inner checkpoint.
    if "weights/checkpoint_150.pt" not in names:
        raise SystemExit(0)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix="checkpoint_150.", suffix=".pt", dir=str(path.parent)
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with zf.open("weights/checkpoint_150.pt") as src, tmp_path.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024 * 64)
                if not chunk:
                    break
                dst.write(chunk)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
print(f"  Extracted inner DPG checkpoint from Google Drive zip wrapper: {path}")
PY
    }

    if [ -f "${dpg_file}" ]; then
        fix_dpg_zip_wrapper
        echo "  checkpoint_150.pt already exists; skipping"
    else
        echo "  The DPG checkpoint is hosted on Google Drive:"
        echo ""
        echo "    https://drive.google.com/file/d/1c2G4z4sA3ouOmkPd2cZHDHnFB_n8LxU-/view"
        echo ""
        echo "  Place the downloaded file at: ${dpg_file}"
        echo ""
        # Use gdown from the active Python environment to avoid user-level PATH conflicts.
        if python -c "import gdown" >/dev/null 2>&1; then
            echo "  gdown detected; trying automatic download ..."
            python -m gdown "1c2G4z4sA3ouOmkPd2cZHDHnFB_n8LxU-" -O "${dpg_file}"
            fix_dpg_zip_wrapper
        else
            echo "  Tip: install gdown for automatic download: pip install gdown"
        fi
    fi

    echo "[DPG] Done"
}

# ---- VGGT (Stage 1 foreground): facebook/VGGT-1B ----
download_vggt() {
    echo -e "\n[VGGT] Pre-downloading VGGT-1B foreground point-cloud model (~4GB) ..."
    echo "  The model will be cached in the Hugging Face cache directory"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/VGGT-1B')
print('  VGGT-1B download complete')
"
    echo "[VGGT] Done"
}

# ---- Wan2.2 (Stage 2): alibaba-pai/Wan2.2-VACE-Fun-A14B ----
download_wan() {
    echo -e "\n[Wan2.2] Downloading Wan2.2-VACE-Fun-A14B video generation checkpoints (~76GB) ..."
    WAN_DIR="${CKPT_DIR}/wan2.2"
    mkdir -p "${WAN_DIR}"

    # Older local setups used symlinks into a repo-root models/ directory. The
    # release no longer ships that non-portable symlink, so remove broken links
    # before asking huggingface-cli to materialize files in-place.
    for p in \
        "${WAN_DIR}/high_noise_model" \
        "${WAN_DIR}/low_noise_model" \
        "${WAN_DIR}/models_t5_umt5-xxl-enc-bf16.pth" \
        "${WAN_DIR}/Wan2.1_VAE.pth" \
        "${WAN_DIR}/tokenizer/umt5-xxl"; do
        if [ -L "$p" ] && [ ! -e "$p" ]; then
            echo "  Removing broken legacy symlink: ${p}"
            rm -f "$p"
        fi
    done

    local required_files=(
        "high_noise_model/diffusion_pytorch_model.safetensors"
        "low_noise_model/diffusion_pytorch_model.safetensors"
        "models_t5_umt5-xxl-enc-bf16.pth"
        "Wan2.1_VAE.pth"
    )
    local missing=0
    for f in "${required_files[@]}"; do
        if [ ! -f "${WAN_DIR}/${f}" ]; then
            missing=1
            break
        fi
    done

    if [ "$missing" -eq 0 ]; then
        echo "  Wan2.2 checkpoints already exist; skipping"
    else
        huggingface-cli download alibaba-pai/Wan2.2-VACE-Fun-A14B \
            high_noise_model/diffusion_pytorch_model.safetensors \
            low_noise_model/diffusion_pytorch_model.safetensors \
            models_t5_umt5-xxl-enc-bf16.pth \
            Wan2.1_VAE.pth \
            --local-dir "${WAN_DIR}"
    fi

    # Tokenizer
    mkdir -p "${WAN_DIR}/tokenizer"
    if [ -d "${WAN_DIR}/tokenizer/umt5-xxl" ]; then
        echo "  Tokenizer already exists; skipping"
    else
        huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
            google/umt5-xxl/tokenizer.json \
            google/umt5-xxl/tokenizer_config.json \
            --local-dir "${WAN_DIR}/tokenizer_tmp"
        mv "${WAN_DIR}/tokenizer_tmp/google/umt5-xxl" "${WAN_DIR}/tokenizer/umt5-xxl"
        rm -rf "${WAN_DIR}/tokenizer_tmp"
    fi

    echo "[Wan2.2] Done"
}

# ---- Qwen3-VL (Stage 2 caption, optional): Qwen/Qwen3-VL-2B-Instruct ----
download_qwen() {
    echo -e "\n[Qwen3-VL] Pre-downloading Qwen3-VL-2B-Instruct caption model (~4GB) ..."
    echo "  The model will be cached in the Hugging Face cache directory"
    huggingface-cli download Qwen/Qwen3-VL-2B-Instruct
    echo "[Qwen3-VL] Done"
}

# ---- SAM2 (interactive mask annotation): sam2_hiera_large.pt ----
download_sam2() {
    local sam2_dir="${CKPT_DIR}/sam2"
    local sam2_file="${sam2_dir}/sam2_hiera_large.pt"
    if [ -f "$sam2_file" ] || [ -L "$sam2_file" ]; then
        echo "[SAM2] Already exists; skipping: ${sam2_file}"
        return
    fi
    echo -e "\n[SAM2] Downloading SAM2 segmentation checkpoint (~900MB) ..."
    mkdir -p "$sam2_dir"
    wget -q -O "$sam2_file" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    echo "[SAM2] Done"
}

# ---- Dispatch target ----
case "$TARGET" in
    sv4d)
        download_sv4d
        ;;
    dpg)
        download_dpg
        ;;
    vggt)
        download_vggt
        ;;
    wan)
        download_wan
        ;;
    qwen)
        download_qwen
        ;;
    sam2)
        download_sam2
        ;;
    required)
        download_sv4d
        download_dpg
        download_vggt
        download_wan
        download_qwen
        ;;
    all)
        download_sv4d
        download_dpg
        download_vggt
        download_wan
        download_qwen
        download_sam2
        ;;
    *)
        echo "Usage: bash download_checkpoints.sh [sv4d|dpg|vggt|wan|qwen|sam2|required|all]"
        exit 1
        ;;
esac

echo -e "\n============================================================"
echo " Download complete"
echo "============================================================"
echo ""
echo " Checkpoint file tree:"
find "${CKPT_DIR}" -type f -o -type l | sort | while read f; do
    size=$(du -h "$f" 2>/dev/null | cut -f1)
    echo "   ${f#${PROJECT_ROOT}/}  (${size})"
done
echo ""

case "$TARGET" in
    required)
        python "${PROJECT_ROOT}/scripts/check_install.py" --checkpoints
        ;;
    all)
        python "${PROJECT_ROOT}/scripts/check_install.py" --checkpoints --sam2
        ;;
    sam2)
        python "${PROJECT_ROOT}/scripts/check_install.py" --sam2
        ;;
esac
