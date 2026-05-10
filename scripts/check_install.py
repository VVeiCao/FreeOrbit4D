#!/usr/bin/env python3
"""Validate a FreeOrbit4D installation without running the full pipeline."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
import warnings
import zipfile
from importlib import metadata
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")


EXPECTED_TORCH = os.environ.get("FREEORBIT_EXPECTED_TORCH", "2.4.0")
EXPECTED_CUDA = os.environ.get("FREEORBIT_EXPECTED_CUDA", "12.1")
EXPECTED_TRANSFORMERS = os.environ.get("FREEORBIT_EXPECTED_TRANSFORMERS", "4.57.1")
EXPECTED_SAM2_COMMIT = os.environ.get(
    "FREEORBIT_EXPECTED_SAM2_COMMIT",
    "aa9b8722d0585b661ded4b3dff1bd103540554ae",
)

REQUIRED_FILES = [
    "checkpoints/sv4d/sv4d2.safetensors",
    "checkpoints/sv4d/sv4d2_8views.safetensors",
    "checkpoints/dpg/checkpoint_150.pt",
    "checkpoints/wan2.2/high_noise_model/diffusion_pytorch_model.safetensors",
    "checkpoints/wan2.2/low_noise_model/diffusion_pytorch_model.safetensors",
    "checkpoints/wan2.2/models_t5_umt5-xxl-enc-bf16.pth",
    "checkpoints/wan2.2/Wan2.1_VAE.pth",
    "checkpoints/wan2.2/tokenizer/umt5-xxl/tokenizer.json",
    "checkpoints/wan2.2/tokenizer/umt5-xxl/tokenizer_config.json",
]

OPTIONAL_FILES = [
    "checkpoints/sam2/sam2_hiera_large.pt",
]

MIN_FILE_BYTES = {
    "checkpoints/sv4d/sv4d2.safetensors": 11_000_000_000,
    "checkpoints/sv4d/sv4d2_8views.safetensors": 11_000_000_000,
    "checkpoints/dpg/checkpoint_150.pt": 6_000_000_000,
    "checkpoints/wan2.2/high_noise_model/diffusion_pytorch_model.safetensors": 34_000_000_000,
    "checkpoints/wan2.2/low_noise_model/diffusion_pytorch_model.safetensors": 34_000_000_000,
    "checkpoints/wan2.2/models_t5_umt5-xxl-enc-bf16.pth": 11_000_000_000,
    "checkpoints/wan2.2/Wan2.1_VAE.pth": 500_000_000,
    "checkpoints/wan2.2/tokenizer/umt5-xxl/tokenizer.json": 10_000_000,
    "checkpoints/wan2.2/tokenizer/umt5-xxl/tokenizer_config.json": 10_000,
    "checkpoints/sam2/sam2_hiera_large.pt": 800_000_000,
}

REQUIRED_HF_REPOS = [
    "facebook/VGGT-1B",
    "Qwen/Qwen3-VL-2B-Instruct",
]

SUBMODULE_PATHS = [
    "DiffSynth-Studio",
    "generative-models",
]

VENDORED_DIRS = [
    "page-4d/vggt",
    "page-4d/vggt_t_mask_mlp_fin10",
]


class Reporter:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK]   {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.errors.append(message)
        print(f"[FAIL] {message}")


def get_project_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def check_command(reporter: Reporter, name: str, required: bool = True) -> None:
    path = shutil.which(name)
    if path:
        reporter.ok(f"{name}: {path}")
    elif required:
        reporter.fail(f"{name} not found in PATH")
    else:
        reporter.warn(f"{name} not found in PATH")


def check_import(reporter: Reporter, module: str, label: str | None = None) -> object | None:
    try:
        imported = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report exact import failure.
        reporter.fail(f"import {module} failed: {exc}")
        return None
    version = getattr(imported, "__version__", None)
    suffix = f" {version}" if version else ""
    reporter.ok(f"import {label or module}{suffix}")
    return imported


def check_environment(reporter: Reporter) -> None:
    reporter.ok(f"Python: {platform.python_version()} ({platform.python_implementation()})")
    if platform.python_implementation() != "CPython":
        reporter.fail("Python must be CPython; GraalPy is not supported")
    if sys.version_info[:2] != (3, 10):
        reporter.warn("Python 3.10 is the tested version")

    check_command(reporter, "git")
    check_command(reporter, "ffmpeg", required=False)
    check_command(reporter, "ffprobe", required=False)
    check_command(reporter, "nvidia-smi", required=False)

    torch = check_import(reporter, "torch")
    if torch is not None:
        if not torch.__version__.startswith(EXPECTED_TORCH):
            reporter.fail(f"torch must start with {EXPECTED_TORCH}, got {torch.__version__}")
        torch_cuda = getattr(torch.version, "cuda", None)
        if not torch_cuda:
            reporter.fail("torch must be a CUDA build; CPU-only PyTorch is not supported")
        elif EXPECTED_CUDA and torch_cuda != EXPECTED_CUDA:
            reporter.fail(f"torch CUDA must be {EXPECTED_CUDA}, got {torch_cuda}")
        elif torch_cuda:
            reporter.ok(f"torch CUDA build: {torch_cuda}")
        if not torch.cuda.is_available():
            reporter.fail("torch.cuda.is_available() is False")
        else:
            reporter.ok(f"CUDA devices visible: {torch.cuda.device_count()}")

    for module in [
        "torchvision",
        "xformers",
        "pytorch3d",
        "open3d",
        "sgm",
        "diffsynth",
        "sam2",
        "gdown",
        "hydra",
        "iopath",
        "fvcore",
    ]:
        check_import(reporter, module)

    transformers = check_import(reporter, "transformers")
    if transformers is not None:
        if transformers.__version__ != EXPECTED_TRANSFORMERS:
            reporter.fail(
                f"transformers must be {EXPECTED_TRANSFORMERS}, got {transformers.__version__}"
            )
        try:
            from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            reporter.fail(f"Qwen3VLForConditionalGeneration import failed: {exc}")
        else:
            reporter.ok("Qwen3-VL class available in transformers")

    try:
        direct_url = metadata.distribution("SAM-2").read_text("direct_url.json") or ""
    except metadata.PackageNotFoundError:
        reporter.fail("SAM-2 package metadata not found")
    else:
        if EXPECTED_SAM2_COMMIT not in direct_url:
            reporter.warn("SAM2 is not installed from the tested sam2.1 commit")
        else:
            reporter.ok("SAM2 commit matches the tested sam2.1 revision")


def check_project_layout(reporter: Reporter, project_root: Path) -> None:
    if not (project_root / "run_pipeline.py").is_file():
        reporter.fail(f"project root is wrong or incomplete: {project_root}")
        return
    reporter.ok(f"project root: {project_root}")

    check_submodules(reporter, project_root)
    check_vendored_dirs(reporter, project_root)


def check_file_list(reporter: Reporter, project_root: Path, files: list[str]) -> None:
    for rel in files:
        path = project_root / rel
        if path.is_file() or path.is_symlink():
            reporter.ok(f"checkpoint present: {rel}")
            min_bytes = MIN_FILE_BYTES.get(rel)
            if min_bytes is not None:
                size = path.stat().st_size
                if size < min_bytes:
                    reporter.fail(
                        f"checkpoint too small, likely incomplete: {rel} "
                        f"({size} bytes < {min_bytes} bytes)"
                    )
        else:
            reporter.fail(f"checkpoint missing: {rel}")

    check_dpg_checkpoint_shape(reporter, project_root)


def check_dpg_checkpoint_shape(reporter: Reporter, project_root: Path) -> None:
    path = project_root / "checkpoints/dpg/checkpoint_150.pt"
    if not path.is_file():
        return
    try:
        if not zipfile.is_zipfile(path):
            reporter.ok("DPG checkpoint is not a zip wrapper")
            return
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
    except Exception as exc:  # noqa: BLE001
        reporter.warn(f"could not inspect DPG checkpoint container: {exc}")
        return

    if "weights/checkpoint_150.pt" in names:
        reporter.fail(
            "DPG checkpoint is a Google Drive zip wrapper; run download_checkpoints.sh "
            "again or extract weights/checkpoint_150.pt to checkpoints/dpg/checkpoint_150.pt"
        )
    else:
        reporter.ok("DPG checkpoint container shape looks loadable")


def run_command(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def check_submodules(reporter: Reporter, project_root: Path) -> None:
    if not (project_root / ".git").exists():
        reporter.warn("not a git checkout; exact submodule commit validation skipped")
        return

    for rel in SUBMODULE_PATHS:
        path = project_root / rel
        if not path.is_dir():
            reporter.fail(f"submodule missing: {rel}; run git submodule update --init --recursive")
            continue

        recorded = run_command(["git", "ls-files", "--stage", "--", rel], project_root)
        if recorded.returncode != 0 or not recorded.stdout.strip():
            reporter.fail(f"cannot read recorded submodule commit for {rel}: {recorded.stderr.strip()}")
            continue

        parts = recorded.stdout.split()
        if len(parts) < 2 or parts[0] != "160000":
            reporter.fail(f"{rel} is not recorded as a git submodule in the superproject")
            continue
        expected = parts[1]

        head = run_command(["git", "-C", rel, "rev-parse", "HEAD"], project_root)
        if head.returncode != 0:
            reporter.fail(f"submodule not initialized: {rel}; run git submodule update --init --recursive")
            continue
        actual = head.stdout.strip()

        if actual == expected:
            reporter.ok(f"submodule pinned: {rel} @ {actual[:12]}")
        else:
            reporter.fail(
                f"submodule commit mismatch for {rel}: expected {expected}, got {actual}"
            )

        dirty = run_command(["git", "-C", rel, "status", "--short"], project_root)
        if dirty.returncode == 0 and dirty.stdout.strip():
            reporter.warn(f"submodule has local changes: {rel}")


def check_vendored_dirs(reporter: Reporter, project_root: Path) -> None:
    for rel in VENDORED_DIRS:
        path = project_root / rel
        if path.is_dir():
            reporter.ok(f"vendored code present: {rel}")
        else:
            reporter.fail(f"vendored code missing: {rel}")


def check_hf_cache(reporter: Reporter) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        reporter.fail(f"huggingface_hub import failed: {exc}")
        return

    for repo_id in REQUIRED_HF_REPOS:
        try:
            path = snapshot_download(repo_id, local_files_only=True)
        except Exception as exc:  # noqa: BLE001
            reporter.fail(f"HF cache missing or incomplete for {repo_id}: {exc}")
        else:
            reporter.ok(f"HF cache present: {repo_id} -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", help="Repository root. Defaults to this script's parent repo.")
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="Also verify required checkpoint files and Hugging Face cache entries.",
    )
    parser.add_argument(
        "--sam2",
        action="store_true",
        help="Also verify the optional SAM2 interactive annotator checkpoint.",
    )
    args = parser.parse_args()

    reporter = Reporter()
    project_root = get_project_root(args.project_root)

    check_environment(reporter)
    check_project_layout(reporter, project_root)

    if args.checkpoints:
        check_file_list(reporter, project_root, REQUIRED_FILES)
        check_hf_cache(reporter)

    if args.sam2:
        check_file_list(reporter, project_root, OPTIONAL_FILES)

    print()
    if reporter.errors:
        print(f"Install check failed with {len(reporter.errors)} error(s).")
        return 1
    if reporter.warnings:
        print(f"Install check passed with {len(reporter.warnings)} warning(s).")
    else:
        print("Install check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
