"""File I/O utilities for directory management, JSON, and image file operations."""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from glob import glob


def ensure_dir(path: str) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_frame_dirs(data_dir: str, max_frames: int = None) -> List[str]:
    """Find all numerically-named subdirectories, sorted and optionally limited."""
    data_dir = Path(data_dir)

    frame_dirs = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ])

    if max_frames is not None:
        frame_dirs = frame_dirs[:max_frames]

    return frame_dirs


def load_json(path: str) -> Dict:
    """Load and return data from a JSON file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def save_json(data: Dict, path: str, indent: int = 2):
    """Save a dictionary as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def find_image_files(directory: str,
                    extensions: List[str] = None,
                    pattern: str = None) -> List[str]:
    """Find all image files in a directory, sorted by name.

    Args:
        directory: Image directory path.
        extensions: File extensions to match (default: common image formats).
        pattern: Glob pattern (overrides extensions if specified).
    """
    directory = Path(directory)

    if pattern is not None:
        return sorted([str(p) for p in directory.glob(pattern)])

    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f'*{ext}'))

    return sorted([str(p) for p in image_files])


def load_image_list(directory: str, max_count: int = None) -> List[str]:
    """Load image file paths from a directory, optionally limited."""
    image_files = find_image_files(directory)

    if max_count is not None:
        image_files = image_files[:max_count]

    return image_files


def copy_file(src: str, dst: str):
    """Copy a file, creating parent directories as needed."""
    src = Path(src)
    dst = Path(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def move_file(src: str, dst: str):
    """Move a file, creating parent directories as needed."""
    src = Path(src)
    dst = Path(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def get_file_size(path: str) -> int:
    """Return file size in bytes."""
    return os.path.getsize(path)
