"""YAML config management with inheritance, deep merge, and dot-path access."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from copy import deepcopy

from utils.trajectory import format_arc_trajectory_name


class Config:
    """Configuration manager with YAML inheritance and dot-path access."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config, loading from config_path or default.yaml."""
        self.config_dir = Path(__file__).parent.parent / 'configs'
        self._data = self._load_config(config_path)
        self._auto_generate_paths()

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load config file with _base_ inheritance support."""
        if config_path is None:
            config_path = self.config_dir / 'default.yaml'
        else:
            config_path = Path(config_path)
            if not config_path.is_absolute():
                if str(config_path).startswith('configs/'):
                    config_path = self.config_dir.parent / config_path
                else:
                    config_path = self.config_dir / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        if '_base_' in config:
            base_path = config['_base_']
            if not Path(base_path).is_absolute():
                base_path = config_path.parent / base_path

            base_config = self._load_config(str(base_path))
            config = self._merge_config(base_config, config)
            del config['_base_']

        return config

    def _merge_config(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two config dicts (override wins)."""
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def _auto_generate_paths(self):
        """Auto-generate output paths from project.name and rendering config.

        Generates output_multiview, output_prepared, and output_rendering_base
        paths only when not explicitly set in config.
        """
        name = self.get('project.name')
        if not name:
            return

        output_root = self.get('project.output_root', 'outputs')
        scene_name = self._extract_scene_name(name)
        trajectory_name = self._get_trajectory_name()

        path_mappings = {
            'project.output_multiview': f"{output_root}/multiview/{scene_name}",
            'project.output_prepared': f"{output_root}/prepared/{scene_name}",
            'project.output_rendering_base': f"{output_root}/rendering/{scene_name}",
        }

        for path_key, path_value in path_mappings.items():
            if not self.get(path_key):
                self.update(path_key, path_value)

    def _extract_scene_name(self, name: str) -> str:
        """Extract scene name. Priority: scene_name field > name field > input_images path."""
        if self.has('project.scene_name'):
            return self.get('project.scene_name')

        if name and name.strip():
            base_name = name.split('_pitch_')[0].split('_yaw_')[0].split('_roll_')[0]
            if base_name:
                return base_name

        input_images = self.get('project.input_images')
        if input_images:
            from pathlib import Path
            scene_name = Path(input_images).name
            if scene_name:
                return scene_name

        return name

    def _get_trajectory_name(self) -> str:
        """Get trajectory name (e.g. 'arc_yaw_-120_scale_1p0') from config."""
        if self.has('project.trajectory_name'):
            return self.get('project.trajectory_name')

        arc_type = self.get('stage_1.rendering.arc_type', 'yaw')
        arc_angle = self.get('stage_1.rendering.arc_angle', 90)
        arc_radius_scale = self.get('stage_1.rendering.arc_radius_scale', 1.0)
        return format_arc_trajectory_name(arc_type, arc_angle, arc_radius_scale)

    def get(self, path: str, default: Any = None) -> Any:
        """Get a config value by dot-separated path."""
        keys = path.split('.')
        value = self._data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def update(self, path: str, value: Any):
        """Set a config value by dot-separated path, creating intermediate dicts."""
        keys = path.split('.')
        data = self._data

        for key in keys[:-1]:
            if key not in data:
                data[key] = {}
            data = data[key]

        data[keys[-1]] = value

    def set(self, path: str, value: Any):
        """Alias for update."""
        self.update(path, value)

    def has(self, path: str) -> bool:
        """Check whether a dot-separated path exists in the config."""
        keys = path.split('.')
        value = self._data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return False

        return True

    def to_dict(self) -> Dict:
        """Return a deep copy of the config as a plain dict."""
        return deepcopy(self._data)

    def save(self, path: str):
        """Save the config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self._data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )

    def __repr__(self) -> str:
        return f"Config({len(self._data)} top-level keys)"

    def __str__(self) -> str:
        return yaml.dump(self._data, default_flow_style=False, allow_unicode=True)

    def keys(self):
        """Return top-level keys."""
        return self._data.keys()

    def items(self):
        """Return top-level key-value pairs."""
        return self._data.items()


def load_config(config_path: Optional[str] = None) -> Config:
    """Convenience function to load a Config."""
    return Config(config_path)
