from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from modeling_gui.foodspec_adapter import FoodSpecPreset, load_preset, default_preset


@dataclass
class PresetInfo:
    name: str
    path: Optional[str]
    valid: bool
    error: Optional[str] = None
    preset: Optional[FoodSpecPreset] = None
    description: Optional[str] = None


class PresetManager:
    def __init__(self):
        self.presets: Dict[str, PresetInfo] = {}
        self.add_preset("Default Raman (oil/chips)", None, default_preset(), valid=True, description="Built-in peaks/ratios for oils/chips")

    def add_preset(self, name: str, path: Optional[str], preset: FoodSpecPreset, valid: bool = True, error: Optional[str] = None, description: Optional[str] = None):
        self.presets[name] = PresetInfo(name=name, path=path, valid=valid, error=error, preset=preset, description=description)

    def load_directory(self, dir_path: str):
        p = Path(dir_path)
        if not p.exists():
            return
        for file in p.glob("*.yml"):
            self._load_file(file)
        for file in p.glob("*.yaml"):
            self._load_file(file)
        for file in p.glob("*.json"):
            self._load_file(file)

    def _validate_payload(self, payload: dict) -> Optional[str]:
        if "peaks" not in payload or "ratios" not in payload:
            return "Missing 'peaks' or 'ratios' section"
        for peak in payload.get("peaks", []):
            if "name" not in peak or ("column" not in peak and "wavenumber" not in peak):
                return "Each peak needs 'name' and 'column' or 'wavenumber'"
        for ratio in payload.get("ratios", []):
            if "name" not in ratio or "numerator" not in ratio or "denominator" not in ratio:
                return "Each ratio needs 'name', 'numerator', 'denominator'"
        return None

    def _load_file(self, file: Path):
        try:
            if file.suffix.lower() in [".yml", ".yaml"]:
                if yaml is None:
                    raise ImportError("PyYAML not installed")
                payload = yaml.safe_load(file.read_text())
            else:
                payload = json.loads(file.read_text())
            err = self._validate_payload(payload)
            if err:
                self.add_preset(file.stem, str(file), default_preset(), valid=False, error=err)
            else:
                preset = load_preset(str(file))
                self.add_preset(file.stem, str(file), preset, valid=True)
        except Exception as exc:
            self.add_preset(file.stem, str(file), default_preset(), valid=False, error=str(exc))

    def list_presets(self) -> List[PresetInfo]:
        return list(self.presets.values())

    def get_preset(self, name: str) -> Optional[FoodSpecPreset]:
        info = self.presets.get(name)
        if info and info.valid:
            return info.preset
        return None
