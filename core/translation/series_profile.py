"""Series profile data model and persistence.

Stores known names, glossary terms, and notes per webtoon series.
Supports atomic JSON save/load with graceful error handling.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_PROFILES_DIR = Path("data/series_profiles")


@dataclass
class SeriesProfile:
    """Series-specific translation guidance profile."""

    series_id: str
    known_names: dict[str, str] = field(default_factory=dict)
    glossary: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def get_known_names_list(self) -> list[str]:
        """Return list of known source name keys."""
        return list(self.known_names.keys())

    def get_glossary_list(self) -> list[str]:
        """Return list of formatted glossary lines ('KEY -> translation')."""
        return [f"{k} -> {v}" for k, v in self.glossary.items()]

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to dictionary."""
        return {
            "series_id": self.series_id,
            "known_names": dict(self.known_names),
            "glossary": dict(self.glossary),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SeriesProfile:
        """Construct profile safely from dictionary."""
        if not isinstance(data, dict):
            logger.warning("SeriesProfile data is not a dict; returning empty profile")
            return cls(series_id="unknown")

        series_id = str(data.get("series_id", "unknown"))
        raw_names = data.get("known_names", {})
        raw_glossary = data.get("glossary", {})
        raw_notes = data.get("notes", [])

        known_names = (
            {str(k): str(v) for k, v in raw_names.items()}
            if isinstance(raw_names, dict)
            else {}
        )
        glossary = (
            {str(k): str(v) for k, v in raw_glossary.items()}
            if isinstance(raw_glossary, dict)
            else {}
        )
        notes = (
            [str(n) for n in raw_notes]
            if isinstance(raw_notes, list)
            else []
        )

        return cls(
            series_id=series_id,
            known_names=known_names,
            glossary=glossary,
            notes=notes,
        )

    @classmethod
    def load_from_json(cls, file_path: str | Path) -> SeriesProfile:
        """Load series profile from JSON file.

        Fails gracefully on missing or malformed JSON files by returning an empty profile.
        """
        path = Path(file_path)
        if not path.exists():
            logger.info(f"Series profile file not found at {path}; using empty profile")
            return cls(series_id=path.stem)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profile = cls.from_dict(data)
            logger.info(f"Loaded series profile '{profile.series_id}' from {path}")
            return profile
        except Exception as e:
            logger.warning(f"Failed to load series profile from {path}: {e}; using empty profile")
            return cls(series_id=path.stem)

    def save_to_json(self, file_path: str | Path | None = None, base_dir: str | Path = DEFAULT_PROFILES_DIR) -> Path:
        """Save series profile to JSON file using atomic safe write.

        Ensures parent directory exists and performs an atomic write via temporary file.
        """
        if file_path is None:
            target_path = Path(base_dir) / f"{self.series_id}.json"
        else:
            target_path = Path(file_path)

        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write pattern: write to temp file in same directory, then rename
        temp_fd, temp_file_path = tempfile.mkstemp(
            dir=target_path.parent, prefix=f"{target_path.stem}_", suffix=".tmp"
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(temp_file_path, target_path)
            logger.info(f"Saved series profile '{self.series_id}' to {target_path}")
        except Exception as e:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
            logger.error(f"Failed to save series profile to {target_path}: {e}")
            raise

        return target_path
