"""Unit tests for SeriesProfile model and persistence."""
import json
import pytest
from pathlib import Path
from core.translation.series_profile import SeriesProfile


def test_empty_profile():
    profile = SeriesProfile(series_id="test_series")
    assert profile.series_id == "test_series"
    assert profile.known_names == {}
    assert profile.glossary == {}
    assert profile.notes == []
    assert profile.get_known_names_list() == []
    assert profile.get_glossary_list() == []


def test_profile_dict_conversion():
    profile = SeriesProfile(
        series_id="test_series",
        known_names={"HERO": "Kahraman"},
        glossary={"GUILD": "Lonca"},
        notes=["Note 1"],
    )
    d = profile.to_dict()
    assert d["series_id"] == "test_series"
    assert d["known_names"] == {"HERO": "Kahraman"}
    assert d["glossary"] == {"GUILD": "Lonca"}
    assert d["notes"] == ["Note 1"]

    restored = SeriesProfile.from_dict(d)
    assert restored.series_id == profile.series_id
    assert restored.known_names == profile.known_names
    assert restored.glossary == profile.glossary
    assert restored.notes == profile.notes


def test_profile_json_save_load(tmp_path):
    target_path = tmp_path / "custom_dir" / "test_profile.json"
    profile = SeriesProfile(
        series_id="my_series",
        known_names={"LUO TIAN": "Luo Tian"},
        glossary={"SECRET REALM": "gizli âlem"},
    )
    saved_path = profile.save_to_json(target_path)
    assert saved_path.exists()

    loaded = SeriesProfile.load_from_json(saved_path)
    assert loaded.series_id == "my_series"
    assert loaded.known_names == {"LUO TIAN": "Luo Tian"}
    assert loaded.glossary == {"SECRET REALM": "gizli âlem"}


def test_malformed_json_safe_failure(tmp_path):
    bad_file = tmp_path / "corrupted.json"
    bad_file.write_text("invalid json content {{{", encoding="utf-8")

    # Should not crash, returns empty profile with file stem as series_id
    profile = SeriesProfile.load_from_json(bad_file)
    assert profile.series_id == "corrupted"
    assert profile.known_names == {}
    assert profile.glossary == {}
