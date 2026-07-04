from __future__ import annotations

from pathlib import Path

from plant_disease.config import deep_update, load_config, resolve_path


def test_deep_update_keeps_base_values() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = deep_update(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1


def test_resolve_path_uses_root_for_relative_paths(tmp_path: Path) -> None:
    assert resolve_path("data/raw", tmp_path) == tmp_path / "data" / "raw"


def test_load_config_merges_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("training:\n  epochs: 3\n", encoding="utf-8")
    config = load_config(config_file, root_dir=tmp_path)
    assert config.section("training")["epochs"] == 3
    assert config.section("model")["architecture"] == "mobilenet_v3_small"
