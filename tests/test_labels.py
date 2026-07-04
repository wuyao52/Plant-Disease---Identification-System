from __future__ import annotations

from plant_disease.labels import display_label, normalize_label


def test_normalize_label_replaces_imagefolder_separators() -> None:
    assert normalize_label("Tomato___Late_blight") == "Tomato - Late blight"


def test_display_label_uses_chinese_name_when_known() -> None:
    label = display_label("Potato___Early_blight")
    assert "马铃薯早疫病" in label
    assert "Potato - Early blight" in label
