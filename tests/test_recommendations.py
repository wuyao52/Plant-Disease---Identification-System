from __future__ import annotations

from plant_disease.recommendations import get_recommendations


def test_recommendations_return_specific_advice() -> None:
    advice = get_recommendations("Potato___Late_blight")
    assert any("晚疫病" in item for item in advice)


def test_recommendations_fallback_for_unknown_label() -> None:
    advice = get_recommendations("Unknown")
    assert advice
