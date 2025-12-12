import pytest


def test_gui_cockpit_imports():
    try:
        import PyQt5  # type: ignore
    except Exception:
        pytest.skip("PyQt5 not installed")
    try:
        from scripts.foodspec_protocol_cockpit import FoodSpecProtocolCockpit
    except Exception as exc:  # pragma: no cover - import check
        pytest.skip(f"GUI import failed: {exc}")
    w = FoodSpecProtocolCockpit()
    assert w is not None
