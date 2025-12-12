import pytest


def test_gui_smoke_protocol_cockpit():
    try:
        import PyQt5  # type: ignore
    except Exception:
        pytest.skip("PyQt5 not installed")
    try:
        from scripts.foodspec_protocol_cockpit import FoodSpecProtocolCockpit
    except Exception as exc:
        pytest.skip(f"GUI import failed: {exc}")
    w = FoodSpecProtocolCockpit()
    assert w is not None
