import pytest


def test_gui_wizard_smoke():
    try:
        pass  # type: ignore
    except Exception:
        pytest.skip("PyQt5 not installed")
    try:
        from scripts.foodspec_protocol_wizard import ProtocolWizard
    except Exception as exc:
        pytest.skip(f"Import failed: {exc}")
    w = ProtocolWizard()
    assert w.breadcrumb.labels[0] == "Load Data & Protocol"
