from foodspec.logo import LOGO_BASE64, get_logo_base64, get_logo_bytes, save_logo


def test_logo_base64_roundtrip(tmp_path):
    b = get_logo_bytes()
    assert isinstance(b, (bytes, bytearray))
    assert len(b) > 1000  # sanity: non-empty
    assert get_logo_base64() == LOGO_BASE64
    out = save_logo(tmp_path / "logo.png")
    assert out.exists()
    assert out.read_bytes() == b
