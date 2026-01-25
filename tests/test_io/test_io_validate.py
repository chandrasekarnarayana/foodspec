from foodspec.io.validators import validate_input


def test_validate_input_unknown(tmp_path):
    fake = tmp_path / "data.unknown"
    fake.write_text("x")
    result = validate_input(str(fake))
    assert result["errors"]

