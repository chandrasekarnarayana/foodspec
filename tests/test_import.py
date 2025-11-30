from foodspec import __version__


def test_import_version():
    assert isinstance(__version__, str)
    assert __version__

