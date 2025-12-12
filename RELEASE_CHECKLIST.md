## Release Checklist

- [ ] Bump version in `pyproject.toml` (MAJOR.MINOR.PATCH).
- [ ] Update CHANGELOG or `docs/release_notes.md` with highlights.
- [ ] Run lint + tests: `ruff check`, `black --check`, `pytest`.
- [ ] Build docs: `mkdocs build`.
- [ ] Tag release: `git tag vX.Y.Z && git push origin vX.Y.Z`.
- [ ] Build + upload to TestPyPI (optional), then PyPI: `python -m build && twine upload dist/*`.
- [ ] Verify example notebooks run (at least synthetic datasets).
- [ ] Announce status in README ("Public-usage ready vX.Y.Z" + known limitations).
