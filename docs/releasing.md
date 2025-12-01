# Release & versioning

FoodSpec versions are tracked in `__version__` and `pyproject.toml`. Releases are published to PyPI and tagged on GitHub.

## Checklist (see RELEASING.md for details)
- Update version strings and CHANGELOG.
- Run full tests and `mkdocs build`; ensure benchmarks/CLI smoke tests pass.
- Build wheel/sdist and upload to TestPyPI, then PyPI.
- Tag the release in Git and create a GitHub release with notes and artifacts.

## PyPI and GitHub linkage
- The PyPI package and GitHub release tag should match (e.g., v0.x.y).
- Include release notes summarizing changes (features, fixes, docs updates).

For coordination, contact Chandrasekar Subramani Narayan; external contributors should propose release candidates via PR.
