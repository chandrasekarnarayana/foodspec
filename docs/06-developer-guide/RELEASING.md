# Releasing foodspec

This project follows [Semantic Versioning](https://semver.org/). Patch releases fix bugs, minor releases add backward-compatible features, and major releases include breaking changes.

## Release checklist

1. **Update version**  
   - Set `__version__` in `src/foodspec/__init__.py`.  
   - Update `[project].version` in `pyproject.toml` if maintained there.

2. **Update changelog**  
   - Add a dated entry for the release in `CHANGELOG.md`, moving items from "Unreleased" to the new version.

3. **Verify build and tests**  
   - Install deps: `pip install -e ".[dev]"`  
   - Run tests: `pytest`  
   - Build docs: `mkdocs build`

4. **Build distribution artifacts**  
   ```bash
   python -m build
   ```

5. **Publish to TestPyPI (optional but recommended)**  
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```
   Install from TestPyPI in a clean env to sanity-check.

6. **Publish to PyPI**  
   ```bash
   python -m twine upload dist/*
   ```

7. **Tag and GitHub release**  
   - Tag the commit: `git tag vX.Y.Z` and push tags: `git push origin vX.Y.Z`  
   - Draft a GitHub release with notes from the changelog and attach artifacts if desired.

8. **Post-release**  
   - Bump version to the next dev iteration if needed.  
   - Move an "Unreleased" section into `CHANGELOG.md` for ongoing work.
