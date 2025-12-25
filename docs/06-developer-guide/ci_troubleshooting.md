# CI/CD Troubleshooting Guide

This guide covers common CI/CD issues, how to diagnose them, and their fixes.

## Quick Reference

| Error | Cause | Fix |
|-------|-------|-----|
| **mkdocs build fails** | Missing plugin or bad nav | `pip install -e ".[docs]"` then `mkdocs build --strict` |
| **"No module named 'mkdocs_material'"** | Plugin not installed | `pip install mkdocs-material>=9.5.0,<10.0` |
| **Broken internal links in docs** | Docs reference non-existent `.md` files | Check `mkdocs.yml` nav paths, run link validation |
| **CLI smoke test fails** | CLI entry point broken or import error | `pip install -e "." && foodspec --help` |
| **pytest import errors** | Missing dependencies or wrong PYTHONPATH | `pip install -e ".[dev]" && pytest tests/` |
| **ruff check fails** | Linting issues in src/tests/scripts | Run `ruff check --fix src tests scripts` |

---

## Common Issues & Solutions

### 1. MkDocs Build Fails: Plugin Import Error

**Error Message:**
```
ModuleNotFoundError: No module named 'mkdocs_material'
```

**Cause:**
- `mkdocs-material` is not installed
- Dependency version mismatch or incomplete `pip install`

**Diagnosis:**
```bash
python -c "import mkdocs_material; print('OK')"  # If this fails, plugin is missing
pip list | grep mkdocs  # Check what's installed
```

**Fix:**
```bash
# Option 1: Install docs-only extras (recommended for CI)
pip install -e ".[docs]"

# Option 2: Install all dev extras
pip install -e ".[dev]"

# Option 3: Install mkdocs-material directly
pip install "mkdocs-material>=9.5.0,<10.0"
```

**Prevention:**
- Always use `pip install -e ".[docs]"` in docs CI job
- `pyproject.toml` specifies version bounds: `mkdocs-material>=9.5.0,<10.0`

---

### 2. MkDocs Build Fails: Bad Navigation

**Error Message:**
```
ERROR (mkdocs): File not found: path/to/nonexistent.md
```

**Cause:**
- `mkdocs.yml` `nav:` section references a `.md` file that doesn't exist
- File was deleted but nav entry not updated

**Diagnosis:**
```bash
# Run strict mode (enabled in CI)
mkdocs build --strict

# Check for specific bad paths
grep -n "nonexistent.md" mkdocs.yml
```

**Fix:**
1. Identify the broken nav entry in `mkdocs.yml`
2. Either:
   - **Delete the nav entry** if page is no longer needed
   - **Create the missing file** if it should exist
   - **Update the nav path** if file was renamed

3. Re-run: `mkdocs build --strict`

**Prevention:**
- Before deleting `.md` files, remove corresponding nav entries in `mkdocs.yml`
- Test locally: `mkdocs build --strict` catches all broken links

---

### 3. Broken Internal Links in Documentation

**Error Message:**
```
ERROR: Found 3 broken internal links:
  mkdocs_site/guides/advanced.html: link to ../nonexistent.html
```

**Cause:**
- Markdown contains broken link: `[text](path/to/nonexistent.md)`
- File was deleted but markdown links not updated

**Diagnosis:**
After `mkdocs build --strict` completes, check the "Upload build logs" artifact for the link checker results:
```python
# CI runs this validation automatically
for html_file in site_dir.rglob("*.html"):
    # Checks all href="..." links exist as files
```

**Fix:**
1. Identify the markdown file with broken link from the log
2. Update the link to point to correct file, or
3. Remove the broken link entirely

**Example:**
```markdown
# Before (BROKEN)
See the [advanced guide](../../advanced/nonexistent.md)

# After (FIXED)
See the [advanced guide](../../advanced/correct_guide.md)
```

**Prevention:**
- Run link validation locally: `mkdocs build --strict`
- Check artifacts in failed CI runs for detailed error messages

---

### 4. CLI Smoke Test Fails

**Error Message:**
```
foodspec: command not found
OR
ImportError: No module named 'foodspec'
```

**Cause:**
- Package not installed
- CLI entry point broken (bad import in source code)
- Python environment different from build environment

**Diagnosis:**
```bash
# Check if CLI is installed
which foodspec
foodspec --help

# Check if package is importable
python -c "import foodspec; print(foodspec.__version__)"

# Check entry points in pyproject.toml
grep -A 10 "\[project.scripts\]" pyproject.toml
```

**Fix:**
```bash
# Reinstall package
pip install -e "."

# If that fails, check for import errors in src/
python -c "from foodspec import __main__; print('OK')"

# Run smoke tests manually
foodspec --help
foodspec --version
foodspec-registry --help
```

**Prevention:**
- CI job `cli-smoke-test` automatically tests all entry points on every build
- Check local: `pip install -e "." && foodspec --help`

---

### 5. pytest Fails: Import Errors

**Error Message:**
```
ImportError: No module named 'foodspec'
ModuleNotFoundError: No module named 'scipy'
```

**Cause:**
- `foodspec` package not installed in editable mode
- Missing optional dependencies ([dev] extras not installed)
- PYTHONPATH not set correctly

**Diagnosis:**
```bash
# Check if foodspec is importable
python -c "import foodspec"

# Check installed packages
pip list | grep foodspec

# Check PYTHONPATH
echo $PYTHONPATH

# Run pytest with verbose output
pytest -vv tests/
```

**Fix:**
```bash
# Option 1: Install in editable mode with all dev extras
pip install -e ".[dev]"

# Option 2: Explicitly set PYTHONPATH (not recommended, use Option 1)
export PYTHONPATH=/path/to/FoodSpec/src:$PYTHONPATH
pytest tests/

# Option 3: If scipy/other dependency missing
pip install scipy  # or other missing package
```

**Prevention:**
- Always install with `pip install -e ".[dev]"` before running tests
- CI job `test` automatically includes all dev extras

---

### 6. ruff Linting Fails

**Error Message:**
```
error: 1 error
src/foodspec/module.py:42:1: F401 Module imported but unused
```

**Cause:**
- Linting violations in `src/`, `tests/`, or `scripts/`
- Unused imports, formatting issues, type issues

**Diagnosis:**
```bash
# Run ruff check to see all violations
ruff check src tests scripts

# Run ruff format to see what will be changed
ruff format --check src tests scripts
```

**Fix:**
```bash
# Option 1: Auto-fix all issues (recommended)
ruff check --fix src tests scripts
ruff format src tests scripts

# Option 2: Manually fix (for complex issues)
# Edit files, then re-run ruff check
```

**Prevention:**
- Install ruff locally: `pip install ruff`
- Run before committing: `ruff check --fix src tests scripts`
- Configure IDE to run ruff on save (VS Code: Ruff extension)

---

### 7. GitHub Actions Workflow Not Triggering

**Error Message:**
- Workflow doesn't run on push/PR
- Workflow file appears invalid

**Diagnosis:**
```bash
# Check workflow syntax
act -l  # If act is installed: list all workflows

# Check .github/workflows/*.yml is valid YAML
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"

# View GitHub Actions logs in web UI
# Settings > Actions > Run logs
```

**Fix:**
1. Ensure `.github/workflows/ci.yml` is valid YAML
2. Check that `on:` trigger events are correct:
   ```yaml
   on:
     push:
     pull_request:
   ```
3. If workflow still doesn't trigger, try re-saving the file:
   ```bash
   git add .github/workflows/ci.yml
   git commit -m "Re-trigger workflow"
   git push
   ```

**Prevention:**
- Test workflow locally with `act` (requires Docker):
  ```bash
  pip install act
  act -j lint  # Test lint job
  ```

---

## Advanced Debugging

### Run a Single CI Job Locally

Use [act](https://github.com/nektos/act) to simulate GitHub Actions locally:

```bash
# Install act (requires Docker)
brew install act  # macOS
sudo apt install act  # Ubuntu (if available)

# List all jobs
act -l

# Run a specific job
act -j lint
act -j test

# Run with specific Python version
act -j test -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:full-latest
```

### Check CI Artifacts

After CI fails, download artifacts from GitHub Actions:
1. Go to repo > Actions > [Failed Run]
2. Download artifacts:
   - `mkdocs-build-logs` - MkDocs build output
   - `mkdocs-site` - Generated HTML site
   - `test-results-*` - JUnit XML test results

### Enable Debug Logging

Add step to CI workflow to see more detail:

```yaml
- name: Enable debug logging
  env:
    RUNNER_DEBUG: 1
  run: echo "Debug mode enabled"
```

Then check GitHub Actions logs for detailed step outputs.

---

## Common Version-Related Issues

### Issue: "pip install -e ".[dev]" is slow or hangs"

**Solution:** Your system may be resolving dependencies. Try:
```bash
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]" --no-cache-dir
```

### Issue: Different Python versions have different dependencies

**Solution:** Always test with the same Python version as CI:
```bash
# Check ci.yml for test matrix
grep "python-version:" .github/workflows/ci.yml

# Test locally with specific version (if using pyenv)
pyenv local 3.11.0
pip install -e ".[dev]"
pytest tests/
```

---

## Performance Tuning

### Speed Up CI Runs

1. **Enable Caching** (already enabled in `ci.yml`):
   - `actions/setup-python` with `cache: "pip"`
   - Caches `~/.cache/pip` based on `pyproject.toml`

2. **Parallel Jobs** (already enabled):
   - `test` job runs Python 3.10, 3.11, 3.12 in parallel
   - Jobs are independent (see `needs:` in workflow)

3. **Fail Fast** (if desired):
   - Set `fail-fast: true` in job matrix to stop other jobs if one fails
   - Currently set to `false` to see all failures

---

## FAQ

**Q: How do I test the docs job locally?**
```bash
pip install -e ".[docs]"
mkdocs build --strict
```

**Q: How do I test the CI job locally?**
Use `act` (see "Advanced Debugging" section above).

**Q: What if I can't fix the CI issue?**
1. Check this guide (start at "Quick Reference")
2. Review the error in the failing job's logs (GitHub Actions UI)
3. Review artifacts (`mkdocs-build-logs`, test results, etc.)
4. Ask for help in GitHub Discussions or Issues

**Q: Can I skip CI checks for a push?**
Not recommended, but you can use `[skip ci]` in commit message (some workflows support it). It's better to fix the issue!

---

## Contact & Support

For CI/CD issues:
- Check this guide first
- Review GitHub Actions logs
- Open an issue with the job name and error message

