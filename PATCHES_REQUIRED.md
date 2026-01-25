# FoodSpec Refactor Deliverables - Patch File
# Date: January 25, 2026
# Apply these patches to fix 5 bugs before executing refactoring

## PATCH 1: scripts/validate_architecture.py - Exit Code Bug (Line 286)
## File: scripts/validate_architecture.py
## Type: CRITICAL - fixes --strict flag not working
## Lines: 286

OLD (Line 286):
```python
    sys.exit(0 if all_passed else 0)  # Always exits 0 unless --strict
```

NEW (Line 286):
```python
    sys.exit(1 if (args.strict and not all_passed) else 0)
```

EXPLANATION:
The old code always exits 0 regardless of --strict flag. The new code properly 
exits with code 1 when --strict is used and tests failed, which will cause CI to fail.

---

## PATCH 2: tests/test_architecture.py - test_no_rewrite_imports Logic Bug (Lines 140-144)
## File: tests/test_architecture.py
## Type: HIGH - fixes test that doesn't actually verify
## Lines: 140-150 (replace)

OLD (Lines 140-144):
```python
    def test_no_rewrite_imports(self):
        """No imports from foodspec_rewrite should work."""
        with pytest.raises(ImportError):
            import foodspec_rewrite  # noqa: F401
```

NEW (Lines 140-155):
```python
    def test_no_rewrite_imports(self):
        """No imports from foodspec_rewrite in codebase."""
        repo_root = Path(__file__).parent.parent
        
        # Verify no rewrite imports in actual source code
        result = subprocess.run(
            ["grep", "-r", "from foodspec_rewrite", "src/", "tests/", "--include=*.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        
        assert result.returncode != 0, (
            f"ERROR: Found foodspec_rewrite imports in codebase:\n{result.stdout}"
        )
```

EXPLANATION:
The old test tried to import a non-existent package (which will always fail).
The new test actually searches the codebase to verify no imports from the old path exist.
This is the same check as in validate_architecture.py line 234.

---

## PATCH 3: scripts/refactor_executor.py - Manifest Serialization Bug (Lines 455-460)
## File: scripts/refactor_executor.py
## Type: CRITICAL - fixes manifest.json being invalid JSON
## Lines: 454-461 (replace function)

OLD (Lines 454-461):
```python
    def save_manifest(self, path: Path):
        """Save operation manifest to JSON."""
        manifest = {
            "timestamp": str(Path("/tmp").stat()),
            "operations": [op.to_dict() for op in self.operations],
            "success_count": sum(1 for op in self.operations if op.success or self.dry_run),
            "total_count": len(self.operations),
        }

        path.write_text(json.dumps(manifest, indent=2))
        self.log("success", f"Manifest saved to {path}")
```

NEW (Lines 454-464):
```python
    def save_manifest(self, path: Path):
        """Save operation manifest to JSON."""
        import time
        
        manifest = {
            "timestamp": time.time(),
            "operations": [op.to_dict() for op in self.operations],
            "success_count": sum(1 for op in self.operations if op.success),
            "total_count": len(self.operations),
        }

        path.write_text(json.dumps(manifest, indent=2))
        self.log("success", f"Manifest saved to {path}")
```

EXPLANATION:
Two issues fixed:
1. Path("/tmp").stat() returns a stat_result object, not a string. Changed to time.time() (epoch seconds)
2. Removed "or self.dry_run" from success_count - dry-run operations shouldn't count as success
This fixes --manifest-output flag and enables rollback functionality.

---

## PATCH 4: tests/test_architecture_ci.py - Class Name Typo (Line 60)
## File: tests/test_architecture_ci.py
## Type: MEDIUM - typo (test still runs but misleading name)
## Lines: 60

OLD (Line 60):
```python
class TestArtefactCreation:
```

NEW (Line 60):
```python
class TestArtifactCreation:
```

EXPLANATION:
Class name misspelled "Artefact" (British spelling) instead of "Artifact" (American).
Not blocking but inconsistent with naming conventions. Tests still run either way.

---

## PATCH 5: .github/workflows/architecture-enforce.yml - CI Integration Tests Not Blocking (Lines 100-103)
## File: .github/workflows/architecture-enforce.yml
## Type: MEDIUM - E2E tests don't block PR merge
## Lines: 100-103

OLD (Lines 100-103):
```yaml
    - name: Run CI integration tests
      run: |
        pip install pytest 2>&1 | tail -3 || true
        python -m pytest tests/test_architecture_ci.py -v --tb=short || true
        echo "⚠ CI integration tests completed (some may be optional)"
```

NEW (Lines 100-103):
```yaml
    - name: Run CI integration tests
      run: |
        pip install pytest 2>&1 | tail -3
        python -m pytest tests/test_architecture_ci.py -v --tb=short
        echo "✓ CI integration tests passed"
```

EXPLANATION:
The "|| true" at the end means test failures are ignored. Removed both instances so that
E2E test failures will block PR merge. If you want tests to be optional, keep as-is.
RECOMMENDATION: Apply patch to make them blocking.

---

# AUTOMATED PATCH SCRIPT

Save this as `apply_patches.sh` and run: `bash apply_patches.sh`

```bash
#!/bin/bash
set -e

echo "=== FoodSpec Refactor Patches ==="
echo "Applying 5 critical bugfixes..."
echo ""

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

# Patch 1: validate_architecture.py line 286
echo "✓ Patch 1/5: Fixing validate_architecture.py exit code..."
sed -i '286s/sys.exit(0 if all_passed else 0)/sys.exit(1 if (args.strict and not all_passed) else 0)/' \
  scripts/validate_architecture.py

# Patch 3: test_architecture_ci.py line 60
echo "✓ Patch 2/5: Fixing test class name typo..."
sed -i 's/class TestArtefactCreation:/class TestArtifactCreation:/' \
  tests/test_architecture_ci.py

# Patch 5: architecture-enforce.yml lines 100-103
echo "✓ Patch 3/5: Fixing CI enforcement (removing || true)..."
sed -i '100,103s/ || true//g' .github/workflows/architecture-enforce.yml

echo ""
echo "⚠ MANUAL PATCHES REQUIRED (2):"
echo ""
echo "Patch 4: tests/test_architecture.py - Lines 140-150"
echo "  Replace test_no_rewrite_imports() with actual codebase check"
echo "  See AUDIT_REFACTOR_DELIVERABLES.md PATCH 2 for exact code"
echo ""
echo "Patch 5: scripts/refactor_executor.py - Lines 454-464"  
echo "  Fix manifest timestamp serialization"
echo "  See AUDIT_REFACTOR_DELIVERABLES.md PATCH 3 for exact code"
echo ""

# Verify syntax
echo "Verifying Python syntax..."
python -m py_compile scripts/refactor_executor.py scripts/validate_architecture.py || {
  echo "ERROR: Syntax error detected. Review manual patches."
  exit 1
}

echo ""
echo "✓ Auto-patches applied successfully (3/5)"
echo "⚠ Please apply manual patches 4 and 5 (see instructions above)"
echo ""
echo "After manual patches, run:"
echo "  python scripts/validate_architecture.py --strict"
echo "  pytest tests/test_architecture.py -v"
```

---

# VERIFICATION CHECKLIST

After applying all patches, verify with:

```bash
# 1. Check syntax
python -m py_compile scripts/refactor_executor.py scripts/validate_architecture.py
echo "✓ Syntax OK"

# 2. Run validation script
python scripts/validate_architecture.py --strict
echo "✓ Validation OK"

# 3. Run unit tests (if environment allows)
pytest tests/test_architecture.py::TestImportPaths::test_no_rewrite_imports -v
echo "✓ test_no_rewrite_imports OK"

# 4. Verify manifest creation works
python scripts/refactor_executor.py --phase 1 --dry-run --manifest-output /tmp/test.json
python -c "import json; json.load(open('/tmp/test.json')); print('✓ Manifest JSON valid')"
```

---

END OF PATCH FILE
