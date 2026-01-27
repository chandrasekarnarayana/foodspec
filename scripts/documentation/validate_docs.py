#!/usr/bin/env python3
"""
Documentation Validation Suite for FoodSpec

Runs comprehensive checks on documentation:
1. Markdown linting (if markdownlint-cli is installed)
2. Link checking (broken links, images, anchors)
3. MkDocs build validation
4. Style checks (heading hierarchy, code block language tags)

Usage:
    python scripts/validate_docs.py
    python scripts/validate_docs.py --full  # Includes slow checks (anchors, external links)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_command(cmd, description, required=False):
    """Run a shell command and return success status."""
    print(f"\n{'=' * 70}")
    print(f"🔍 {description}")
    print(f"{'=' * 70}")

    try:
        result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True, cwd=PROJECT_ROOT)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            if required:
                print("   This check is required for documentation quality.")
            return False

    except Exception as e:
        print(f"⚠️  Could not run: {e}")
        if not required:
            print("   (Optional check, skipping)")
        return not required


def check_markdownlint():
    """Run markdownlint if available."""
    # Check if markdownlint-cli is installed
    check_installed = subprocess.run("which markdownlint", shell=True, capture_output=True)

    if check_installed.returncode != 0:
        print("\n" + "=" * 70)
        print("⚠️  Markdownlint not installed (optional check)")
        print("=" * 70)
        print("To install: npm install -g markdownlint-cli")
        print("Skipping markdown linting...")
        return True  # Not required

    return run_command(
        "markdownlint docs/**/*.md --config .markdownlint.json", "Markdown Linting (markdownlint)", required=False
    )


def check_links(full=False):
    """Run link checker."""
    cmd = "python scripts/check_docs_links.py"
    if full:
        cmd += " --check-anchors"

    return run_command(cmd, "Link Validation (check_docs_links.py)", required=True)


def check_mkdocs_build():
    """Run mkdocs build and fail on warnings."""
    print(f"\n{'=' * 70}")
    print("🔍 MkDocs Build Validation")
    print(f"{'=' * 70}")

    try:
        result = subprocess.run(
            "mkdocs build --strict",
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )

        output = result.stdout + result.stderr
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # Check for common warning patterns that should be errors
        warning_patterns = [
            r"INFO: The following pages exist in docs directory but are not included in nav",
            r"doc file.*contains a link.*but the doc.*does not contain an anchor",
            r"WARNING",
            r"ERROR",
        ]

        has_warnings = any(re.search(pattern, output, re.IGNORECASE) for pattern in warning_patterns)

        if result.returncode == 0 and not has_warnings:
            print("✅ MkDocs Build Validation - PASSED")
            return True
        else:
            if has_warnings:
                print("❌ MkDocs Build Validation - FAILED (warnings found)")
            else:
                print("❌ MkDocs Build Validation - FAILED (build error)")
            print("   This check is required for documentation quality.")
            return False

    except Exception as e:
        print(f"❌ Could not run mkdocs build: {e}")
        return False


def check_style_issues():
    """Check for common style issues."""
    print(f"\n{'=' * 70}")
    print("🔍 Style Validation (custom checks)")
    print(f"{'=' * 70}")

    issues = []

    # Pattern checks
    code_block_no_lang = re.compile(r"^```\s*$", re.MULTILINE)
    heading_with_period = re.compile(r"^#+\s+.+\.\s*$", re.MULTILINE)

    for md_file in DOCS_ROOT.rglob("*.md"):
        rel_path = md_file.relative_to(DOCS_ROOT)
        content = md_file.read_text(encoding="utf-8", errors="ignore")

        # Check for code blocks without language tags
        if code_block_no_lang.search(content):
            issues.append(f"   {rel_path}: Code block(s) without language tag")

        # Check for headings with trailing periods
        if heading_with_period.search(content):
            issues.append(f"   {rel_path}: Heading(s) with trailing period")

    if issues:
        print("⚠️  Style issues found:")
        for issue in issues[:20]:  # Limit to first 20
            print(issue)
        if len(issues) > 20:
            print(f"   ... and {len(issues) - 20} more")
        print("\nThese are warnings, not errors.")
        return True  # Don't fail on style warnings
    else:
        print("✅ No style issues found")
        return True


def print_summary(results):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for check, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {check}")

    print("=" * 70)
    if passed == total:
        print(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
        print("Documentation is ready for publication!")
    else:
        failed = total - passed
        print(f"❌ SOME CHECKS FAILED ({failed} failed, {passed} passed)")
        print("Please fix the errors above before submitting.")
    print("=" * 70)

    return passed == total


def main():
    parser = argparse.ArgumentParser(description="Validate FoodSpec documentation quality")
    parser.add_argument("--full", action="store_true", help="Run all checks including slow ones (anchor validation)")
    parser.add_argument("--skip-build", action="store_true", help="Skip mkdocs build (faster, for quick checks)")
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 FoodSpec Documentation Validation Suite")
    print("=" * 70)
    print(f"Documentation root: {DOCS_ROOT}")
    print(f"Full validation: {args.full}")
    print()

    # Run checks
    results = {}

    # 1. Markdown linting (optional)
    results["Markdownlint"] = check_markdownlint()

    # 2. Link checking (required)
    results["Link Validation"] = check_links(full=args.full)

    # 3. MkDocs build (required unless skipped)
    if not args.skip_build:
        results["MkDocs Build"] = check_mkdocs_build()

    # 4. Style checks (warnings only)
    results["Style Checks"] = check_style_issues()

    # Print summary
    all_passed = print_summary(results)

    # Exit code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
