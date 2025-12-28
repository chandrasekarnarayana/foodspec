#!/usr/bin/env python3
"""
Documentation Link Checker for FoodSpec

Checks for:
- Broken internal links (missing markdown files)
- Broken image links (missing image files)
- Invalid anchor links (non-existent headings)
- Missing alt text on images

Usage:
    python scripts/check_docs_links.py
    python scripts/check_docs_links.py --check-anchors
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"
IGNORE_MD_DIRS = {
    DOCS_ROOT / "_internal",
}
IGNORE_MD_FILES = {
    DOCS_ROOT / "06-developer-guide" / "documentation_style_guide.md",
    DOCS_ROOT / "06-developer-guide" / "documentation_maintainer_guide.md",
}

link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def extract_headings(md_file):
    """Extract all headings from a markdown file as anchor slugs."""
    text = md_file.read_text(encoding="utf-8", errors="ignore")
    headings = []
    for match in heading_pattern.finditer(text):
        title = match.group(2).strip()
        # Convert to anchor slug (lowercase, spaces to hyphens, remove special chars)
        slug = re.sub(r"[^\w\s-]", "", title.lower())
        slug = re.sub(r"[-\s]+", "-", slug).strip("-")
        headings.append(slug)
    return headings


def check_links(check_anchors=False):
    """Check all markdown files for broken links."""
    missing_links = []
    missing_images = []
    missing_anchors = []
    missing_alt_text = []
    
    # Cache headings for anchor checking
    heading_cache = {}
    if check_anchors:
        print("Building heading cache for anchor validation...")
        for md in DOCS_ROOT.rglob("*.md"):
            heading_cache[md] = extract_headings(md)

    print(f"Checking {len(list(DOCS_ROOT.rglob('*.md')))} markdown files...\n")

    for md in DOCS_ROOT.rglob("*.md"):
        # Skip ignored directories/files
        try:
            md.relative_to(DOCS_ROOT)
        except ValueError:
            continue
        if any(md.is_relative_to(d) for d in IGNORE_MD_DIRS):
            continue
        if md in IGNORE_MD_FILES:
            continue
        rel_md = md.relative_to(DOCS_ROOT)
        text = md.read_text(encoding="utf-8", errors="ignore")
        
        # Check regular links
        for match in link_pattern.finditer(text):
            link_text = match.group(1)
            target = match.group(2)
            
            # Skip external links
            if target.startswith(("http://", "https://", "mailto:", "tel:", "ftp://")):
                continue
            
            # Split anchor from path
            if "#" in target:
                path_part, anchor_part = target.split("#", 1)
            else:
                path_part, anchor_part = target, None
            
            # Skip empty paths (same-page anchors)
            if not path_part:
                if check_anchors and anchor_part:
                    # Check anchor exists in current file
                    if anchor_part not in heading_cache.get(md, []):
                        missing_anchors.append((str(rel_md), target, f"Anchor #{anchor_part} not found in current file"))
                continue
            
            # Resolve target path
            target_path = (md.parent / path_part).resolve()
            
            # Check if target is within docs/
            try:
                target_path.relative_to(DOCS_ROOT)
            except ValueError:
                continue
            
            # Check if file exists
            if not target_path.exists():
                missing_links.append((str(rel_md), target))
            elif check_anchors and anchor_part:
                # Check if anchor exists in target file
                if anchor_part not in heading_cache.get(target_path, []):
                    missing_anchors.append((str(rel_md), target, f"Anchor #{anchor_part} not found in {target_path.relative_to(DOCS_ROOT)}"))
        
        # Check images
        for match in image_pattern.finditer(text):
            alt_text = match.group(1)
            target = match.group(2)
            
            # Check for missing alt text
            if not alt_text.strip():
                missing_alt_text.append((str(rel_md), target))
            
            # Skip external images
            if target.startswith(("http://", "https://", "data:", "//")):
                continue
            
            # Resolve image path
            target_path = (md.parent / target).resolve()
            
            # Check if image is within docs/
            try:
                target_path.relative_to(DOCS_ROOT)
            except ValueError:
                continue
            
            # Check if image exists
            if not target_path.exists():
                missing_images.append((str(rel_md), target))

    return missing_links, missing_images, missing_anchors, missing_alt_text


def print_results(missing_links, missing_images, missing_anchors, missing_alt_text):
    """Print check results."""
    errors = 0
    warnings = 0
    
    if missing_links:
        print("❌ MISSING INTERNAL LINKS (relative to docs/):")
        for src, tgt in sorted(missing_links):
            print(f"   {src} -> {tgt}")
        errors += len(missing_links)
        print()
    
    if missing_images:
        print("❌ MISSING IMAGES (relative to docs/):")
        for src, tgt in sorted(missing_images):
            print(f"   {src} -> {tgt}")
        errors += len(missing_images)
        print()
    
    if missing_anchors:
        print("❌ INVALID ANCHOR LINKS:")
        for src, tgt, reason in sorted(missing_anchors):
            print(f"   {src} -> {tgt}")
            print(f"      {reason}")
        errors += len(missing_anchors)
        print()
    
    if missing_alt_text:
        print("⚠️  MISSING ALT TEXT (accessibility issue):")
        for src, tgt in sorted(missing_alt_text):
            print(f"   {src} -> {tgt}")
        warnings += len(missing_alt_text)
        print()
    
    # Summary
    print("=" * 70)
    if errors == 0 and warnings == 0:
        print("✅ ALL CHECKS PASSED!")
        print(f"   No broken links or missing images found.")
    else:
        if errors > 0:
            print(f"❌ ERRORS: {errors}")
        if warnings > 0:
            print(f"⚠️  WARNINGS: {warnings}")
    print("=" * 70)
    
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Check FoodSpec documentation for broken links and images"
    )
    parser.add_argument(
        "--check-anchors",
        action="store_true",
        help="Also validate anchor links (slower)"
    )
    args = parser.parse_args()
    
    missing_links, missing_images, missing_anchors, missing_alt_text = check_links(
        check_anchors=args.check_anchors
    )
    
    errors = print_results(missing_links, missing_images, missing_anchors, missing_alt_text)
    
    # Exit with error code if any errors found
    sys.exit(1 if errors > 0 else 0)


if __name__ == "__main__":
    main()
