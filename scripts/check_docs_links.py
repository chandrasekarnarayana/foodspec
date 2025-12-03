import re
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"

link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

missing_links = []
missing_images = []

for md in DOCS_ROOT.rglob("*.md"):
    rel_md = md.relative_to(DOCS_ROOT)
    text = md.read_text(encoding="utf-8", errors="ignore")
    for pat, bucket in ((link_pattern, missing_links), (image_pattern, missing_images)):
        for match in pat.finditer(text):
            target = match.group(1).split("#")[0]
            if target.startswith("http") or target.startswith("mailto") or target.startswith("tel"):
                continue
            target_path = (md.parent / target).resolve()
            try:
                target_path.relative_to(DOCS_ROOT)
            except ValueError:
                # outside docs/; skip
                continue
            if not target_path.exists():
                bucket.append((str(rel_md), target))

print("Missing internal links (relative to docs/):")
for src, tgt in missing_links:
    print(f"  {src} -> {tgt}")

print("\nMissing images (relative to docs/):")
for src, tgt in missing_images:
    print(f"  {src} -> {tgt}")

if not missing_links and not missing_images:
    print("\nNo missing links or images detected.")
