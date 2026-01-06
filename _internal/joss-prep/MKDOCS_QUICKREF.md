# MkDocs Navigation Proposal - Quick Reference

## Navigation Comparison

### Current (96+ entries)
```
Home
├─ Examples
├─ Getting Started (6)
├─ Tutorials (9) ← Expand/collapse tree
├─ Workflows (9)
├─ Methods (18)
├─ User Guide (14) ← Very deep
├─ Theory (7)
├─ API Reference (11)
├─ Developer Guide (10) ← Internal
├─ Reference (10)
├─ Help & Support (5)
├─ 05-Advanced Topics (2) ← Internal
└─ ... (many collapsed)
```

### Proposed (48 entries)
```
Home
├─ Examples
├─ Getting Started (6)
├─ Workflows (9)
├─ Methods (21)
├─ API Reference (11)
├─ Theory (7)
└─ Help & Docs (12)
  └─ Includes: Troubleshooting, Reproducibility, Data Formats, Changelog, etc.

[Archived but searchable]
├─ Tutorials/ (still accessible via search)
├─ User Guide/ (still accessible via search)
└─ Developer Guide/ (still accessible via search)
```

---

## What Changes for Reviewers

### ✅ Better
- **Cleaner first impression:** 7 top sections vs 15
- **Clear path:** Getting Started → Workflows → Methods → Theory
- **Focus on examples:** Workflows section is prominent and organized
- **Reproducibility visible:** Dedicated "Help & Docs" section
- **API documented:** Separate section for code review

### ⚠️ Different (But Not Worse)
- Need to search for tutorials (e.g., "search: tutorials")
- CLI docs not in main nav (search: "CLI")
- Developer guide not visible (search: "contributing")

### ✅ Everything Still Works
- All links within docs still work
- All files still on disk
- Search finds all content
- No 404s

---

## Implementation (3 Minutes)

1. Open `mkdocs.yml`
2. Find `nav:` section (line ~163)
3. Replace with content from `MKDOCS_NAV_PASTE_THIS.md`
4. Run: `mkdocs build --strict`
5. If OK, run: `mkdocs serve` to test

---

## Rollback (1 Command)

```bash
git checkout mkdocs.yml
```

---

## Six Top Reviewer Entry Points

After Home/Examples, these are immediately visible:

| # | Section | Why | Example |
|---|---------|-----|---------|
| 1 | **Getting Started** | First thing reviewers do | Installation, 15-min quickstart |
| 2 | **Workflows** | "Does this work?" | Oil authentication complete example |
| 3 | **Methods** | "Is the science sound?" | Validation, statistics, preprocessing |
| 4 | **API Reference** | Code engineering review | Core modules, datasets, ML |
| 5 | **Theory** | Scientific foundation | Spectroscopy, chemometrics, FAIR |
| 6 | **Help & Docs** | Support + reproducibility | Troubleshooting, reproducibility checklist |

---

## Files Modified

Only: `mkdocs.yml` (nav section only)

## Files Kept

All docs remain intact in their current locations.

## Files Deleted

None.

---

## Key Design Principle

> **Show what matters to reviewers. Archive the rest but keep it searchable.**

---

## See Also

- `MKDOCS_NAV_PASTE_THIS.md` — Copy/paste the YAML
- `MKDOCS_IMPLEMENTATION_GUIDE.md` — Step-by-step instructions
- `MKDOCS_NAVIGATION_PROPOSAL.md` — Full design rationale
- `rollback_mkdocs_nav.sh` — Automated rollback script

