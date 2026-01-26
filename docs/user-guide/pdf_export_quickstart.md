# PDF Export Quick Start

Export HTML reports to PDF with automatic fallback when WeasyPrint is unavailable.

## 30 Seconds

```python
from foodspec.reporting.pdf import export_pdf

# One line - works with or without WeasyPrint
pdf_path = export_pdf("report.html", "report.pdf")
```

Done! You have a PDF (or HTML if WeasyPrint not installed).

## Check Capability

```python
from foodspec.reporting.pdf import is_pdf_capable

if is_pdf_capable():
    print("✓ PDF export available")
else:
    print("⚠ PDF export unavailable - HTML fallback will be used")
```

## Enable PDF Export

PDF export is optional. To enable it:

```bash
pip install weasyprint
```

Then re-run your export - it will generate true PDFs.

## Common Workflows

### Export with Warnings

```python
from foodspec.reporting.pdf import export_pdf

# Warns if WeasyPrint missing
pdf = export_pdf("report.html", "report.pdf")
```

### Export Silently

```python
from foodspec.reporting.pdf import export_pdf

# No warnings, even if falling back
pdf = export_pdf("report.html", "report.pdf", warn_on_fallback=False)
```

### Batch Export

```python
from pathlib import Path
from foodspec.reporting.pdf import PDFExporter

exporter = PDFExporter()

for html_file in Path("reports").glob("*.html"):
    exporter.export(html_file, html_file.with_suffix(".pdf"))

# Get statistics
stats = exporter.get_stats()
print(f"Exported {stats['pdf_exports']} PDFs, {stats['fallback_exports']} HTML")
```

### Check Before Exporting

```python
from foodspec.reporting.pdf import is_pdf_capable, export_pdf

html_file = "report.html"

if is_pdf_capable():
    pdf = export_pdf(html_file, "report.pdf")
    print("✓ PDF generated")
else:
    print("⚠ PDF export unavailable")
    print("  To enable: pip install weasyprint")
```

## How It Works

### With WeasyPrint ✓

```
report.html → [WeasyPrint] → report.pdf
                              (true PDF)
```

### Without WeasyPrint ⚠

```
report.html → [Copy file] → report.pdf
                           (actually HTML)
                           + ⚠ Warning issued
```

Then users can:
- Install WeasyPrint and re-export
- Open HTML in browser and print to PDF
- Use the HTML directly

## Understanding Messages

### "PDF export available"
✓ WeasyPrint is installed. PDFs will be generated.

### "PDF export not available"
⚠ WeasyPrint not installed. HTML fallback will be used.
   Install with: `pip install weasyprint`

## Key Points

✨ **Never Crashes** - Missing WeasyPrint doesn't break anything
✨ **Clear Warnings** - Users always know what happened
✨ **Optional** - Works fine without WeasyPrint
✨ **Simple API** - One function: `export_pdf()`

## API

```python
export_pdf(
    html_path,           # Input HTML file
    pdf_path,           # Output PDF path
    warn_on_fallback=True  # Warn if falling back
) -> Path               # Returns output file path
```

## Tips

1. **Check availability first:**
   ```python
   if is_pdf_capable():
       # PDF will be generated
   ```

2. **Silent exports for scripts:**
   ```python
   export_pdf(file, out, warn_on_fallback=False)
   ```

3. **Batch processing:**
   ```python
   exporter = PDFExporter()
   for html in html_files:
       exporter.export(html, f"{html.stem}.pdf")
   ```

4. **Monitor exports:**
   ```python
   stats = exporter.get_stats()
   print(f"PDF: {stats['pdf_exports']}, HTML: {stats['fallback_exports']}")
   ```

## Next Steps

- 📖 Read [full documentation](./pdf_export.md)
- 🔬 Run [demo script](../../examples/new-features/pdf_export_demo.py)
- ✅ Check [test examples](../../tests/reporting/test_pdf.py)

---

**PDF export is optional but clean** - always works, with or without WeasyPrint.
