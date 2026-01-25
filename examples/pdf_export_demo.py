"""Demo of PDF export functionality with graceful fallback.

Demonstrates PDF export with WeasyPrint support and fallback handling.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from foodspec.reporting.pdf import (
    PDFExporter,
    export_pdf,
    get_pdf_capability_status,
    get_pdf_export_message,
    is_pdf_capable,
)


def create_demo_html(html_path: Path) -> None:
    """Create a demo HTML report."""
    html_path.write_text(
        """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FoodSpec Analysis Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }
            .header {
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
            }
            .section {
                background-color: white;
                padding: 20px;
                margin-bottom: 20px;
                border-left: 4px solid #3498db;
                border-radius: 3px;
            }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; margin-top: 0; }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            th {
                background-color: #3498db;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .metric {
                display: inline-block;
                margin-right: 30px;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #27ae60;
            }
            .warning {
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 3px;
            }
            footer {
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                font-size: 12px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🥗 FoodSpec Analysis Report</h1>
            <p>Oil Authentication Analysis - 2024</p>
        </div>

        <div class="section">
            <h2>Executive Summary</h2>
            <p>
                This report presents the results of a comprehensive spectral analysis
                of olive oil samples for authentication and quality assessment.
            </p>
            <div class="metric">
                <div>Samples Analyzed</div>
                <div class="metric-value">250</div>
            </div>
            <div class="metric">
                <div>Model Accuracy</div>
                <div class="metric-value">95.4%</div>
            </div>
            <div class="metric">
                <div>Confidence</div>
                <div class="metric-value">98.9%</div>
            </div>
        </div>

        <div class="section">
            <h2>Methodology</h2>
            <p>
                Samples were analyzed using visible-near infrared spectroscopy
                with the following preprocessing steps:
            </p>
            <ul>
                <li>Standard Normal Variate (SNV) normalization</li>
                <li>Baseline removal using polynomial fitting (order 3)</li>
                <li>Partial Least Squares Discriminant Analysis (PLS-DA)</li>
            </ul>
        </div>

        <div class="section">
            <h2>Key Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Classification Accuracy</td>
                        <td>95.4%</td>
                        <td>✓ Excellent</td>
                    </tr>
                    <tr>
                        <td>Sensitivity</td>
                        <td>94.8%</td>
                        <td>✓ Good</td>
                    </tr>
                    <tr>
                        <td>Specificity</td>
                        <td>95.1%</td>
                        <td>✓ Good</td>
                    </tr>
                    <tr>
                        <td>ROC-AUC Score</td>
                        <td>0.989</td>
                        <td>✓ Excellent</td>
                    </tr>
                    <tr>
                        <td>F1 Score</td>
                        <td>0.950</td>
                        <td>✓ Excellent</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Conclusions</h2>
            <p>
                The spectral analysis successfully discriminated between authentic
                and counterfeit olive oils with high accuracy. The model is suitable
                for deployment in quality control workflows.
            </p>
            <p>
                <strong>Recommendations:</strong>
            </p>
            <ul>
                <li>Deploy model for rapid authentication screening</li>
                <li>Conduct periodic model retraining with new samples</li>
                <li>Implement confidence thresholding for borderline cases</li>
            </ul>
        </div>

        <footer>
            <p>Generated by FoodSpec - Spectral Analysis Framework</p>
            <p>Report Date: 2024-01-25</p>
        </footer>
    </body>
    </html>
    """
    )


def main() -> None:
    """Run demo."""
    print("=" * 70)
    print("FoodSpec PDF Export Demo")
    print("=" * 70)

    # Step 1: Check PDF capability
    print("\n1. Checking PDF export capability...")
    status = get_pdf_capability_status()
    message = get_pdf_export_message("detailed")

    print(f"   ✓ {message}")
    print(f"   ✓ Status: {status}")

    # Step 2: Create demo HTML
    print("\n2. Creating demo HTML report...")
    demo_dir = Path("demo_pdf_export")
    demo_dir.mkdir(exist_ok=True)

    html_path = demo_dir / "report.html"
    create_demo_html(html_path)
    print(f"   ✓ HTML report created: {html_path}")
    print(f"   ✓ File size: {html_path.stat().st_size} bytes")

    # Step 3: Export to PDF (Option A or B)
    print("\n3. Exporting to PDF...")
    pdf_path = demo_dir / "report.pdf"

    if is_pdf_capable():
        print("   ✓ WeasyPrint is installed - generating true PDF")
    else:
        print("   ⚠ WeasyPrint not installed - will fallback to HTML export")

    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = export_pdf(html_path, pdf_path)

            if w:
                for warning in w:
                    print(f"   ⚠ {warning.message}")
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        return

    print(f"   ✓ Export successful: {result}")
    print(f"   ✓ File size: {result.stat().st_size} bytes")

    # Step 4: Batch export with PDFExporter
    print("\n4. Batch export with PDFExporter...")
    exporter = PDFExporter(warn_on_fallback=False)

    html_files = []
    for i in range(3):
        html_file = demo_dir / f"report_{i}.html"
        create_demo_html(html_file)
        html_files.append(html_file)

    for i, html_file in enumerate(html_files):
        pdf_file = demo_dir / f"report_{i}.pdf"
        exporter.export(html_file, pdf_file)
        print(f"   ✓ Exported report_{i}: {pdf_file.stat().st_size} bytes")

    # Step 5: Display statistics
    print("\n5. Export Statistics")
    stats = exporter.get_stats()
    print(f"   • Total exports: {stats['total_exports']}")
    print(f"   • PDF exports: {stats['pdf_exports']}")
    print(f"   • Fallback exports: {stats['fallback_exports']}")

    # Step 6: Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    files = list(demo_dir.glob("*.pdf")) + list(demo_dir.glob("*.html"))
    print(f"\n✓ Generated {len(files)} files in {demo_dir}")

    total_size = sum(f.stat().st_size for f in files)
    print(f"✓ Total size: {total_size:,} bytes")

    if is_pdf_capable():
        print("\n✓ PDF export is FULLY FUNCTIONAL")
    else:
        print("\n⚠ PDF export is in FALLBACK MODE (HTML-only)")
        print("  To enable PDF export, install WeasyPrint:")
        print("    pip install weasyprint")

    print("\n✓ All exports completed successfully!")


if __name__ == "__main__":
    main()
