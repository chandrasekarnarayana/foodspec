"""PDF export functionality with graceful dependency handling.

Supports PDF generation from HTML with optional WeasyPrint support.
Falls back to HTML-only mode if WeasyPrint is not installed.

Usage:
    from foodspec.reporting.pdf import export_pdf, is_pdf_capable
    
    # Check if PDF export is available
    if is_pdf_capable():
        pdf_path = export_pdf("report.html", "report.pdf")
    else:
        print("PDF export not available - HTML export generated instead")
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

# Attempt to import WeasyPrint
try:
    import weasyprint

    HAS_WEASYPRINT = True
    WEASYPRINT_ERROR = None
except ImportError as e:
    HAS_WEASYPRINT = False
    WEASYPRINT_ERROR = str(e)


class PDFExportWarning(UserWarning):
    """Warning issued when PDF export falls back to HTML-only."""

    pass


def is_pdf_capable() -> bool:
    """Check if PDF export is available.
    
    Returns True if WeasyPrint is installed, False otherwise.
    
    Returns
    -------
    bool
        True if PDF export is available
        
    Examples
    --------
    >>> if is_pdf_capable():
    ...     print("PDF export available")
    ... else:
    ...     print("PDF export not available")
    """
    return HAS_WEASYPRINT


def get_pdf_capability_status() -> dict[str, bool | str]:
    """Get detailed PDF capability status.
    
    Returns
    -------
    dict
        Dictionary with 'available' (bool) and 'error' (str or None) keys
        
    Examples
    --------
    >>> status = get_pdf_capability_status()
    >>> print(f"PDF capable: {status['available']}")
    """
    return {
        "available": HAS_WEASYPRINT,
        "error": WEASYPRINT_ERROR,
    }


def export_pdf(
    html_path: str | Path,
    pdf_path: str | Path,
    warn_on_fallback: bool = True,
) -> Path:
    """Export HTML to PDF with graceful fallback.
    
    Attempts to generate PDF using WeasyPrint. If WeasyPrint is not
    installed, falls back to copying the HTML file with a warning.
    
    Parameters
    ----------
    html_path : str | Path
        Path to HTML file
    pdf_path : str | Path
        Output PDF file path
    warn_on_fallback : bool, default True
        Whether to warn if falling back to HTML-only mode
        
    Returns
    -------
    Path
        Path to output file (PDF if successful, HTML copy if fallback)
        
    Raises
    ------
    FileNotFoundError
        If HTML file doesn't exist
    IOError
        If file operations fail
        
    Notes
    -----
    When WeasyPrint is available:
    - Converts HTML to PDF with styling support
    - Generates true PDF file
    
    When WeasyPrint is not available:
    - Copies HTML file to PDF path (with .html or .pdf extension)
    - Issues PDFExportWarning
    - Still returns valid path
    
    Examples
    --------
    >>> output_path = export_pdf("report.html", "report.pdf")
    >>> print(f"Exported to: {output_path}")
    
    >>> # Check capability before exporting
    >>> if is_pdf_capable():
    ...     export_pdf("report.html", "report.pdf")
    ... else:
    ...     print("PDF export not available")
    """
    html_path = Path(html_path)
    pdf_path = Path(pdf_path)

    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    # Ensure output directory exists
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to export as PDF using WeasyPrint
    if HAS_WEASYPRINT:
        try:
            weasyprint.HTML(str(html_path)).write_pdf(str(pdf_path))
            return pdf_path.resolve()
        except Exception as e:
            # If WeasyPrint fails, fall back to HTML copy
            if warn_on_fallback:
                warnings.warn(
                    f"PDF export failed: {e}. Falling back to HTML-only export.",
                    PDFExportWarning,
                    stacklevel=2,
                )
            return _fallback_html_export(html_path, pdf_path)

    # WeasyPrint not available - fall back to HTML copy
    if warn_on_fallback:
        warnings.warn(
            "WeasyPrint not installed. PDF export not available. "
            "Install with: pip install weasyprint. "
            "Falling back to HTML export.",
            PDFExportWarning,
            stacklevel=2,
        )

    return _fallback_html_export(html_path, pdf_path)


def _fallback_html_export(html_path: Path, pdf_path: Path) -> Path:
    """Fallback: copy HTML file when PDF export unavailable.
    
    Parameters
    ----------
    html_path : Path
        Source HTML file
    pdf_path : Path
        Destination path
        
    Returns
    -------
    Path
        Path to output file (HTML copy)
    """
    import shutil

    # If output path has .pdf extension, keep it but content is HTML
    output_path = pdf_path

    # Copy HTML file to output path
    shutil.copy2(html_path, output_path)

    return output_path.resolve()


def get_pdf_export_message(
    mode: Literal["brief", "detailed"] = "brief",
) -> str:
    """Get informative message about PDF export capability.
    
    Parameters
    ----------
    mode : {'brief', 'detailed'}, default 'brief'
        Message detail level
        
    Returns
    -------
    str
        Informative message about PDF export status
        
    Examples
    --------
    >>> print(get_pdf_export_message())
    PDF export available
    
    >>> print(get_pdf_export_message("detailed"))
    PDF export not available. Install weasyprint: pip install weasyprint
    """
    status = get_pdf_capability_status()

    if status["available"]:
        if mode == "brief":
            return "PDF export available"
        else:
            return "PDF export available via WeasyPrint"
    else:
        if mode == "brief":
            return "PDF export not available"
        else:
            return (
                "PDF export not available. "
                "Install weasyprint: pip install weasyprint"
            )


class PDFExporter:
    """Batch PDF exporter with capability tracking.
    
    Handles multiple PDF exports with consistent error handling
    and capability checking.
    
    Examples
    --------
    >>> exporter = PDFExporter(warn_on_fallback=True)
    >>> 
    >>> # Check capability
    >>> if exporter.is_capable():
    ...     print("PDF export available")
    >>> 
    >>> # Export with consistent handling
    >>> for html_file in html_files:
    ...     pdf_path = exporter.export(html_file, f"{html_file.stem}.pdf")
    """

    def __init__(self, warn_on_fallback: bool = True) -> None:
        """Initialize exporter.
        
        Parameters
        ----------
        warn_on_fallback : bool, default True
            Whether to warn on fallback to HTML-only mode
        """
        self.warn_on_fallback = warn_on_fallback
        self.export_count = 0
        self.fallback_count = 0

    def is_capable(self) -> bool:
        """Check if PDF export is available.
        
        Returns
        -------
        bool
            True if PDF export is available
        """
        return HAS_WEASYPRINT

    def export(
        self,
        html_path: str | Path,
        pdf_path: str | Path,
    ) -> Path:
        """Export HTML to PDF.
        
        Parameters
        ----------
        html_path : str | Path
            Path to HTML file
        pdf_path : str | Path
            Output PDF path
            
        Returns
        -------
        Path
            Path to output file
        """
        self.export_count += 1
        if not self.is_capable():
            self.fallback_count += 1

        return export_pdf(
            html_path,
            pdf_path,
            warn_on_fallback=self.warn_on_fallback,
        )

    def get_stats(self) -> dict[str, int]:
        """Get export statistics.
        
        Returns
        -------
        dict
            Dictionary with export and fallback counts
        """
        return {
            "total_exports": self.export_count,
            "fallback_exports": self.fallback_count,
            "pdf_exports": self.export_count - self.fallback_count,
        }
