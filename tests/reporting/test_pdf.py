"""Tests for PDF export functionality."""

import warnings
from pathlib import Path
from unittest import mock

import pytest

from foodspec.reporting.pdf import (
    PDFExporter,
    PDFExportWarning,
    export_pdf,
    get_pdf_capability_status,
    get_pdf_export_message,
    is_pdf_capable,
)


@pytest.fixture
def temp_html_file(tmp_path):
    """Create a temporary HTML file."""
    html_file = tmp_path / "report.html"
    html_file.write_text(
        """
        <html>
            <head><title>Test Report</title></head>
            <body>
                <h1>Test Report</h1>
                <p>This is a test HTML file.</p>
                <table>
                    <tr><th>Header 1</th><th>Header 2</th></tr>
                    <tr><td>Data 1</td><td>Data 2</td></tr>
                </table>
            </body>
        </html>
    """
    )
    return html_file


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


class TestCapabilityChecking:
    """Test PDF capability detection."""

    def test_is_pdf_capable_returns_bool(self):
        """Test is_pdf_capable returns boolean."""
        result = is_pdf_capable()
        assert isinstance(result, bool)

    def test_get_pdf_capability_status_structure(self):
        """Test capability status has correct structure."""
        status = get_pdf_capability_status()
        assert isinstance(status, dict)
        assert "available" in status
        assert "error" in status
        assert isinstance(status["available"], bool)

    def test_capability_consistency(self):
        """Test capability status is consistent."""
        is_capable = is_pdf_capable()
        status = get_pdf_capability_status()
        assert is_capable == status["available"]

    def test_error_field_logic(self):
        """Test error field is None when available."""
        status = get_pdf_capability_status()
        if status["available"]:
            # If available, error should be None
            assert status["error"] is None
        else:
            # If not available, error should be a string
            assert isinstance(status["error"], str)


class TestPDFExportMessage:
    """Test PDF export message generation."""

    def test_brief_message_available(self):
        """Test brief message when PDF capable."""
        message = get_pdf_export_message("brief")
        assert isinstance(message, str)
        assert len(message) > 0

    def test_detailed_message_available(self):
        """Test detailed message when PDF capable."""
        message = get_pdf_export_message("detailed")
        assert isinstance(message, str)
        assert len(message) > 0

    def test_detailed_longer_than_brief(self):
        """Test detailed message is longer than brief."""
        brief = get_pdf_export_message("brief")
        detailed = get_pdf_export_message("detailed")
        assert len(detailed) >= len(brief)

    def test_weasyprint_mention_if_not_capable(self):
        """Test message mentions WeasyPrint when not available."""
        status = get_pdf_capability_status()
        detailed = get_pdf_export_message("detailed")

        if not status["available"]:
            assert "weasyprint" in detailed.lower()


class TestExportPDFWithWeasyPrint:
    """Test PDF export when WeasyPrint is available."""

    @pytest.mark.skipif(
        not is_pdf_capable(), reason="WeasyPrint not installed"
    )
    def test_export_pdf_creates_file(self, temp_html_file, temp_output_dir):
        """Test export_pdf creates output file."""
        pdf_path = temp_output_dir / "report.pdf"
        result = export_pdf(temp_html_file, pdf_path)

        assert result.exists()
        assert result.name == "report.pdf"

    @pytest.mark.skipif(
        not is_pdf_capable(), reason="WeasyPrint not installed"
    )
    def test_export_pdf_returns_path(
        self, temp_html_file, temp_output_dir
    ):
        """Test export_pdf returns resolved path."""
        pdf_path = temp_output_dir / "report.pdf"
        result = export_pdf(temp_html_file, pdf_path)

        assert isinstance(result, Path)
        assert result.is_absolute()

    @pytest.mark.skipif(
        not is_pdf_capable(), reason="WeasyPrint not installed"
    )
    def test_export_pdf_creates_parent_dirs(
        self, temp_html_file, tmp_path
    ):
        """Test export_pdf creates parent directories."""
        pdf_path = (
            tmp_path
            / "deep"
            / "nested"
            / "directory"
            / "structure"
            / "report.pdf"
        )
        result = export_pdf(temp_html_file, pdf_path)

        assert result.exists()
        assert result.parent.exists()


class TestExportPDFWithoutWeasyPrint:
    """Test PDF export graceful fallback."""

    def test_export_pdf_missing_file_raises_error(self, temp_output_dir):
        """Test export_pdf raises error for missing HTML file."""
        with pytest.raises(FileNotFoundError):
            export_pdf(
                temp_output_dir / "nonexistent.html",
                temp_output_dir / "report.pdf",
            )

    def test_export_pdf_fallback_warns(self, temp_html_file, temp_output_dir):
        """Test export_pdf warns on fallback."""
        # Mock WeasyPrint to be unavailable
        with mock.patch(
            "foodspec.reporting.pdf.HAS_WEASYPRINT", False
        ):
            pdf_path = temp_output_dir / "report.pdf"

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                export_pdf(temp_html_file, pdf_path)

                assert len(w) == 1
                assert issubclass(w[0].category, PDFExportWarning)
                assert "WeasyPrint" in str(w[0].message)

    def test_export_pdf_fallback_no_warn_if_disabled(
        self, temp_html_file, temp_output_dir
    ):
        """Test export_pdf doesn't warn when warn_on_fallback=False."""
        with mock.patch(
            "foodspec.reporting.pdf.HAS_WEASYPRINT", False
        ):
            pdf_path = temp_output_dir / "report.pdf"

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                export_pdf(
                    temp_html_file, pdf_path, warn_on_fallback=False
                )

                assert len(w) == 0

    def test_export_pdf_fallback_creates_file(
        self, temp_html_file, temp_output_dir
    ):
        """Test export_pdf creates file on fallback."""
        with mock.patch(
            "foodspec.reporting.pdf.HAS_WEASYPRINT", False
        ):
            pdf_path = temp_output_dir / "report.pdf"

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = export_pdf(temp_html_file, pdf_path)

            assert result.exists()

    def test_export_pdf_fallback_preserves_content(
        self, temp_html_file, temp_output_dir
    ):
        """Test export_pdf preserves HTML content on fallback."""
        with mock.patch(
            "foodspec.reporting.pdf.HAS_WEASYPRINT", False
        ):
            pdf_path = temp_output_dir / "report.pdf"

            original_content = temp_html_file.read_text()

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                export_pdf(temp_html_file, pdf_path)

            result_content = pdf_path.read_text()
            assert original_content == result_content


class TestPDFExporter:
    """Test PDFExporter class."""

    def test_exporter_initialization(self):
        """Test PDFExporter initializes correctly."""
        exporter = PDFExporter()
        assert exporter.warn_on_fallback is True
        assert exporter.export_count == 0
        assert exporter.fallback_count == 0

    def test_exporter_initialization_with_warn_false(self):
        """Test PDFExporter initializes with warn_on_fallback=False."""
        exporter = PDFExporter(warn_on_fallback=False)
        assert exporter.warn_on_fallback is False

    def test_exporter_is_capable(self):
        """Test exporter is_capable matches global check."""
        exporter = PDFExporter()
        assert exporter.is_capable() == is_pdf_capable()

    def test_exporter_export_updates_count(
        self, temp_html_file, temp_output_dir
    ):
        """Test export updates export count."""
        exporter = PDFExporter()
        assert exporter.export_count == 0

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            exporter.export(
                temp_html_file, temp_output_dir / "report1.pdf"
            )

        assert exporter.export_count == 1

    def test_exporter_multiple_exports(
        self, temp_html_file, temp_output_dir
    ):
        """Test exporter handles multiple exports."""
        exporter = PDFExporter()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            for i in range(3):
                exporter.export(
                    temp_html_file,
                    temp_output_dir / f"report{i}.pdf",
                )

        assert exporter.export_count == 3

    def test_exporter_get_stats(self, temp_html_file, temp_output_dir):
        """Test exporter statistics."""
        exporter = PDFExporter()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            exporter.export(temp_html_file, temp_output_dir / "report.pdf")

        stats = exporter.get_stats()
        assert "total_exports" in stats
        assert "fallback_exports" in stats
        assert "pdf_exports" in stats
        assert stats["total_exports"] == 1

    def test_exporter_stats_consistent(self, temp_html_file, temp_output_dir):
        """Test exporter statistics are consistent."""
        exporter = PDFExporter()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            for i in range(5):
                exporter.export(
                    temp_html_file,
                    temp_output_dir / f"report{i}.pdf",
                )

        stats = exporter.get_stats()
        assert (
            stats["pdf_exports"] + stats["fallback_exports"]
            == stats["total_exports"]
        )


class TestPDFExportWarning:
    """Test PDF export warning class."""

    def test_warning_is_user_warning(self):
        """Test PDFExportWarning is a UserWarning."""
        assert issubclass(PDFExportWarning, UserWarning)

    def test_warning_can_be_caught(self):
        """Test PDFExportWarning can be caught."""
        with mock.patch(
            "foodspec.reporting.pdf.HAS_WEASYPRINT", False
        ):
            pdf_path_obj = Path("/tmp/test.pdf")
            html_path_obj = Path("/tmp/test.html")

            # Create HTML file for test
            html_path_obj.write_text("<html><body>Test</body></html>")

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                export_pdf(html_path_obj, pdf_path_obj)

                assert any(
                    issubclass(warning.category, PDFExportWarning)
                    for warning in w
                )


class TestIntegration:
    """Integration tests for PDF export."""

    def test_export_workflow_html_file(self, temp_html_file, temp_output_dir):
        """Test complete export workflow."""
        pdf_path = temp_output_dir / "workflow_report.pdf"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = export_pdf(temp_html_file, pdf_path)

        assert result.exists()
        assert result.is_absolute()

    def test_export_with_exporter_class(
        self, temp_html_file, temp_output_dir
    ):
        """Test complete workflow using PDFExporter class."""
        exporter = PDFExporter()

        # Check capability
        if exporter.is_capable():
            pdf_path = temp_output_dir / "exporter_report.pdf"
            result = exporter.export(temp_html_file, pdf_path)
            assert result.exists()

    def test_batch_export_multiple_formats(
        self, tmp_path
    ):
        """Test batch export with mixed file types."""
        # Create multiple HTML files
        html_files = []
        for i in range(3):
            html_file = tmp_path / f"report_{i}.html"
            html_file.write_text(
                f"<html><body><h1>Report {i}</h1></body></html>"
            )
            html_files.append(html_file)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            for html_file in html_files:
                export_pdf(
                    html_file,
                    output_dir / html_file.with_suffix(".pdf").name,
                )

        # Verify all outputs were created
        assert len(list(output_dir.glob("*.pdf"))) == 3

    def test_export_capabilities_consistent_across_calls(
        self, temp_html_file, temp_output_dir
    ):
        """Test PDF capability is consistent across multiple calls."""
        capability_1 = is_pdf_capable()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            export_pdf(temp_html_file, temp_output_dir / "report1.pdf")

        capability_2 = is_pdf_capable()

        assert capability_1 == capability_2
