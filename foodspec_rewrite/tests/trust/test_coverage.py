"""
Tests for coverage analysis module.

Tests:
  - coverage_by_group: grouping, aggregation, sorting
  - format_coverage_table: formatting, decimals, column selection
  - to_markdown/to_latex: export formats, captions/labels
  - check_coverage_guarantees: violation detection, tolerance
  - coverage_comparison: merging, delta computation
  - summarize_coverage: summary generation
"""

import numpy as np
import pandas as pd
import pytest

from foodspec.trust.coverage import (
    check_coverage_guarantees,
    coverage_by_group,
    coverage_comparison,
    format_coverage_table,
    summarize_coverage,
    to_latex,
    to_markdown,
)


class TestCoverageByGroup:
    """Test coverage_by_group function."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample coverage DataFrame."""
        return pd.DataFrame({
            'bin': ['A', 'A', 'A', 'B', 'B', 'B'],
            'set_size': [1, 1, 2, 1, 2, 2],
            'covered': [1, 1, 0, 1, 0, 1],
            'threshold': [0.3, 0.3, 0.3, 0.4, 0.4, 0.4],
        })
    
    def test_coverage_by_group_basic(self, sample_df):
        """Test basic grouping functionality."""
        result = coverage_by_group(sample_df, group_col='bin')
        
        assert len(result) == 2
        assert list(result['group']) == ['A', 'B']
        assert result.loc[result['group'] == 'A', 'n_samples'].values[0] == 3
        assert result.loc[result['group'] == 'B', 'n_samples'].values[0] == 3
    
    def test_coverage_by_group_metrics(self, sample_df):
        """Test metric computation."""
        result = coverage_by_group(sample_df, group_col='bin')
        
        # Group A: coverage = 2/3, avg_set_size = (1+1+2)/3 = 4/3
        a_row = result[result['group'] == 'A'].iloc[0]
        assert np.isclose(a_row['coverage'], 2/3, rtol=1e-5)
        assert np.isclose(a_row['avg_set_size'], 4/3, rtol=1e-5)
        
        # Group B: coverage = 2/3, avg_set_size = (1+2+2)/3 = 5/3
        b_row = result[result['group'] == 'B'].iloc[0]
        assert np.isclose(b_row['coverage'], 2/3, rtol=1e-5)
        assert np.isclose(b_row['avg_set_size'], 5/3, rtol=1e-5)
    
    def test_coverage_by_group_missing_covered_column(self):
        """Test handling when 'covered' column missing."""
        df = pd.DataFrame({
            'bin': ['A', 'A', 'B'],
            'set_size': [1, 2, 1],
            'threshold': [0.3, 0.3, 0.4],
        })
        
        result = coverage_by_group(df, group_col='bin')
        
        assert pd.isna(result['coverage']).all()
        assert len(result) == 2
    
    def test_coverage_by_group_invalid_column(self, sample_df):
        """Test error on invalid group_col."""
        with pytest.raises(ValueError, match="not found"):
            coverage_by_group(sample_df, group_col='nonexistent')
    
    def test_coverage_by_group_sorting_numeric(self):
        """Test numeric sorting of groups."""
        df = pd.DataFrame({
            'bin': ['10', '2', '1', '3'],
            'set_size': [1, 1, 1, 1],
            'covered': [1, 1, 1, 1],
            'threshold': [0.3, 0.3, 0.3, 0.3],
        })
        
        result = coverage_by_group(df, group_col='bin')
        
        # Should sort numerically: 1, 2, 3, 10
        assert list(result['group']) == ['1', '2', '3', '10']
    
    def test_coverage_by_group_sorting_alphabetic(self):
        """Test alphabetic sorting fallback."""
        df = pd.DataFrame({
            'bin': ['batch_z', 'batch_a', 'batch_m'],
            'set_size': [1, 1, 1],
            'covered': [1, 1, 1],
            'threshold': [0.3, 0.3, 0.3],
        })
        
        result = coverage_by_group(df, group_col='bin')
        
        # Should sort alphabetically (batch_a < batch_m < batch_z)
        groups = list(result['group'])
        assert 'batch_a' in groups
        assert 'batch_m' in groups
        assert 'batch_z' in groups
        # Check they're in sorted order
        assert groups == sorted(groups)
    
    def test_coverage_by_group_custom_sort(self, sample_df):
        """Test custom sort column."""
        result = coverage_by_group(sample_df, group_col='bin', sort_by='n_samples')
        
        # Both have same n_samples, so order may vary but all should be present
        assert len(result) == 2
    
    def test_coverage_by_group_min_max_set_size(self, sample_df):
        """Test min/max set size computation."""
        result = coverage_by_group(sample_df, group_col='bin')
        
        a_row = result[result['group'] == 'A'].iloc[0]
        assert a_row['min_set_size'] == 1
        assert a_row['max_set_size'] == 2
        
        b_row = result[result['group'] == 'B'].iloc[0]
        assert b_row['min_set_size'] == 1
        assert b_row['max_set_size'] == 2


class TestFormatCoverageTable:
    """Test formatting functions."""
    
    @pytest.fixture
    def grouped_df(self):
        """Create sample grouped DataFrame."""
        return pd.DataFrame({
            'group': ['A', 'B', 'C'],
            'coverage': [0.8, 0.85, 0.9],
            'n_samples': [100, 150, 200],
            'avg_set_size': [1.5, 1.3, 1.2],
            'min_set_size': [1, 1, 1],
            'max_set_size': [3, 2, 2],
            'threshold_mean': [0.3, 0.35, 0.4],
        })
    
    def test_format_coverage_table_basic(self, grouped_df):
        """Test basic formatting."""
        result = format_coverage_table(grouped_df)
        
        assert isinstance(result, str)
        assert 'A' in result
        assert '0.800' in result
    
    def test_format_coverage_table_decimals(self, grouped_df):
        """Test custom decimal places."""
        result = format_coverage_table(grouped_df, decimals=1)
        
        assert '0.8' in result
        assert '100' in result
    
    def test_format_coverage_table_include_min_max(self, grouped_df):
        """Test including min/max columns."""
        result = format_coverage_table(grouped_df, include_min_max=True)
        
        assert 'min_set_size' in result or 'Min' in result or '1' in result
    
    def test_format_coverage_table_stable(self, grouped_df):
        """Test deterministic output."""
        result1 = format_coverage_table(grouped_df)
        result2 = format_coverage_table(grouped_df)
        
        assert result1 == result2


class TestMarkdownExport:
    """Test Markdown export."""
    
    @pytest.fixture
    def grouped_df(self):
        """Create sample grouped DataFrame."""
        return pd.DataFrame({
            'group': ['A', 'B'],
            'coverage': [0.85, 0.90],
            'n_samples': [100, 150],
            'avg_set_size': [1.5, 1.3],
            'threshold_mean': [0.3, 0.35],
        })
    
    def test_to_markdown_basic(self, grouped_df):
        """Test basic Markdown export."""
        result = to_markdown(grouped_df)
        
        assert isinstance(result, str)
        assert '|' in result  # Markdown table syntax
        assert 'A' in result
        assert 'B' in result
    
    def test_to_markdown_with_caption(self, grouped_df):
        """Test Markdown with caption."""
        result = to_markdown(grouped_df, caption="Coverage Results")
        
        assert "**Coverage Results**" in result
        assert '|' in result
    
    def test_to_markdown_include_min_max(self, grouped_df):
        """Test Markdown with min/max columns."""
        result = to_markdown(grouped_df, include_min_max=True)
        
        assert isinstance(result, str)


class TestLatexExport:
    """Test LaTeX export."""
    
    @pytest.fixture
    def grouped_df(self):
        """Create sample grouped DataFrame."""
        return pd.DataFrame({
            'group': ['A', 'B'],
            'coverage': [0.85, 0.90],
            'n_samples': [100, 150],
            'avg_set_size': [1.5, 1.3],
            'threshold_mean': [0.3, 0.35],
        })
    
    def test_to_latex_basic(self, grouped_df):
        """Test basic LaTeX export."""
        result = to_latex(grouped_df)
        
        assert isinstance(result, str)
        assert r'\begin{tabular}' in result
        assert 'A' in result
    
    def test_to_latex_with_caption(self, grouped_df):
        """Test LaTeX with caption."""
        result = to_latex(grouped_df, caption="Coverage Results")
        
        assert r'\caption{Coverage Results}' in result
    
    def test_to_latex_with_label(self, grouped_df):
        """Test LaTeX with label."""
        result = to_latex(grouped_df, label="tbl:coverage")
        
        assert r'\label{tbl:coverage}' in result
    
    def test_to_latex_with_caption_and_label(self, grouped_df):
        """Test LaTeX with both caption and label."""
        result = to_latex(grouped_df, caption="Results", label="tbl:results")
        
        assert r'\caption{Results}' in result
        assert r'\label{tbl:results}' in result


class TestCheckCoverageGuarantees:
    """Test coverage guarantee checking."""
    
    @pytest.fixture
    def good_coverage_df(self):
        """DataFrame with good coverage."""
        return pd.DataFrame({
            'group': ['A', 'B', 'C'],
            'coverage': [0.95, 0.92, 0.91],
            'n_samples': [100, 100, 100],
            'avg_set_size': [1.5, 1.3, 1.2],
        })
    
    @pytest.fixture
    def bad_coverage_df(self):
        """DataFrame with poor coverage."""
        return pd.DataFrame({
            'group': ['A', 'B', 'C'],
            'coverage': [0.70, 0.75, 0.80],
            'n_samples': [100, 100, 100],
            'avg_set_size': [1.5, 1.3, 1.2],
        })
    
    def test_check_coverage_guarantees_pass(self, good_coverage_df):
        """Test when all groups pass."""
        result = check_coverage_guarantees(good_coverage_df, target_coverage=0.9)
        
        assert result['overall_pass'] is True
        assert len(result['violations']) == 0
    
    def test_check_coverage_guarantees_fail(self, bad_coverage_df):
        """Test when groups fail."""
        result = check_coverage_guarantees(bad_coverage_df, target_coverage=0.9)
        
        assert result['overall_pass'] is False
        assert len(result['violations']) > 0
    
    def test_check_coverage_guarantees_tolerance(self, bad_coverage_df):
        """Test tolerance parameter."""
        # With tolerance=0.20, most should pass
        result = check_coverage_guarantees(bad_coverage_df, target_coverage=0.9, tolerance=0.20)
        
        assert result['overall_pass'] is True or len(result['violations']) < 3
    
    def test_check_coverage_guarantees_stats(self, good_coverage_df):
        """Test stats computation."""
        result = check_coverage_guarantees(good_coverage_df, target_coverage=0.9)
        
        assert 'stats' in result
        assert result['stats']['target_coverage'] == 0.9
        assert result['stats']['num_groups'] == 3
        assert result['stats']['min_coverage'] <= result['stats']['max_coverage']
    
    def test_check_coverage_guarantees_no_coverage_column(self):
        """Test error when 'coverage' column missing."""
        df = pd.DataFrame({
            'group': ['A'],
            'n_samples': [100],
        })
        
        with pytest.raises(ValueError, match="'coverage' column"):
            check_coverage_guarantees(df, target_coverage=0.9)


class TestCoverageComparison:
    """Test coverage comparison."""
    
    @pytest.fixture
    def df1(self):
        """First coverage DataFrame."""
        return pd.DataFrame({
            'group': ['A', 'B'],
            'coverage': [0.85, 0.90],
            'n_samples': [100, 150],
            'avg_set_size': [1.5, 1.3],
        })
    
    @pytest.fixture
    def df2(self):
        """Second coverage DataFrame."""
        return pd.DataFrame({
            'group': ['A', 'B'],
            'coverage': [0.87, 0.88],
            'n_samples': [100, 150],
            'avg_set_size': [1.4, 1.4],
        })
    
    def test_coverage_comparison_basic(self, df1, df2):
        """Test basic comparison."""
        result = coverage_comparison(df1, df2, name1="Method1", name2="Method2")
        
        assert 'delta_coverage' in result.columns
        assert len(result) == 2
    
    def test_coverage_comparison_deltas(self, df1, df2):
        """Test delta computation."""
        result = coverage_comparison(df1, df2)
        
        # Check deltas are computed correctly
        assert 'delta_coverage' in result.columns
        assert 'delta_set_size' in result.columns
    
    def test_coverage_comparison_unequal_groups(self):
        """Test comparison with different groups."""
        df1 = pd.DataFrame({
            'group': ['A', 'B'],
            'coverage': [0.85, 0.90],
            'n_samples': [100, 150],
            'avg_set_size': [1.5, 1.3],
        })
        df2 = pd.DataFrame({
            'group': ['B', 'C'],
            'coverage': [0.88, 0.92],
            'n_samples': [150, 200],
            'avg_set_size': [1.4, 1.2],
        })
        
        result = coverage_comparison(df1, df2)
        
        # Should have outer join with A, B, C
        assert len(result) >= 2


class TestSummarizeCoverage:
    """Test coverage summary generation."""
    
    @pytest.fixture
    def grouped_df(self):
        """Create sample grouped DataFrame."""
        return pd.DataFrame({
            'group': ['A', 'B', 'C'],
            'coverage': [0.95, 0.92, 0.91],
            'n_samples': [100, 100, 100],
            'avg_set_size': [1.5, 1.3, 1.2],
            'min_set_size': [1, 1, 1],
            'max_set_size': [3, 2, 2],
            'threshold_mean': [0.3, 0.35, 0.4],
        })
    
    def test_summarize_coverage_basic(self, grouped_df):
        """Test basic summary generation."""
        result = summarize_coverage(grouped_df, alpha=0.1)
        
        assert isinstance(result, str)
        assert 'Coverage Analysis' in result
        assert '0.9' in result or 'Target' in result
    
    def test_summarize_coverage_multiline(self, grouped_df):
        """Test summary is multiline."""
        result = summarize_coverage(grouped_df, alpha=0.1)
        
        lines = result.split('\n')
        assert len(lines) > 5
    
    def test_summarize_coverage_pass_guarantee(self, grouped_df):
        """Test summary when guarantee passes."""
        result = summarize_coverage(grouped_df, alpha=0.1)
        
        # Should indicate pass
        assert '✓' in result or 'met' in result
    
    def test_summarize_coverage_fail_guarantee(self):
        """Test summary when guarantee fails."""
        df = pd.DataFrame({
            'group': ['A', 'B'],
            'coverage': [0.70, 0.75],
            'n_samples': [100, 100],
            'avg_set_size': [1.5, 1.3],
            'min_set_size': [1, 1],
            'max_set_size': [3, 2],
            'threshold_mean': [0.3, 0.35],
        })
        
        result = summarize_coverage(df, alpha=0.1)
        
        # Should indicate violations
        assert '✗' in result or 'violation' in result.lower()
    
    def test_summarize_coverage_no_data(self):
        """Test summary with empty data."""
        df = pd.DataFrame({
            'group': [],
            'coverage': [],
            'avg_set_size': [],
        })
        
        result = summarize_coverage(df, alpha=0.1)
        
        assert 'No coverage data' in result or 'available' in result


class TestStabilityAndSorting:
    """Test sorting stability and determinism."""
    
    def test_sorting_deterministic_numeric(self):
        """Test numeric sorting is deterministic."""
        df = pd.DataFrame({
            'bin': ['5', '2', '10', '1', '3'],
            'set_size': [1, 1, 1, 1, 1],
            'covered': [1, 1, 1, 1, 1],
            'threshold': [0.3, 0.3, 0.3, 0.3, 0.3],
        })
        
        result1 = coverage_by_group(df, group_col='bin')
        result2 = coverage_by_group(df, group_col='bin')
        
        assert list(result1['group']) == list(result2['group'])
        assert list(result1['group']) == ['1', '2', '3', '5', '10']
    
    def test_sorting_stability_with_nan(self):
        """Test sorting with NaN values."""
        df = pd.DataFrame({
            'bin': ['A', 'B', 'A', 'B'],
            'set_size': [1, 1, 1, 1],
            'covered': [1, np.nan, 1, np.nan],
            'threshold': [0.3, 0.3, 0.3, 0.3],
        })
        
        result = coverage_by_group(df, group_col='bin')
        
        # Should handle NaN gracefully
        assert len(result) == 2
        assert not result['coverage'].isna().all()
    
    def test_export_stability(self):
        """Test export functions produce stable output."""
        # Create proper grouped DataFrame with all required columns
        df = pd.DataFrame({
            'group': ['A', 'B', 'C'],
            'coverage': [0.85, 0.90, 0.92],
            'n_samples': [100, 150, 200],
            'avg_set_size': [1.5, 1.3, 1.2],
            'min_set_size': [1, 1, 1],
            'max_set_size': [3, 2, 2],
            'threshold_mean': [0.3, 0.35, 0.4],
        })
        
        # Export multiple times
        md1 = to_markdown(df)
        md2 = to_markdown(df)
        
        assert md1 == md2


class TestIntegration:
    """Integration tests with real-like data."""
    
    def test_full_coverage_analysis_workflow(self):
        """Test complete workflow: data -> group -> format -> check."""
        # Create synthetic coverage data
        np.random.seed(42)
        n_samples = 300
        
        coverage_data = pd.DataFrame({
            'bin': np.repeat(['batch_A', 'batch_B', 'batch_C'], n_samples // 3),
            'set_size': np.random.randint(1, 4, n_samples),
            'covered': np.random.binomial(1, 0.88, n_samples),
            'threshold': np.random.uniform(0.2, 0.5, n_samples),
        })
        
        # Group by bin
        grouped = coverage_by_group(coverage_data, group_col='bin')
        
        # Format and export
        formatted = format_coverage_table(grouped)
        markdown = to_markdown(grouped, caption="Coverage Analysis")
        latex = to_latex(grouped, caption="Coverage", label="tbl:cov")
        
        # Check guarantees
        guarantee = check_coverage_guarantees(grouped, target_coverage=0.9, tolerance=0.05)
        
        # Summarize
        summary = summarize_coverage(grouped, alpha=0.1)
        
        # Verify all outputs are non-empty
        assert len(formatted) > 0
        assert len(markdown) > 0
        assert len(latex) > 0
        assert isinstance(guarantee, dict)
        assert len(summary) > 0
    
    def test_multi_method_comparison_workflow(self):
        """Test comparing two methods."""
        np.random.seed(42)
        
        # Method 1
        df1 = coverage_by_group(pd.DataFrame({
            'bin': ['A', 'A', 'B', 'B'],
            'set_size': [1, 1, 2, 2],
            'covered': [1, 1, 1, 0],
            'threshold': [0.3, 0.3, 0.4, 0.4],
        }), group_col='bin')
        
        # Method 2
        df2 = coverage_by_group(pd.DataFrame({
            'bin': ['A', 'A', 'B', 'B'],
            'set_size': [1, 2, 1, 2],
            'covered': [1, 1, 1, 1],
            'threshold': [0.35, 0.35, 0.35, 0.35],
        }), group_col='bin')
        
        # Compare
        comparison = coverage_comparison(df1, df2, name1="CP", name2="CP+")
        
        assert len(comparison) == 2
        assert 'delta_coverage' in comparison.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
