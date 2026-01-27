"""
Coverage analysis utilities for conformal prediction.

Provides helpers to compute, format, and analyze coverage tables with
optional grouping by metadata bins. Ensures stable, reproducible output
with consistent sorting.

Key Functions:
    coverage_by_group - Compute coverage metrics per group
    format_coverage_table - Format as human-readable table
    to_markdown - Export as Markdown
    to_latex - Export as LaTeX
    check_coverage_guarantees - Verify coverage ≥ target
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def coverage_by_group(
    df: pd.DataFrame,
    group_col: str = "bin",
    sort_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute coverage metrics grouped by a column.

    Aggregates per-sample coverage data into group-level statistics:
    - coverage: empirical coverage in group
    - n_samples: number of samples
    - avg_set_size: mean prediction set size
    - min_set_size: minimum set size
    - max_set_size: maximum set size

    Parameters
    ----------
    df : pd.DataFrame
        Output from `ConformalPredictionResult.to_dataframe()` with columns:
        - set_size (int): prediction set size per sample
        - covered (int, optional): binary indicator if true label in set
        - threshold (float): nonconformity threshold used
        - [group_col] (str): grouping variable
    group_col : str, default 'bin'
        Column name to group by (typically 'bin' for Mondrian conditioning).
    sort_by : str, optional
        Column to sort results by. If None, sorts by group_col alphabetically.

    Returns
    -------
    grouped_df : pd.DataFrame
        Aggregated table with columns:
        - group: group identifier
        - coverage: empirical coverage [0, 1]
        - n_samples: number of samples in group
        - avg_set_size: mean set size
        - min_set_size: minimum set size
        - max_set_size: maximum set size
        - threshold_mean: mean threshold used

    Raises
    ------
    ValueError
        If group_col not in df.
    """
    if group_col not in df.columns:
        raise ValueError(f"group_col='{group_col}' not found in DataFrame columns: {list(df.columns)}")

    # Group by the specified column
    grouped = df.groupby(group_col, sort=False)

    results = []
    for group_name, group_df in grouped:
        n = len(group_df)
        avg_set_size = float(np.mean(group_df['set_size']))
        min_set_size = int(np.min(group_df['set_size']))
        max_set_size = int(np.max(group_df['set_size']))
        threshold_mean = float(np.mean(group_df['threshold']))

        # Compute coverage if 'covered' column exists
        if 'covered' in group_df.columns:
            coverage = float(np.mean(group_df['covered']))
        else:
            coverage = np.nan

        results.append({
            'group': str(group_name),
            'coverage': coverage,
            'n_samples': n,
            'avg_set_size': avg_set_size,
            'min_set_size': min_set_size,
            'max_set_size': max_set_size,
            'threshold_mean': threshold_mean,
        })

    result_df = pd.DataFrame(results)

    # Sort: first by sort_by if specified, else by group
    if sort_by and sort_by in result_df.columns:
        result_df = result_df.sort_values(sort_by, na_position='last').reset_index(drop=True)
    else:
        # Sort by group column (alphabetically/numerically stable)
        numeric_groups = pd.to_numeric(result_df['group'], errors='coerce')
        if numeric_groups.notna().any():
            # Has at least some numeric groups - sort numerically
            result_df['_sort_key'] = numeric_groups
            result_df = result_df.sort_values('_sort_key', na_position='last').drop('_sort_key', axis=1)
        else:
            # All non-numeric - sort alphabetically
            result_df = result_df.sort_values('group').reset_index(drop=True)

        result_df = result_df.reset_index(drop=True)

    return result_df


def format_coverage_table(
    grouped_df: pd.DataFrame,
    decimals: int = 3,
    include_min_max: bool = False,
) -> str:
    """
    Format coverage table as human-readable string.

    Parameters
    ----------
    grouped_df : pd.DataFrame
        Output from coverage_by_group().
    decimals : int, default 3
        Decimal places for floating-point columns.
    include_min_max : bool, default False
        If True, include min/max set size columns.

    Returns
    -------
    formatted_str : str
        Formatted table string.
    """
    # Make a copy to avoid modifying original
    df_copy = grouped_df.copy()

    # Select columns to display
    if include_min_max:
        cols = ['group', 'n_samples', 'coverage', 'avg_set_size', 'min_set_size', 'max_set_size', 'threshold_mean']
    else:
        cols = ['group', 'n_samples', 'coverage', 'avg_set_size', 'threshold_mean']

    cols = [c for c in cols if c in df_copy.columns]
    display_df = df_copy[cols].copy()

    # Format floating-point columns
    float_cols = ['coverage', 'avg_set_size', 'threshold_mean']
    for col in float_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "N/A"
            )

    # Convert to string with nice formatting
    return display_df.to_string(index=False)


def to_markdown(
    grouped_df: pd.DataFrame,
    decimals: int = 3,
    include_min_max: bool = False,
    caption: Optional[str] = None,
) -> str:
    """
    Export coverage table as Markdown.

    Parameters
    ----------
    grouped_df : pd.DataFrame
        Output from coverage_by_group().
    decimals : int, default 3
        Decimal places.
    include_min_max : bool, default False
        Include min/max set size.
    caption : str, optional
        Table caption.

    Returns
    -------
    markdown_str : str
        Markdown formatted table.
    """
    df_copy = grouped_df.copy()

    # Select columns
    if include_min_max:
        cols = ['group', 'n_samples', 'coverage', 'avg_set_size', 'min_set_size', 'max_set_size', 'threshold_mean']
    else:
        cols = ['group', 'n_samples', 'coverage', 'avg_set_size', 'threshold_mean']

    cols = [c for c in cols if c in df_copy.columns]
    display_df = df_copy[cols].copy()

    # Format floating-point columns
    float_cols = ['coverage', 'avg_set_size', 'threshold_mean']
    for col in float_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "N/A"
            )

    # Rename columns for display
    rename_map = {
        'group': 'Group',
        'n_samples': 'N Samples',
        'coverage': 'Coverage',
        'avg_set_size': 'Avg Set Size',
        'min_set_size': 'Min Set Size',
        'max_set_size': 'Max Set Size',
        'threshold_mean': 'Mean Threshold',
    }
    display_df = display_df.rename(columns=rename_map)

    # Convert to markdown
    markdown = display_df.to_markdown(index=False)

    if caption:
        markdown = f"**{caption}**\n\n{markdown}"

    return markdown


def to_latex(
    grouped_df: pd.DataFrame,
    decimals: int = 3,
    include_min_max: bool = False,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """
    Export coverage table as LaTeX.

    Parameters
    ----------
    grouped_df : pd.DataFrame
        Output from coverage_by_group().
    decimals : int, default 3
        Decimal places.
    include_min_max : bool, default False
        Include min/max set size.
    caption : str, optional
        Table caption.
    label : str, optional
        LaTeX label for cross-referencing.

    Returns
    -------
    latex_str : str
        LaTeX formatted table.
    """
    df_copy = grouped_df.copy()

    # Select columns
    if include_min_max:
        cols = ['group', 'n_samples', 'coverage', 'avg_set_size', 'min_set_size', 'max_set_size', 'threshold_mean']
    else:
        cols = ['group', 'n_samples', 'coverage', 'avg_set_size', 'threshold_mean']

    cols = [c for c in cols if c in df_copy.columns]
    display_df = df_copy[cols].copy()

    # Format floating-point columns
    float_cols = ['coverage', 'avg_set_size', 'threshold_mean']
    for col in float_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "N/A"
            )

    # Rename columns for display
    rename_map = {
        'group': 'Group',
        'n_samples': 'N Samples',
        'coverage': 'Coverage',
        'avg_set_size': 'Avg Set Size',
        'min_set_size': 'Min Set Size',
        'max_set_size': 'Max Set Size',
        'threshold_mean': 'Mean Threshold',
    }
    display_df = display_df.rename(columns=rename_map)

    # Convert to LaTeX
    latex = display_df.to_latex(index=False, escape=False)

    # Add caption and label if provided
    if caption or label:
        lines = latex.split('\n')
        # Find the line after \begin{tabular}...
        insert_idx = 1
        for i, line in enumerate(lines):
            if r'\begin{tabular}' in line:
                insert_idx = i + 1
                break

        caption_lines = []
        if caption:
            caption_lines.append(f"  \\caption{{{caption}}}")
        if label:
            caption_lines.append(f"  \\label{{{label}}}")

        for cap_line in reversed(caption_lines):
            lines.insert(insert_idx, cap_line)

        latex = '\n'.join(lines)

    return latex


def check_coverage_guarantees(
    grouped_df: pd.DataFrame,
    target_coverage: float,
    tolerance: float = 0.05,
) -> Dict[str, bool | float]:
    """
    Check if coverage ≥ target in each group (within tolerance).

    Parameters
    ----------
    grouped_df : pd.DataFrame
        Output from coverage_by_group().
    target_coverage : float
        Target coverage level (e.g., 0.9 for 90%).
    tolerance : float, default 0.05
        Tolerance (typically 0.05 for statistical fluctuation).

    Returns
    -------
    results : dict
        Keys: 'overall_pass', 'violations', 'near_violations', 'stats'
        - overall_pass (bool): All groups meet target within tolerance
        - violations (list): Groups failing guarantee
        - near_violations (list): Groups within tolerance but below target
        - stats (dict): Min/max coverage, num groups
    """
    if 'coverage' not in grouped_df.columns:
        raise ValueError("grouped_df must have 'coverage' column")

    coverages = grouped_df['coverage'].dropna()

    violations = []
    near_violations = []

    for _, row in grouped_df.iterrows():
        cov = row['coverage']
        group = row['group']

        if pd.isna(cov):
            continue

        # Hard failure: below (target - tolerance)
        if cov < target_coverage - tolerance:
            violations.append({
                'group': group,
                'coverage': float(cov),
                'gap': float(target_coverage - cov),
            })
        # Warning: within tolerance but below target
        elif cov < target_coverage:
            near_violations.append({
                'group': group,
                'coverage': float(cov),
                'gap': float(target_coverage - cov),
            })

    overall_pass = len(violations) == 0

    return {
        'overall_pass': overall_pass,
        'violations': violations,
        'near_violations': near_violations,
        'stats': {
            'target_coverage': target_coverage,
            'min_coverage': float(np.min(coverages)) if len(coverages) > 0 else np.nan,
            'max_coverage': float(np.max(coverages)) if len(coverages) > 0 else np.nan,
            'num_groups': len(grouped_df),
        },
    }


def coverage_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str = "Method 1",
    name2: str = "Method 2",
) -> pd.DataFrame:
    """
    Compare coverage tables from two methods.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Output from coverage_by_group() for each method.
    name1, name2 : str
        Names for the two methods.

    Returns
    -------
    comparison_df : pd.DataFrame
        Side-by-side comparison with delta columns.
    """
    # Merge on 'group'
    merged = pd.merge(
        df1[['group', 'coverage', 'n_samples', 'avg_set_size']],
        df2[['group', 'coverage', 'n_samples', 'avg_set_size']],
        on='group',
        how='outer',
        suffixes=(f'_{name1.replace(" ", "_")}', f'_{name2.replace(" ", "_")}'),
    )

    # Compute deltas
    cov_cols = [c for c in merged.columns if 'coverage' in c]
    if len(cov_cols) == 2:
        merged['delta_coverage'] = merged[cov_cols[1]] - merged[cov_cols[0]]

    set_size_cols = [c for c in merged.columns if 'avg_set_size' in c]
    if len(set_size_cols) == 2:
        merged['delta_set_size'] = merged[set_size_cols[1]] - merged[set_size_cols[0]]

    # Sort by group
    try:
        merged['_sort_key'] = pd.to_numeric(merged['group'], errors='coerce')
        merged = merged.sort_values('_sort_key', na_position='last').drop('_sort_key', axis=1)
    except Exception:
        merged = merged.sort_values('group')

    merged = merged.reset_index(drop=True)

    return merged


def summarize_coverage(
    grouped_df: pd.DataFrame,
    alpha: float = 0.1,
) -> str:
    """
    Generate a human-readable summary of coverage analysis.

    Parameters
    ----------
    grouped_df : pd.DataFrame
        Output from coverage_by_group().
    alpha : float, default 0.1
        Significance level (for computing target_coverage = 1 - alpha).

    Returns
    -------
    summary_str : str
        Human-readable summary.
    """
    target_coverage = 1.0 - alpha

    if 'coverage' not in grouped_df.columns:
        return "No coverage data available."

    coverages = grouped_df['coverage'].dropna()

    if len(coverages) == 0:
        return "No coverage data available."

    min_cov = float(np.min(coverages))
    max_cov = float(np.max(coverages))
    mean_cov = float(np.mean(coverages))

    summary_lines = [
        f"Coverage Analysis (α={alpha})",
        f"{'=' * 50}",
        f"Target coverage: {target_coverage:.1%}",
        f"Groups analyzed: {len(grouped_df)}",
        "",
        "Empirical Coverage:",
        f"  Min:     {min_cov:.1%}",
        f"  Mean:    {mean_cov:.1%}",
        f"  Max:     {max_cov:.1%}",
        "",
        "Set Sizes:",
        f"  Mean: {float(np.mean(grouped_df['avg_set_size'])):.2f}",
        f"  Min:  {int(np.min(grouped_df['min_set_size']))}",
        f"  Max:  {int(np.max(grouped_df['max_set_size']))}",
    ]

    # Check guarantees
    guarantee_check = check_coverage_guarantees(grouped_df, target_coverage, tolerance=0.05)

    if guarantee_check['overall_pass']:
        summary_lines.append("✓ Coverage guarantee met in all groups")
    else:
        summary_lines.append(f"✗ Coverage violations in {len(guarantee_check['violations'])} groups:")
        for viol in guarantee_check['violations']:
            summary_lines.append(f"    - {viol['group']}: {viol['coverage']:.1%} (gap: {viol['gap']:.1%})")

    return "\n".join(summary_lines)


__all__ = [
    "coverage_by_group",
    "format_coverage_table",
    "to_markdown",
    "to_latex",
    "check_coverage_guarantees",
    "coverage_comparison",
    "summarize_coverage",
]
