"""Statistical justification snippets for manuscripts."""
from __future__ import annotations


from typing import Dict, Optional

TEST_RATIONALES: Dict[str, str] = {
    "t-test": "Independent samples t-test was selected for approximately normal, equal-variance groups.",
    "paired_t-test": "Paired t-test was used to account for within-sample pairing across conditions.",
    "mann-whitney": "Mann-Whitney U was used for non-parametric two-group comparisons.",
    "wilcoxon": "Wilcoxon signed-rank test handled paired non-parametric data.",
    "anova": "ANOVA was applied for multi-group mean comparison with post-hoc control for multiplicity.",
    "kruskal-wallis": "Kruskal-Wallis tested stochastic dominance across more than two groups.",
    "mcnemar": "McNemar's test compared paired binary outcomes from champion-challenger models.",
}

ASSUMPTION_CHECKS: Dict[str, str] = {
    "normality": "Normality screened via Shapiro-Wilk with visual QQ confirmation.",
    "variance": "Variance homogeneity evaluated using Levene's test.",
    "independence": "Sampling independence enforced through study design and deduplication checks.",
}


def summarize_assumptions(include: Optional[list[str]] = None) -> str:
    """Return a compact assumptions paragraph."""

    items = include or list(ASSUMPTION_CHECKS.keys())
    checks = [ASSUMPTION_CHECKS[k] for k in items if k in ASSUMPTION_CHECKS]
    return " ".join(checks)


def statistical_justification(
    test: str,
    design: str,
    alpha: float = 0.05,
    correction: Optional[str] = "holm-bonferroni",
    effect_size: Optional[str] = None,
    power: Optional[float] = None,
) -> str:
    """Generate a concise statistical justification paragraph."""

    rationale = TEST_RATIONALES.get(test.lower(), f"{test} was applied per study design.")
    parts = [rationale]
    parts.append(
        f"Family-wise error controlled at alpha={alpha:.3f} using {correction} where applicable."
        if correction
        else f"Significance threshold set at alpha={alpha:.3f} without multiplicity correction."
    )
    parts.append(f"Design: {design}.")
    if effect_size:
        parts.append(f"Effect sizes reported as {effect_size} with confidence intervals.")
    if power:
        parts.append(f"A priori power analysis targeted {power:.0%} power at alpha={alpha:.3f}.")
    parts.append(summarize_assumptions())
    return " ".join(parts)
