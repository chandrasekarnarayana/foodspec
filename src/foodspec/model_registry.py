"""Simple model registry for saving/loading trained models with metadata."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional, Tuple

import joblib
import numpy as np
from scipy import stats as scipy_stats

__all__ = [
    "ModelMetadata",
    "save_model",
    "load_model",
    "ChampionChallengerComparison",
    "compare_champion_challenger",
    "promote_challenger",
]


@dataclass
class ModelMetadata:
    """Metadata describing a saved model artifact."""

    name: str
    version: str
    created_at: str
    foodspec_version: str
    extra: Dict[str, Any]


def _base_paths(path: str | pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    p = pathlib.Path(path)
    return p.with_suffix(".joblib"), p.with_suffix(".json")


def save_model(
    model: Any,
    path: str | pathlib.Path,
    name: str,
    version: str,
    foodspec_version: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model and metadata to disk."""

    model_path, meta_path = _base_paths(path)
    now_utc = datetime.now(timezone.utc)
    metadata = ModelMetadata(
        name=name,
        version=version,
        created_at=now_utc.isoformat().replace("+00:00", "Z"),
        foodspec_version=foodspec_version,
        extra=extra or {},
    )
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(metadata.__dict__), encoding="utf-8")


def load_model(path: str | pathlib.Path) -> tuple[Any, ModelMetadata]:
    """Load model and associated metadata."""

    model_path, meta_path = _base_paths(path)
    model = joblib.load(model_path)
    meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = ModelMetadata(**meta_dict)
    return model, metadata


# ============================================================================
# Champion-Challenger Comparison
# ============================================================================


@dataclass
class ChampionChallengerComparison:
    """Results of champion vs challenger model comparison.

    Attributes
    ----------
    champion_metric : float
        Champion model performance.
    challenger_metric : float
        Challenger model performance.
    metric_name : str
        Performance metric (e.g., "accuracy", "f1", "auc").
    improvement : float
        Challenger improvement over champion (positive = better).
    improvement_pct : float
        Percentage improvement.
    n_samples : int
        Number of test samples.
    statistical_test : str
        Test used for significance (e.g., "paired_ttest", "mcnemar").
    test_statistic : float
        Test statistic value.
    pvalue : float
        Statistical significance p-value.
    is_significant : bool
        True if improvement is statistically significant.
    confidence_interval : Tuple[float, float]
        95% CI for improvement.
    recommendation : str
        Promotion decision recommendation.
    metadata : Dict
        Additional context (deployment info, A/B test results, etc.).
    """

    champion_metric: float
    challenger_metric: float
    metric_name: str
    improvement: float
    improvement_pct: float
    n_samples: int
    statistical_test: str
    test_statistic: float
    pvalue: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    recommendation: str
    metadata: Dict = field(default_factory=dict)


def compare_champion_challenger(
    champion_scores: np.ndarray,
    challenger_scores: np.ndarray,
    metric_name: str = "accuracy",
    alpha: float = 0.05,
    min_improvement: float = 0.01,
    test_type: Literal["paired_ttest", "mcnemar", "wilcoxon"] = "paired_ttest",
) -> ChampionChallengerComparison:
    """Compare champion and challenger models using statistical tests.

    Parameters
    ----------
    champion_scores : np.ndarray
        Per-sample scores for champion model (e.g., 1=correct, 0=wrong).
    challenger_scores : np.ndarray
        Per-sample scores for challenger model.
    metric_name : str
        Performance metric name.
    alpha : float
        Significance level (default 0.05 for 95% confidence).
    min_improvement : float
        Minimum improvement required for promotion (default 1%).
    test_type : Literal["paired_ttest", "mcnemar", "wilcoxon"]
        Statistical test:
        - "paired_ttest": Paired t-test (continuous scores).
        - "mcnemar": McNemar's test (binary predictions).
        - "wilcoxon": Wilcoxon signed-rank test (non-parametric).

    Returns
    -------
    ChampionChallengerComparison
        Comprehensive comparison results with promotion recommendation.
    """
    champion_scores = np.asarray(champion_scores)
    challenger_scores = np.asarray(challenger_scores)

    if champion_scores.shape != challenger_scores.shape:
        raise ValueError("Champion and challenger scores must have same shape.")

    n_samples = len(champion_scores)

    # Compute metrics
    champion_metric = float(champion_scores.mean())
    challenger_metric = float(challenger_scores.mean())
    improvement = challenger_metric - champion_metric
    improvement_pct = (improvement / (champion_metric + 1e-10)) * 100.0

    # Statistical test
    if test_type == "paired_ttest":
        t_stat, p_value = scipy_stats.ttest_rel(challenger_scores, champion_scores)
        test_statistic = float(t_stat)
        pvalue = float(p_value)

        # 95% CI for difference
        diff = challenger_scores - champion_scores
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        ci_low = improvement - 1.96 * se
        ci_high = improvement + 1.96 * se

    elif test_type == "mcnemar":
        # McNemar's test for binary predictions
        # Contingency table: (both correct, only champ correct, only chal correct, both wrong)
        b = ((champion_scores == 1) & (challenger_scores == 0)).sum()  # Champ correct, Chal wrong
        c = ((champion_scores == 0) & (challenger_scores == 1)).sum()  # Champ wrong, Chal correct

        if b + c == 0:
            test_statistic = 0.0
            pvalue = 1.0
        else:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # McNemar with continuity correction
            test_statistic = float(chi2)
            pvalue = float(scipy_stats.chi2.sf(chi2, df=1))

        # CI via normal approximation
        diff = c - b
        se = np.sqrt(b + c)
        ci_low = (diff - 1.96 * se) / n_samples
        ci_high = (diff + 1.96 * se) / n_samples

    elif test_type == "wilcoxon":
        # Wilcoxon signed-rank test (non-parametric)
        stat, p_value = scipy_stats.wilcoxon(challenger_scores, champion_scores, alternative="greater")
        test_statistic = float(stat)
        pvalue = float(p_value)

        # Bootstrap CI for difference
        diff = challenger_scores - champion_scores
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        ci_low = improvement - 1.96 * se
        ci_high = improvement + 1.96 * se

    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    confidence_interval = (float(ci_low), float(ci_high))

    # Significance decision
    is_significant = pvalue < alpha

    # Promotion recommendation (prioritize negative performance declines)
    if improvement < 0:
        recommendation = f"❌ REJECT: Challenger underperforms champion ({improvement_pct:+.2f}%, p={pvalue:.4f})."
    elif is_significant and improvement >= min_improvement:
        recommendation = (
            f"✅ PROMOTE: Challenger significantly outperforms champion ({improvement_pct:+.2f}%, p={pvalue:.4f})."
        )
    elif improvement >= min_improvement:
        recommendation = (
            f"⚡ BORDERLINE: Improvement meets threshold ({improvement_pct:+.2f}%) "
            f"but not statistically significant (p={pvalue:.4f}). Consider more data."
        )
    elif is_significant:
        recommendation = (
            f"⚠️ CAUTION: Statistically significant but improvement too small "
            f"({improvement_pct:+.2f}% < {min_improvement * 100:.1f}%)."
        )
    else:
        recommendation = f"❌ REJECT: No significant improvement ({improvement_pct:+.2f}%, p={pvalue:.4f})."

    return ChampionChallengerComparison(
        champion_metric=champion_metric,
        challenger_metric=challenger_metric,
        metric_name=metric_name,
        improvement=improvement,
        improvement_pct=improvement_pct,
        n_samples=n_samples,
        statistical_test=test_type,
        test_statistic=test_statistic,
        pvalue=pvalue,
        is_significant=is_significant,
        confidence_interval=confidence_interval,
        recommendation=recommendation,
    )


def promote_challenger(
    comparison: ChampionChallengerComparison,
    champion_path: str | pathlib.Path,
    challenger_path: str | pathlib.Path,
    backup_suffix: str = ".backup",
    force: bool = False,
) -> bool:
    """Promote challenger to champion if comparison recommends it.

    Parameters
    ----------
    comparison : ChampionChallengerComparison
        Comparison results from compare_champion_challenger.
    champion_path : str | pathlib.Path
        Path to current champion model.
    challenger_path : str | pathlib.Path
        Path to challenger model.
    backup_suffix : str
        Suffix for backing up old champion (default ".backup").
    force : bool
        If True, promote even without statistical significance.

    Returns
    -------
    bool
        True if promotion occurred.
    """
    champion_path = pathlib.Path(champion_path)
    challenger_path = pathlib.Path(challenger_path)

    # Check promotion criteria
    should_promote = comparison.is_significant or force

    if not should_promote:
        return False

    # Backup current champion
    backup_path = champion_path.with_suffix(champion_path.suffix + backup_suffix)
    if champion_path.exists():
        import shutil

        shutil.copy2(champion_path, backup_path)

    # Promote challenger
    import shutil

    shutil.copy2(challenger_path, champion_path)

    # Also handle metadata
    champion_meta_path = champion_path.with_suffix(".json")
    challenger_meta_path = challenger_path.with_suffix(".json")
    backup_meta_path = backup_path.with_suffix(".json")

    if champion_meta_path.exists():
        shutil.copy2(champion_meta_path, backup_meta_path)
    if challenger_meta_path.exists():
        shutil.copy2(challenger_meta_path, champion_meta_path)

    return True
