"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

Inference-only prediction from deployment bundles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# Mapping from method names to preprocessor classes
_PREPROCESSOR_MAP: Dict[str, Any] = {}


def _get_preprocessor_class(method: str) -> Any:
    """Get preprocessor class by method name.

    Parameters
    ----------
    method : str
        Preprocessing method name (e.g., 'snv', 'savgol', 'msc').

    Returns
    -------
    cls : class
        Preprocessor class.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    global _PREPROCESSOR_MAP

    # Lazy import to avoid circular dependencies
    if not _PREPROCESSOR_MAP:
        from ..preprocess.common import SNV, Derivative, SavitzkyGolay, VectorNormalize
        from ..preprocess.ir import ExtendedMultiplicativeScatterCorrection, MultiplicativeScatterCorrection
        from ..preprocess.raman import AsLSBaseline, DespikeHampel

        _PREPROCESSOR_MAP = {
            "snv": SNV,
            "savgol": SavitzkyGolay,
            "savitzky_golay": SavitzkyGolay,
            "derivative": Derivative,
            "vector_normalize": VectorNormalize,
            "msc": MultiplicativeScatterCorrection,
            "emsc": ExtendedMultiplicativeScatterCorrection,
            "asls": AsLSBaseline,
            "despike": DespikeHampel,
            "hampel": DespikeHampel,
        }

    if method not in _PREPROCESSOR_MAP:
        available = ", ".join(sorted(_PREPROCESSOR_MAP.keys()))
        raise ValueError(
            f"Unknown preprocessing method '{method}'. Available methods: {available}"
        )

    return _PREPROCESSOR_MAP[method]


def predict_from_bundle(
    bundle: Dict[str, Any],
    input_csv: Path | str,
    output_dir: Path | str,
    wavenumber_col: str = "wavenumber",
    intensity_col: str = "intensity",
    sample_id_col: str = "sample_id",
    save_probabilities: bool = True,
) -> pd.DataFrame:
    """Make predictions on new data using a loaded bundle.

    Applies preprocessing pipeline, feature extraction (if applicable), and model
    prediction. Saves predictions and probabilities to CSV files.

    Parameters
    ----------
    bundle : dict
        Loaded bundle from load_bundle().
    input_csv : Path | str
        Path to input CSV file with spectra.
    output_dir : Path | str
        Directory for output CSV files.
    wavenumber_col : str, default="wavenumber"
        Column name for wavenumber values.
    intensity_col : str, default="intensity"
        Column name for intensity values.
    sample_id_col : str, default="sample_id"
        Column name for sample identifiers.
    save_probabilities : bool, default=True
        Whether to save probability matrix to CSV.

    Returns
    -------
    predictions_df : DataFrame
        DataFrame with sample_id, predicted_class, and predicted_label columns.

    Raises
    ------
    ValueError
        If bundle does not contain required model.
    FileNotFoundError
        If input CSV file does not exist.

    Examples
    --------
    >>> from foodspec.deploy import load_bundle, predict_from_bundle
    >>> bundle = load_bundle("./bundle")  # doctest: +SKIP
    >>> predictions = predict_from_bundle(
    ...     bundle=bundle,
    ...     input_csv="new_samples.csv",
    ...     output_dir="./predictions"
    ... )  # doctest: +SKIP
    """

    # Validate bundle has required components
    if bundle["model"] is None:
        raise ValueError("Bundle does not contain a trained model")

    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load input data
    df = pd.read_csv(input_path)

    # Validate required columns
    required_cols = [sample_id_col, wavenumber_col, intensity_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    # Group by sample_id and extract spectra
    samples = []
    sample_ids = []
    for sample_id, group in df.groupby(sample_id_col):
        wavenumbers = group[wavenumber_col].values
        intensities = group[intensity_col].values

        # Sort by wavenumber
        sort_idx = np.argsort(wavenumbers)
        wavenumbers = wavenumbers[sort_idx]
        intensities = intensities[sort_idx]

        samples.append(intensities)
        sample_ids.append(sample_id)

    X = np.array(samples)

    # Apply preprocessing pipeline if available
    if bundle["preprocess_pipeline"] is not None:
        X_processed = _apply_preprocessing_pipeline(X, bundle["preprocess_pipeline"])
    else:
        X_processed = X

    # Apply model
    model = bundle["model"]
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)

    # Create predictions DataFrame
    predictions_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "predicted_class": predictions,
        }
    )

    # Add predicted labels if encoder available
    if bundle["label_encoder"] is not None:
        encoder = bundle["label_encoder"]
        predictions_df["predicted_label"] = predictions_df["predicted_class"].map(encoder)
    else:
        predictions_df["predicted_label"] = predictions_df["predicted_class"]

    # Save predictions
    predictions_file = output_path / "predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)

    # Save probabilities if requested
    if save_probabilities:
        # Create probability column names
        if bundle["label_encoder"] is not None:
            encoder = bundle["label_encoder"]
            n_classes = len(encoder)
            prob_cols = [f"prob_{encoder[i]}" for i in range(n_classes)]
        else:
            n_classes = probabilities.shape[1]
            prob_cols = [f"prob_class_{i}" for i in range(n_classes)]

        prob_df = pd.DataFrame(probabilities, columns=prob_cols)
        prob_df.insert(0, "sample_id", sample_ids)

        probabilities_file = output_path / "probabilities.csv"
        prob_df.to_csv(probabilities_file, index=False)

    return predictions_df


def _apply_preprocessing_pipeline(
    X: np.ndarray, pipeline: list[Dict[str, Any]]
) -> np.ndarray:
    """Apply preprocessing pipeline to spectra.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input spectra.
    pipeline : list of dict
        Preprocessing pipeline specification.

    Returns
    -------
    X_processed : ndarray of shape (n_samples, n_features)
        Preprocessed spectra.
    """
    X_out = X.copy()

    for step in pipeline:
        method = step["method"]
        params = step.get("params", {})

        try:
            preprocessor_cls = _get_preprocessor_class(method)
            preprocessor = preprocessor_cls(**params)
            X_out = preprocessor.transform(X_out)
        except Exception as e:
            raise ValueError(
                f"Failed to apply preprocessing step '{method}' with params {params}: {e}"
            ) from e

    return X_out


def predict_from_bundle_path(
    bundle_dir: Path | str,
    input_csv: Path | str,
    output_dir: Path | str,
    wavenumber_col: str = "wavenumber",
    intensity_col: str = "intensity",
    sample_id_col: str = "sample_id",
    save_probabilities: bool = True,
) -> pd.DataFrame:
    """Convenience function to load bundle and make predictions.

    Combines bundle loading and prediction in a single call.

    Parameters
    ----------
    bundle_dir : Path | str
        Path to bundle directory.
    input_csv : Path | str
        Path to input CSV file with spectra.
    output_dir : Path | str
        Directory for output CSV files.
    wavenumber_col : str, default="wavenumber"
        Column name for wavenumber values.
    intensity_col : str, default="intensity"
        Column name for intensity values.
    sample_id_col : str, default="sample_id"
        Column name for sample identifiers.
    save_probabilities : bool, default=True
        Whether to save probability matrix to CSV.

    Returns
    -------
    predictions_df : DataFrame
        DataFrame with sample_id, predicted_class, and predicted_label columns.

    Examples
    --------
    >>> from foodspec.deploy import predict_from_bundle_path
    >>> predictions = predict_from_bundle_path(
    ...     bundle_dir="./bundle",
    ...     input_csv="new_samples.csv",
    ...     output_dir="./predictions"
    ... )  # doctest: +SKIP
    """
    from .bundle import load_bundle

    bundle = load_bundle(bundle_dir)

    return predict_from_bundle(
        bundle=bundle,
        input_csv=input_csv,
        output_dir=output_dir,
        wavenumber_col=wavenumber_col,
        intensity_col=intensity_col,
        sample_id_col=sample_id_col,
        save_probabilities=save_probabilities,
    )


__all__ = ["predict_from_bundle", "predict_from_bundle_path"]
