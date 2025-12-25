"""Modeling mixin for FoodSpec API - feature extraction, training, library search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.features.specs import FeatureEngine, FeatureSpec


class FoodSpecModelingMixin:
    """Mixin class providing modeling capabilities for FoodSpec."""

    def features(
        self,
        preset: str = "specs",
        specs: Optional[list[FeatureSpec]] = None,
        **kwargs,
    ):
        """Extract features (peaks, ratios, etc.).

        Parameters
        ----------
        preset : str, optional
            Preset name. Default: 'specs' uses provided FeatureSpec list.
        specs : list[FeatureSpec], optional
            Feature specifications to evaluate when preset='specs'.
        **kwargs
            Override preset parameters or provide default specs for other presets.

        Returns
        -------
        FoodSpec
            Self (for chaining).
        """
        if preset == "specs":
            if not specs:
                raise ValueError("Feature specs are required when preset='specs'")
            engine = FeatureEngine(specs)
            features_df, diag = engine.evaluate(self.data)
        else:
            # Minimal defaults for quick/standard presets using bands and peak heights
            default_specs = {
                "quick": [
                    FeatureSpec(
                        name="band_full",
                        ftype="band",
                        regions=[(float(self.data.wavenumbers.min()), float(self.data.wavenumbers.max()))],
                    ),
                ],
                "standard": [
                    FeatureSpec(
                        name="band_mid",
                        ftype="band",
                        regions=[
                            (
                                float(np.percentile(self.data.wavenumbers, 25)),
                                float(np.percentile(self.data.wavenumbers, 75)),
                            )
                        ],
                    ),
                    FeatureSpec(
                        name="peak_max",
                        ftype="peak",
                        regions=[
                            (
                                float(self.data.wavenumbers[np.argmax(self.data.x.mean(axis=0))]) - 5,
                                float(self.data.wavenumbers[np.argmax(self.data.x.mean(axis=0))]) + 5,
                            )
                        ],
                        params={"tolerance": 5.0, "metrics": ("height",)},
                    ),
                ],
            }
            chosen = default_specs.get(preset)
            if chosen is None:
                raise ValueError(f"Unknown features preset: {preset}")
            engine = FeatureEngine(chosen)
            features_df, diag = engine.evaluate(self.data)

        self.bundle.add_diagnostic("features_table", features_df.to_dict(orient="list"))
        self.bundle.add_metrics("features", {"n_features": features_df.shape[1]})

        spec_hashes = [s.hash() for s in (specs or [])]
        step_hash = hashlib.sha256(
            json.dumps({"preset": preset, "spec_hashes": spec_hashes}, sort_keys=True).encode()
        ).hexdigest()[:8]
        self.bundle.run_record.add_step(
            "features",
            step_hash,
            metadata={"preset": preset, "features": spec_hashes or ["default"]},
        )

        self._steps_applied.append(f"features({preset})")
        return self

    def train(
        self,
        algorithm: str = "rf",
        label_column: str = "label",
        test_size: float = 0.3,
        cv_folds: int = 5,
        **kwargs,
    ):
        """Train a model on the data.

        Parameters
        ----------
        algorithm : {'rf', 'lr', 'svm', 'pls_da'}, optional
            Algorithm to use. Default: 'rf'.
        label_column : str, optional
            Metadata column for labels. Default: 'label'.
        test_size : float, optional
            Test set fraction. Default: 0.3.
        cv_folds : int, optional
            Cross-validation folds. Default: 5.
        **kwargs
            Algorithm-specific parameters.

        Returns
        -------
        FoodSpec
            Self (for chaining).
        """
        from foodspec.chemometrics.models import make_classifier
        from foodspec.chemometrics.validation import compute_classification_metrics

        # Extract features and labels
        X, y = self.data.to_X_y(target_col=label_column)

        # Train model
        pipeline = make_classifier(
            X,
            y,
            algorithm=algorithm,
            cv_folds=cv_folds,
            **kwargs,
        )

        # Compute metrics
        cv_metrics = compute_classification_metrics(pipeline, X, y, cv=cv_folds)

        # Store model and metrics
        self.bundle.add_artifact("model", pipeline)
        self.bundle.add_metrics("cv_metrics", cv_metrics)

        # Log step
        self.bundle.run_record.add_step(
            "train",
            hashlib.sha256(json.dumps({"algorithm": algorithm}).encode()).hexdigest()[:8],
            metadata={"algorithm": algorithm, "cv_folds": cv_folds},
        )

        self._steps_applied.append(f"train({algorithm})")
        return self

    def library_similarity(
        self,
        library: FoodSpectrumSet,
        metric: Literal["euclidean", "cosine", "pearson", "sid", "sam"] = "cosine",
        top_k: int = 5,
        add_conf: bool = True,
    ) -> pd.DataFrame:
        """Run a similarity search against a reference library and record outputs.

        Records a similarity table and an overlay plot (query 0 vs. its top match)
        into the OutputBundle diagnostics.

        Parameters
        ----------
        library : FoodSpectrumSet
            Reference spectra library.
        metric : str
            Distance metric (euclidean, cosine, pearson, sid, sam). Default 'cosine'.
        top_k : int
            Number of top matches to report per query. Default 5.
        add_conf : bool
            If True, append confidence and decision columns to the similarity table.

        Returns
        -------
        pandas.DataFrame
            Similarity table with distances (and confidence/decision if enabled).
        """
        from foodspec.features.confidence import add_confidence
        from foodspec.features.library import LibraryIndex, overlay_plot

        # Build library index and compute similarity table
        lib = LibraryIndex.from_dataset(library)
        query_ids = list(self.data.metadata.get("sample_id", pd.Series(np.arange(len(self.data))).astype(str)))
        sim_table = lib.search(self.data.x, metric=metric, top_k=top_k, query_ids=query_ids)

        # Confidence and decision mapping
        if add_conf:
            sim_table = add_confidence(sim_table, metric=metric)

        # Add diagnostics
        self.bundle.add_diagnostic("similarity_table", sim_table)

        # Overlay plot for first query vs its top-1 match (if any)
        try:
            first_q = 0
            top_row = sim_table[sim_table["query_index"] == first_q].sort_values("rank").iloc[0]
            fig, ax = overlay_plot(
                self.data.x[first_q],
                lib.X[int(top_row["library_index"])],
                self.data.wavenumbers,
            )
            self.bundle.add_diagnostic("overlay_query0_top1", fig)
        except Exception:
            # Non-critical; skip if plotting fails
            pass

        # Log step
        self.bundle.run_record.add_step(
            "library_similarity",
            hashlib.sha256(json.dumps({"metric": metric, "top_k": top_k}, sort_keys=True).encode()).hexdigest()[:8],
            metadata={"metric": metric, "top_k": top_k},
        )
        self._steps_applied.append(f"library_similarity({metric},k={top_k})")

        return sim_table

    def export(
        self,
        path: Optional[Union[str, Path]] = None,
        formats: Optional[list] = None,
    ) -> Path:
        """Export all outputs to disk.

        Parameters
        ----------
        path : Path or str, optional
            Output directory. If None, uses self.output_dir.
        formats : list, optional
            Export formats. Default: ['json', 'csv', 'png', 'joblib'].

        Returns
        -------
        Path
            Directory containing all outputs.
        """
        path = path or self.output_dir
        return self.bundle.export(path, formats=formats)
