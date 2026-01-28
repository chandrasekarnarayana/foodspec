"""Research Output Generation for Publications.

Auto-generates methods sections, reproducibility packages, and dataset cards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class DatasetCard:
    """Model card for datasets (inspired by Datasheets for Datasets).

    References:
        Gebru et al. (2021). Datasheets for Datasets. Commun. ACM, 64(12).
    """

    name: str
    version: str
    description: str
    n_samples: int
    n_features: int
    feature_type: str  # e.g., "NIR spectra", "Raman spectra"
    target_type: str  # e.g., "concentration", "classification"

    # Provenance
    collection_date: Optional[str] = None
    collection_method: Optional[str] = None
    instrument: Optional[str] = None

    # Characteristics
    wavelength_range: Optional[tuple] = None
    sample_types: List[str] = field(default_factory=list)
    preprocessing_applied: List[str] = field(default_factory=list)

    # Quality
    missing_data_fraction: float = 0.0
    outlier_fraction: float = 0.0
    quality_notes: Optional[str] = None

    # Usage
    intended_use: Optional[str] = None
    limitations: Optional[str] = None
    ethical_considerations: Optional[str] = None

    # Metadata
    created_by: Optional[str] = None
    license: str = "CC-BY-4.0"
    doi: Optional[str] = None

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "dataset_info": {
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "feature_type": self.feature_type,
                "target_type": self.target_type,
            },
            "provenance": {
                "collection_date": self.collection_date,
                "collection_method": self.collection_method,
                "instrument": self.instrument,
            },
            "characteristics": {
                "wavelength_range": self.wavelength_range,
                "sample_types": self.sample_types,
                "preprocessing": self.preprocessing_applied,
            },
            "quality": {
                "missing_data_fraction": self.missing_data_fraction,
                "outlier_fraction": self.outlier_fraction,
                "notes": self.quality_notes,
            },
            "usage": {
                "intended_use": self.intended_use,
                "limitations": self.limitations,
                "ethical_considerations": self.ethical_considerations,
            },
            "metadata": {
                "created_by": self.created_by,
                "license": self.license,
                "doi": self.doi,
            },
        }


@dataclass
class ReproducibilityPackage:
    """Complete reproducibility package for publication."""

    title: str
    authors: List[str]
    date: str

    # Code
    code_repository: Optional[str] = None
    commit_hash: Optional[str] = None
    requirements: List[str] = field(default_factory=list)

    # Data
    dataset_cards: List[DatasetCard] = field(default_factory=list)
    data_availability: Optional[str] = None

    # Methods
    methods_section: Optional[str] = None
    preprocessing_steps: List[str] = field(default_factory=list)
    model_hyperparameters: Dict = field(default_factory=dict)

    # Results
    random_seeds: List[int] = field(default_factory=list)
    computational_environment: Dict = field(default_factory=dict)

    def to_json(self, filepath: str):
        """Save as JSON."""
        data = {
            "title": self.title,
            "authors": self.authors,
            "date": self.date,
            "code": {
                "repository": self.code_repository,
                "commit_hash": self.commit_hash,
                "requirements": self.requirements,
            },
            "data": {
                "datasets": [ds.to_dict() for ds in self.dataset_cards],
                "availability": self.data_availability,
            },
            "methods": {
                "description": self.methods_section,
                "preprocessing": self.preprocessing_steps,
                "hyperparameters": self.model_hyperparameters,
            },
            "reproducibility": {
                "random_seeds": self.random_seeds,
                "environment": self.computational_environment,
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


class MethodsSectionGenerator:
    """Auto-generate methods sections for papers.

    Example
    -------
    >>> from foodspec.reporting import MethodsSectionGenerator
    >>>
    >>> gen = MethodsSectionGenerator()
    >>> methods_text = gen.generate(
    ...     dataset_info={'n_samples': 100, 'n_features': 200},
    ...     preprocessing=['SNV', 'Savitzky-Golay'],
    ...     model_type='PLS',
    ...     model_params={'n_components': 5},
    ...     validation_strategy='5-fold CV',
    ... )
    """

    def __init__(self):
        self.templates = {
            "dataset": "A dataset of {n_samples} samples with {n_features} spectral features was used.",
            "preprocessing": "Preprocessing steps included: {steps}.",
            "model": "{model_type} regression was performed with the following parameters: {params}.",
            "validation": "Model performance was evaluated using {strategy}.",
        }

    def generate(
        self,
        dataset_info: Dict,
        preprocessing: List[str],
        model_type: str,
        model_params: Dict,
        validation_strategy: str,
    ) -> str:
        """Generate methods section text."""
        sections = []

        # Dataset
        sections.append(
            self.templates["dataset"].format(
                n_samples=dataset_info.get("n_samples", "N"),
                n_features=dataset_info.get("n_features", "M"),
            )
        )

        # Preprocessing
        if preprocessing:
            steps_str = ", ".join(preprocessing)
            sections.append(self.templates["preprocessing"].format(steps=steps_str))

        # Model
        params_str = ", ".join([f"{k}={v}" for k, v in model_params.items()])
        sections.append(
            self.templates["model"].format(
                model_type=model_type,
                params=params_str,
            )
        )

        # Validation
        sections.append(
            self.templates["validation"].format(
                strategy=validation_strategy,
            )
        )

        return " ".join(sections)


class ResearchOutputGenerator:
    """Complete research output generation system.

    Example
    -------
    >>> from foodspec.reporting import ResearchOutputGenerator
    >>>
    >>> gen = ResearchOutputGenerator()
    >>>
    >>> # Create dataset card
    >>> dataset_card = gen.create_dataset_card(
    ...     X=X_train, y=y_train,
    ...     name='OliveOil_NIR',
    ...     description='NIR spectra of olive oil samples',
    ... )
    >>>
    >>> # Generate methods
    >>> methods = gen.generate_methods(
    ...     preprocessing_pipeline=pipeline,
    ...     model=model,
    ...     validation_results=cv_results,
    ... )
    >>>
    >>> # Create reproducibility package
    >>> package = gen.create_repro_package(
    ...     title='Olive Oil Classification',
    ...     authors=['Author1', 'Author2'],
    ...     dataset_cards=[dataset_card],
    ...     methods_section=methods,
    ... )
    >>> package.to_json('repro_package.json')
    """

    def __init__(self):
        self.methods_generator = MethodsSectionGenerator()

    def create_dataset_card(
        self,
        X,
        y,
        name: str,
        description: str,
        **kwargs,
    ) -> DatasetCard:
        """Create dataset card from data arrays."""
        import numpy as np

        n_samples, n_features = X.shape

        return DatasetCard(
            name=name,
            version="1.0",
            description=description,
            n_samples=n_samples,
            n_features=n_features,
            feature_type=kwargs.get("feature_type", "spectra"),
            target_type=kwargs.get("target_type", "regression"),
            missing_data_fraction=float(np.isnan(X).sum() / X.size),
            **{k: v for k, v in kwargs.items() if k not in ["feature_type", "target_type"]},
        )

    def generate_methods(
        self,
        preprocessing_pipeline,
        model,
        validation_results: Dict,
    ) -> str:
        """Generate methods section from pipeline and model."""
        dataset_info = validation_results.get("dataset_info", {})
        preprocessing = [step[0] for step in getattr(preprocessing_pipeline, "steps", [])]
        model_type = model.__class__.__name__
        model_params = model.get_params()
        validation_strategy = validation_results.get("cv_strategy", "cross-validation")

        return self.methods_generator.generate(
            dataset_info=dataset_info,
            preprocessing=preprocessing,
            model_type=model_type,
            model_params=model_params,
            validation_strategy=validation_strategy,
        )

    def create_repro_package(
        self,
        title: str,
        authors: List[str],
        dataset_cards: List[DatasetCard],
        methods_section: str,
        **kwargs,
    ) -> ReproducibilityPackage:
        """Create complete reproducibility package."""
        import platform
        import sys

        return ReproducibilityPackage(
            title=title,
            authors=authors,
            date=datetime.now().isoformat(),
            dataset_cards=dataset_cards,
            methods_section=methods_section,
            computational_environment={
                "python_version": sys.version,
                "platform": platform.platform(),
            },
            **kwargs,
        )
