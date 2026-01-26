from foodspec.engine.preprocessing.engine import (
    AlignmentStep,
    AutoPreprocess,
    AutoPreprocessResult,
    BaselineStep,
    DerivativeStep,
    NormalizationStep,
    PreprocessPipeline,
    ResampleStep,
    SmoothingStep,
    Step,
)
from foodspec.preprocess.recipes import PreprocessingRecipe

# New enhanced preprocessing modules
from foodspec.preprocess.data import SpectraData, load_csv, validate_modality
from foodspec.preprocess.spectroscopy_operators import (
    AtmosphericCorrectionOperator,
    DespikeOperator,
    EMSCOperator,
    FluorescenceRemovalOperator,
    InterpolationOperator,
    MSCOperator,
)
from foodspec.preprocess.loaders import (
    build_pipeline_from_recipe,
    list_operators,
    load_preset_yaml,
    load_recipe,
    merge_recipe,
)
from foodspec.preprocess.cache import (
    PreprocessCache,
    PreprocessManifest,
    compute_cache_key,
    compute_data_hash,
    compute_recipe_hash,
)
from foodspec.preprocess.qc import (
    generate_qc_report,
    plot_baseline_overlay,
    plot_outlier_summary,
    plot_raw_vs_processed,
)

__all__ = [
    # Core engine
    "AlignmentStep",
    "AutoPreprocess",
    "AutoPreprocessResult",
    "BaselineStep",
    "DerivativeStep",
    "NormalizationStep",
    "PreprocessPipeline",
    "ResampleStep",
    "SmoothingStep",
    "Step",
    "PreprocessingRecipe",
    # Data model
    "SpectraData",
    "load_csv",
    "validate_modality",
    # Spectroscopy operators
    "DespikeOperator",
    "FluorescenceRemovalOperator",
    "EMSCOperator",
    "MSCOperator",
    "AtmosphericCorrectionOperator",
    "InterpolationOperator",
    # Recipe loading
    "load_preset_yaml",
    "build_pipeline_from_recipe",
    "load_recipe",
    "merge_recipe",
    "list_operators",
    # Caching and provenance
    "PreprocessCache",
    "PreprocessManifest",
    "compute_data_hash",
    "compute_recipe_hash",
    "compute_cache_key",
    # QC visualization
    "plot_raw_vs_processed",
    "plot_baseline_overlay",
    "plot_outlier_summary",
    "generate_qc_report",
]
