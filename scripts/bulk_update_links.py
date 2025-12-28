#!/usr/bin/env python3
"""
Bulk link updater for docs restructure.
Fixes references to moved pages based on the new IA.
"""

import re
from pathlib import Path

# Old → New path mappings (relative to docs/)
LINK_MAPPINGS = {
    # Metrics & visualization
    r'\.\.\/metrics\/metrics_and_evaluation\.md': '../reference/metrics_reference.md',
    r'metrics\/metrics_and_evaluation\.md': 'reference/metrics_reference.md',
    r'\.\.\/visualization\/plotting_with_foodspec\.md': '../user-guide/visualization.md',
    r'visualization\/plotting_with_foodspec\.md': 'user-guide/visualization.md',
    
    # Theory pages
    r'\.\.\/07-theory-and-background\/rq_engine_theory\.md': '../theory/rq_engine_detailed.md',
    r'07-theory-and-background\/rq_engine_theory\.md': 'theory/rq_engine_detailed.md',
    r'\.\.\/07-theory-and-background\/harmonization_theory\.md': '../theory/harmonization_theory.md',
    r'07-theory-and-background\/harmonization_theory\.md': 'theory/harmonization_theory.md',
    r'\.\.\/07-theory-and-background\/chemometrics_and_ml_basics\.md': '../theory/chemometrics_and_ml_basics.md',
    r'07-theory-and-background\/chemometrics_and_ml_basics\.md': 'theory/chemometrics_and_ml_basics.md',
    
    # Cookbook / preprocessing
    r'\.\.\/03-cookbook\/cookbook_preprocessing\.md': '../methods/preprocessing/normalization_smoothing.md',
    r'03-cookbook\/cookbook_preprocessing\.md': 'methods/preprocessing/normalization_smoothing.md',
    r'\.\.\/03-cookbook\/preprocessing_guide\.md': '../methods/preprocessing/normalization_smoothing.md',
    r'03-cookbook\/preprocessing_guide\.md': 'methods/preprocessing/normalization_smoothing.md',
    r'\.\.\/03-cookbook\/chemometrics_guide\.md': '../methods/chemometrics/models_and_best_practices.md',
    r'03-cookbook\/chemometrics_guide\.md': 'methods/chemometrics/models_and_best_practices.md',
    r'\.\.\/03-cookbook\/ftir_raman_preprocessing\.md': '../methods/preprocessing/baseline_correction.md',
    r'03-cookbook\/ftir_raman_preprocessing\.md': 'methods/preprocessing/baseline_correction.md',
    r'\.\.\/03-cookbook\/cookbook_validation\.md': '../methods/validation/cross_validation_and_leakage.md',
    r'03-cookbook\/cookbook_validation\.md': 'methods/validation/cross_validation_and_leakage.md',
    r'\.\.\/03-cookbook\/validation_baseline\.md': '../methods/validation/robustness_checks.md',
    r'03-cookbook\/validation_baseline\.md': 'methods/validation/robustness_checks.md',
    r'\.\.\/03-cookbook\/cookbook_rq_questions\.md': '../theory/rq_engine_detailed.md',
    r'03-cookbook\/cookbook_rq_questions\.md': 'theory/rq_engine_detailed.md',
    r'\.\.\/03-cookbook\/index\.md': '../workflows/index.md',
    r'03-cookbook\/index\.md': 'workflows/index.md',
    
    # ML → methods/chemometrics
    r'\.\.\/ml\/classification_regression\.md': '../methods/chemometrics/classification_regression.md',
    r'ml\/classification_regression\.md': 'methods/chemometrics/classification_regression.md',
    r'\.\.\/ml\/model_evaluation_and_validation\.md': '../methods/chemometrics/model_evaluation_and_validation.md',
    r'ml\/model_evaluation_and_validation\.md': 'methods/chemometrics/model_evaluation_and_validation.md',
    r'\.\.\/ml\/mixture_models\.md': '../methods/chemometrics/mixture_models.md',
    r'ml\/mixture_models\.md': 'methods/chemometrics/mixture_models.md',
    r'\.\.\/ml\/models_and_best_practices\.md': '../methods/chemometrics/models_and_best_practices.md',
    r'ml\/models_and_best_practices\.md': 'methods/chemometrics/models_and_best_practices.md',
    r'\.\.\/ml\/pca_and_dimensionality_reduction\.md': '../methods/chemometrics/pca_and_dimensionality_reduction.md',
    r'ml\/pca_and_dimensionality_reduction\.md': 'methods/chemometrics/pca_and_dimensionality_reduction.md',
    r'\.\.\/ml\/model_interpretability\.md': '../methods/chemometrics/model_interpretability.md',
    r'ml\/model_interpretability\.md': 'methods/chemometrics/model_interpretability.md',
    
    # Preprocessing → methods/preprocessing
    r'\.\.\/preprocessing\/baseline_correction\.md': '../methods/preprocessing/baseline_correction.md',
    r'preprocessing\/baseline_correction\.md': 'methods/preprocessing/baseline_correction.md',
    r'\.\.\/preprocessing\/normalization_smoothing\.md': '../methods/preprocessing/normalization_smoothing.md',
    r'preprocessing\/normalization_smoothing\.md': 'methods/preprocessing/normalization_smoothing.md',
    r'\.\.\/preprocessing\/feature_extraction\.md': '../methods/preprocessing/feature_extraction.md',
    r'preprocessing\/feature_extraction\.md': 'methods/preprocessing/feature_extraction.md',
    r'\.\.\/preprocessing\/derivatives_and_feature_enhancement\.md': '../methods/preprocessing/derivatives_and_feature_enhancement.md',
    r'preprocessing\/derivatives_and_feature_enhancement\.md': 'methods/preprocessing/derivatives_and_feature_enhancement.md',
    
    # API paths (08-api → api)
    r'\.\.\/08-api\/stats\.md': '../api/stats.md',
    r'08-api\/stats\.md': 'api/stats.md',
    r'\.\.\/08-api\/metrics\.md': '../api/metrics.md',
    r'08-api\/metrics\.md': 'api/metrics.md',
    r'\.\.\/08-api\/core\.md': '../api/core.md',
    r'08-api\/core\.md': 'api/core.md',
    r'\.\.\/08-api\/preprocessing\.md': '../api/preprocessing.md',
    r'08-api\/preprocessing\.md': 'api/preprocessing.md',
    
    # 09-reference → reference
    r'\.\.\/09-reference\/glossary\.md': '../reference/glossary.md',
    r'09-reference\/glossary\.md': 'reference/glossary.md',
    r'\.\.\/09-reference\/metric_significance_tables\.md': '../reference/metric_significance_tables.md',
    r'09-reference\/metric_significance_tables\.md': 'reference/metric_significance_tables.md',
    
    # Stats paths
    r'\.\.\/stats\/hypothesis_testing_in_food_spectroscopy\.md': '../methods/statistics/hypothesis_testing_in_food_spectroscopy.md',
    r'stats\/hypothesis_testing_in_food_spectroscopy\.md': 'methods/statistics/hypothesis_testing_in_food_spectroscopy.md',
    r'\.\.\/stats\/anova_and_manova\.md': '../methods/statistics/anova_and_manova.md',
    r'stats\/anova_and_manova\.md': 'methods/statistics/anova_and_manova.md',
    r'\.\.\/stats\/nonparametric_methods_and_robustness\.md': '../methods/statistics/nonparametric_methods_and_robustness.md',
    r'stats\/nonparametric_methods_and_robustness\.md': 'methods/statistics/nonparametric_methods_and_robustness.md',
    r'\.\.\/stats\/overview\.md': '../methods/statistics/overview.md',
    r'stats\/overview\.md': 'methods/statistics/overview.md',
    r'\.\.\/stats\/study_design_and_data_requirements\.md': '../methods/statistics/study_design_and_data_requirements.md',
    r'stats\/study_design_and_data_requirements\.md': 'methods/statistics/study_design_and_data_requirements.md',
    r'\.\.\/stats\/t_tests_effect_sizes_and_power\.md': '../methods/statistics/t_tests_effect_sizes_and_power.md',
    r'stats\/t_tests_effect_sizes_and_power\.md': 'methods/statistics/t_tests_effect_sizes_and_power.md',
    
    # Workflows moved into subfolders
    r'\.\.\/workflows\/oil_authentication\.md': '../workflows/authentication/oil_authentication.md',
    r'workflows\/oil_authentication\.md': 'workflows/authentication/oil_authentication.md',
    r'\.\.\/workflows\/heating_quality_monitoring\.md': '../workflows/quality-monitoring/heating_quality_monitoring.md',
    r'workflows\/heating_quality_monitoring\.md': 'workflows/quality-monitoring/heating_quality_monitoring.md',
    r'\.\.\/workflows\/mixture_analysis\.md': '../workflows/quantification/mixture_analysis.md',
    r'workflows\/mixture_analysis\.md': 'workflows/quantification/mixture_analysis.md',
    r'\.\.\/workflows\/calibration_regression_example\.md': '../workflows/quantification/calibration_regression_example.md',
    r'workflows\/calibration_regression_example\.md': 'workflows/quantification/calibration_regression_example.md',
    r'\.\.\/workflows\/harmonization_automated_calibration\.md': '../workflows/harmonization/harmonization_automated_calibration.md',
    r'workflows\/harmonization_automated_calibration\.md': 'workflows/harmonization/harmonization_automated_calibration.md',
    r'\.\.\/workflows\/hyperspectral_mapping\.md': '../workflows/spatial/hyperspectral_mapping.md',
    r'workflows\/hyperspectral_mapping\.md': 'workflows/spatial/hyperspectral_mapping.md',
    
    # User guide
    r'cli_guide\.md': 'cli.md',
    r'logging\.md': 'config_logging.md',
    r'\.\.\/user_guide\/instrument_file_formats\.md': '../user-guide/vendor_formats.md',
    r'user_guide\/instrument_file_formats\.md': 'user-guide/vendor_formats.md',
    
    # Other common paths
    r'\.\.\/04-user-guide\/automation\.md': '../user-guide/automation.md',
    r'04-user-guide\/automation\.md': 'user-guide/automation.md',
    r'\.\.\/04-user-guide\/protocols_and_yaml\.md': '../user-guide/protocols_and_yaml.md',
    r'04-user-guide\/protocols_and_yaml\.md': 'user-guide/protocols_and_yaml.md',
    r'\.\.\/04-user-guide\/data_formats_and_hdf5\.md': '../user-guide/data_formats_and_hdf5.md',
    r'04-user-guide\/data_formats_and_hdf5\.md': 'user-guide/data_formats_and_hdf5.md',
    
    r'\.\.\/06-developer-guide\/contributing\.md': '../developer-guide/contributing.md',
    r'06-developer-guide\/contributing\.md': 'developer-guide/contributing.md',
    r'\.\.\/06-developer-guide\/documentation_guidelines\.md': '../developer-guide/documentation_guidelines.md',
    r'06-developer-guide\/documentation_guidelines\.md': 'developer-guide/documentation_guidelines.md',
    
    r'\.\.\/05-advanced-topics\/hsi_and_harmonization\.md': '../theory/harmonization_theory.md',
    r'05-advanced-topics\/hsi_and_harmonization\.md': 'theory/harmonization_theory.md',
    
    # Foundations
    r'\.\.\/foundations\/spectroscopy_basics\.md': '../theory/spectroscopy_basics.md',
    r'foundations\/spectroscopy_basics\.md': 'theory/spectroscopy_basics.md',
}


def update_links_in_file(file_path: Path) -> tuple[int, list[str]]:
    """Update links in a single markdown file."""
    try:
        content = file_path.read_text()
        original = content
        changes = []
        
        for old_pattern, new_path in LINK_MAPPINGS.items():
            # Match markdown links: [text](old_path) or [text](old_path#anchor)
            pattern = rf'(\[.*?\])\({old_pattern}(#[^\)]+)?\)'
            matches = re.findall(pattern, content)
            if matches:
                # Replace the link
                new_content = re.sub(
                    pattern,
                    rf'\1({new_path}\2)',
                    content
                )
                if new_content != content:
                    changes.append(f"{old_pattern} → {new_path}")
                    content = new_content
        
        if content != original:
            file_path.write_text(content)
            return len(changes), changes
        return 0, []
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, []


def main():
    docs_dir = Path('docs')
    if not docs_dir.exists():
        print("Error: docs/ directory not found")
        return
    
    total_files = 0
    total_changes = 0
    files_changed = []
    
    # Process all markdown files
    for md_file in docs_dir.rglob('*.md'):
        if '_internal' in str(md_file) or 'site/' in str(md_file):
            continue
        
        total_files += 1
        num_changes, changes = update_links_in_file(md_file)
        
        if num_changes > 0:
            total_changes += num_changes
            files_changed.append((md_file, changes))
            print(f"✓ {md_file.relative_to(docs_dir)}: {num_changes} links updated")
    
    print(f"\n{'='*70}")
    print(f"Processed {total_files} files")
    print(f"Updated {total_changes} links in {len(files_changed)} files")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
