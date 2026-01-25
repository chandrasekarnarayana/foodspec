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
Quick-start example for FoodSpec 2.0.
"""

from foodspec import Registry, Orchestrator
from foodspec.core import Preprocessor, FeatureExtractor, Model

# Example: How to use the rewritten FoodSpec


def example_registry():
    """Example 1: Using the component registry."""
    registry = Registry()
    
    # Register different baseline correction methods
    registry.register("als", "BaselineALS")
    registry.register("polynomial", "BaselinePolynomial")
    
    # Select at runtime
    method = "als"
    baseline = registry.get(method, lamb=100, p=0.001)
    print(f"✓ Created {method} preprocessor")


def example_orchestrator():
    """Example 2: Using the orchestrator for workflows."""
    
    workflow = Orchestrator()
    workflow.add("load", "LoadData(path='./data/oils.csv')")
    workflow.add("preprocess", "Preprocess(method='als')")
    workflow.add("extract", "FeatureExtraction(features=['peak_height', 'area'])")
    workflow.add("train", "TrainModel(algorithm='RandomForest')")
    
    # result = workflow.run()
    print("✓ Workflow chain created")


def example_protocols():
    """Example 3: Using protocols for flexible implementations."""
    
    # Any class that has wavenumbers and intensities properties
    # satisfies the Spectrum protocol—no explicit inheritance needed
    
    class MinimalSpectrum:
        """Minimal implementation satisfying Spectrum protocol."""
        
        @property
        def wavenumbers(self):
            return [400, 500, 600, 700]
        
        @property
        def intensities(self):
            return [100, 150, 200, 180]
    
    spectrum = MinimalSpectrum()
    print(f"✓ Spectrum created with {len(spectrum.wavenumbers)} points")


def example_artifacts():
    """Example 4: Collecting artifacts for reproducibility."""
    
    # Conceptual example (ArtifactBundle not yet implemented)
    artifacts = {
        "model": "trained_model_object",
        "metrics": {"accuracy": 0.95, "f1": 0.93},
        "manifest": {
            "version": "2.0.0",
            "date": "2025-01-24",
            "preprocessing": "als_baseline",
            "features": ["peak_1030", "peak_1050", "ratio_1030_1050"],
        }
    }
    
    print("✓ Artifacts collected:")
    for key in artifacts:
        print(f"  - {key}")


if __name__ == "__main__":
    print("🔬 FoodSpec 2.0 Quick-Start Examples\n")
    
    print("1. Component Registry")
    example_registry()
    print()
    
    print("2. Orchestrator Pattern")
    example_orchestrator()
    print()
    
    print("3. Protocol-Based Design")
    example_protocols()
    print()
    
    print("4. Artifact Collection")
    example_artifacts()
    print()
    
    print("✅ All examples completed!")
