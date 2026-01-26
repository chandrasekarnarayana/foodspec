"""
Comprehensive tests for Phase 3 Research Platform.

Tests scientific validity of:
- MCR-ALS decomposition
- Bayesian chemometrics
- Physics-informed losses
- Digital twin simulator
- Active DOE
- Knowledge graph
- Research outputs
"""

import numpy as np
import pytest
from scipy.stats import norm

# MCR-ALS tests
from foodspec.features.mcr_als import MCRALS, RotationalAmbiguityAnalysis


class TestMCRALS:
    """Test MCR-ALS spectral decomposition."""
    
    def test_mcr_initialization(self):
        """Test MCR-ALS initializes correctly."""
        mcr = MCRALS(n_components=2, max_iter=10)
        assert mcr.n_components == 2
        assert mcr.max_iter == 10
    
    def test_mcr_fit_convergence(self):
        """Test MCR-ALS converges on synthetic mixture."""
        # Create synthetic mixture data
        np.random.seed(42)
        n_samples, n_features = 50, 100
        n_components = 2
        
        # True components
        C_true = np.random.rand(n_samples, n_components)
        C_true = C_true / C_true.sum(axis=1, keepdims=True)
        
        ST_true = np.random.rand(n_components, n_features)
        ST_true = np.maximum(ST_true, 0)
        
        X = C_true @ ST_true + np.random.randn(n_samples, n_features) * 0.01
        
        # Fit MCR
        mcr = MCRALS(n_components=2, max_iter=50, tol=1e-6)
        mcr.fit(X)
        
        assert mcr.converged_
        assert mcr.n_iter_ < 50
        assert mcr.components_.shape == (n_components, n_features)
    
    def test_mcr_transform(self):
        """Test MCR-ALS transforms data correctly."""
        np.random.seed(42)
        X = np.random.rand(50, 100)
        
        mcr = MCRALS(n_components=2, max_iter=20)
        mcr.fit(X)
        
        C = mcr.transform(X)
        assert C.shape == (50, 2)
        assert np.all(C >= 0)  # Non-negativity constraint
    
    def test_mcr_reconstruction(self):
        """Test MCR-ALS reconstructs data well."""
        np.random.seed(42)
        X = np.random.rand(50, 100) + 1.0  # Positive data
        
        mcr = MCRALS(n_components=3, max_iter=30)
        mcr.fit(X)
        
        r2 = mcr.score(X)
        assert r2 > 0.8  # Should explain >80% variance
    
    def test_rotational_ambiguity(self):
        """Test rotational ambiguity analysis."""
        np.random.seed(42)
        X = np.random.rand(40, 80) + 0.5
        
        mcr = MCRALS(n_components=2, max_iter=20, random_state=42)
        mcr.fit(X)
        
        ambiguity = RotationalAmbiguityAnalysis(mcr)
        ambiguity.compute_band_boundaries(X, n_rotations=20)
        
        assert hasattr(ambiguity, 'c_min_')
        assert hasattr(ambiguity, 'c_max_')
        assert 0 <= ambiguity.ambiguity_index_c_ <= 1
        assert 0 <= ambiguity.ambiguity_index_st_ <= 1


# Bayesian chemometrics tests
from foodspec.modeling.bayesian.bayesian_pls import BayesianPLS, BayesianNNLS, VariationalPLS


class TestBayesianPLS:
    """Test Bayesian PLS regression."""
    
    def test_bayesian_pls_fit(self):
        """Test Bayesian PLS fits data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = X[:, :5].sum(axis=1) + np.random.randn(100) * 0.1
        
        bpls = BayesianPLS(n_components=5, n_samples=200, burn_in=50, random_state=42)
        bpls.fit(X, y)
        
        assert hasattr(bpls, 'coef_samples_')
        assert bpls.coef_samples_.shape[0] == 150  # n_samples - burn_in
        assert bpls.coef_samples_.shape[1] == 50   # n_features
    
    def test_bayesian_pls_predict_with_uncertainty(self):
        """Test Bayesian PLS predictions with uncertainty."""
        np.random.seed(42)
        X_train = np.random.randn(80, 30)
        y_train = X_train[:, :3].sum(axis=1) + np.random.randn(80) * 0.1
        
        X_test = np.random.randn(20, 30)
        
        bpls = BayesianPLS(n_components=3, n_samples=100, burn_in=20, random_state=42)
        bpls.fit(X_train, y_train)
        
        y_pred, y_std = bpls.predict(X_test, return_std=True)
        
        assert y_pred.shape == (20,)
        assert y_std.shape == (20,)
        assert np.all(y_std > 0)  # Positive uncertainty
    
    def test_bayesian_pls_credible_intervals(self):
        """Test credible interval coverage."""
        np.random.seed(42)
        X = np.random.randn(100, 40)
        y = X[:, :5].mean(axis=1) + np.random.randn(100) * 0.2
        
        bpls = BayesianPLS(n_components=5, n_samples=150, burn_in=30, random_state=42)
        bpls.fit(X, y)
        
        y_lower, y_upper = bpls.predict_interval(X, credibility=0.95)
        
        # Check coverage (should be ~95%)
        coverage = np.mean((y >= y_lower) & (y <= y_upper))
        assert 0.90 <= coverage <= 1.0  # Allow some slack


class TestBayesianNNLS:
    """Test Bayesian NNLS."""
    
    def test_bayesian_nnls_fit(self):
        """Test Bayesian NNLS fits with non-negativity."""
        np.random.seed(42)
        # Pure component spectra (rows)
        S = np.abs(np.random.randn(3, 50))
        # Concentrations (non-negative)
        c_true = np.abs(np.random.randn(3))
        # Mixture
        y = S.T @ c_true + np.random.randn(50) * 0.01
        
        bnnls = BayesianNNLS(n_samples=200, burn_in=50, random_state=42)
        bnnls.fit(S.T, y)
        
        assert np.all(bnnls.coef_ >= 0)  # Non-negative coefficients
    
    def test_bayesian_nnls_uncertainty(self):
        """Test Bayesian NNLS provides uncertainty."""
        np.random.seed(42)
        X = np.abs(np.random.randn(40, 5))
        y = X.sum(axis=1) + np.random.randn(40) * 0.05
        
        bnnls = BayesianNNLS(n_samples=150, burn_in=30, random_state=42)
        bnnls.fit(X, y)
        
        y_pred, y_std = bnnls.predict(X, return_std=True)
        assert np.all(y_std > 0)


class TestVariationalPLS:
    """Test Variational PLS."""
    
    def test_variational_pls_fit_fast(self):
        """Test Variational PLS fits quickly."""
        import time
        
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = X[:, :3].sum(axis=1) + np.random.randn(100) * 0.1
        
        vpls = VariationalPLS(n_components=5, max_iter=50)
        
        start = time.time()
        vpls.fit(X, y)
        elapsed = time.time() - start
        
        assert elapsed < 5.0  # Should be fast (<5 seconds)
        assert hasattr(vpls, 'coef_mean_')
        assert hasattr(vpls, 'elbo_')
    
    def test_variational_pls_convergence(self):
        """Test Variational PLS converges (ELBO increases)."""
        np.random.seed(42)
        X = np.random.randn(80, 40)
        y = X[:, :4].mean(axis=1)
        
        vpls = VariationalPLS(n_components=4, max_iter=100, tol=1e-4)
        vpls.fit(X, y)
        
        # ELBO should generally increase
        elbo_diffs = np.diff(vpls.elbo_)
        assert np.mean(elbo_diffs > 0) > 0.7  # Most iterations increase ELBO


# Physics-informed ML tests
from foodspec.hybrid.physics_loss import (
    PhysicsInformedLoss, BeerLambertLoss, SmoothnessLoss,
    PeakConstraintLoss, SparsityLoss
)


class TestPhysicsInformedLoss:
    """Test physics-informed loss functions."""
    
    def test_beer_lambert_loss(self):
        """Test Beer-Lambert law enforcement."""
        np.random.seed(42)
        # Pure spectra
        pure_spectra = np.random.rand(2, 50)
        # Concentrations
        c = np.array([[0.3, 0.7], [0.5, 0.5]])
        # True mixtures
        y_true = c @ pure_spectra
        
        # Predicted (slightly off)
        y_pred = y_true + np.random.randn(*y_true.shape) * 0.01
        
        bl_loss = BeerLambertLoss(reference_spectra=pure_spectra, weight=1.0)
        loss = bl_loss.compute_loss(c, y_pred, y_true)
        
        assert loss >= 0
        assert loss < 0.1  # Small loss since y_pred close to y_true
    
    def test_smoothness_loss(self):
        """Test smoothness constraint."""
        # Smooth spectrum
        x_smooth = np.sin(np.linspace(0, 4 * np.pi, 100))
        # Noisy spectrum
        x_noisy = x_smooth + np.random.randn(100) * 0.5
        
        smooth_loss = SmoothnessLoss(order=2, weight=1.0)
        
        loss_smooth = smooth_loss.compute_loss(None, x_smooth.reshape(1, -1))
        loss_noisy = smooth_loss.compute_loss(None, x_noisy.reshape(1, -1))
        
        assert loss_noisy > loss_smooth  # Noisy has higher penalty
    
    def test_peak_constraint_loss(self):
        """Test peak shape constraint."""
        np.random.seed(42)
        # Spectrum with Gaussian peaks
        x = np.arange(100)
        spectrum = np.exp(-((x - 30) ** 2) / 50) + np.exp(-((x - 70) ** 2) / 50)
        
        peak_loss = PeakConstraintLoss(peak_positions=[30, 70], weight=1.0)
        loss = peak_loss.compute_loss(None, spectrum.reshape(1, -1))
        
        assert loss >= 0
    
    def test_sparsity_loss(self):
        """Test sparsity constraint."""
        # Sparse vector
        x_sparse = np.array([1.0, 0, 0, 2.0, 0, 0, 0, 0, 3.0, 0])
        # Dense vector
        x_dense = np.ones(10)
        
        sparsity_loss = SparsityLoss(weight=1.0)
        
        loss_sparse = sparsity_loss.compute_loss(None, x_sparse.reshape(1, -1))
        loss_dense = sparsity_loss.compute_loss(None, x_dense.reshape(1, -1))
        
        assert loss_dense > loss_sparse  # Dense has higher penalty
    
    def test_composite_physics_loss(self):
        """Test combined physics constraints."""
        np.random.seed(42)
        X = np.random.rand(10, 50)
        y_pred = np.random.rand(10, 50)
        
        physics_loss = PhysicsInformedLoss()
        physics_loss.add_constraint(SmoothnessLoss(order=2, weight=0.1))
        physics_loss.add_constraint(SparsityLoss(weight=0.05))
        
        total_loss = physics_loss(X, y_pred)
        assert total_loss >= 0
        
        losses_dict = physics_loss.get_constraint_losses(X, y_pred)
        assert 'Smoothness' in losses_dict
        assert 'Sparsity' in losses_dict


# Simulation tests
from foodspec.simulation.spectra_sim import (
    SpectraSimulator, NoiseModel, InstrumentModel, DomainShiftGenerator
)


class TestSpectraSimulator:
    """Test digital twin simulator."""
    
    def test_simulator_initialization(self):
        """Test simulator initializes correctly."""
        sim = SpectraSimulator(n_wavelengths=100, random_state=42)
        assert sim.n_wavelengths == 100
        assert len(sim.wavelengths) == 100
    
    def test_generate_pure_component(self):
        """Test pure component generation."""
        sim = SpectraSimulator(n_wavelengths=200, random_state=42)
        spectrum = sim.generate_pure_component(peak_positions=[50, 100, 150])
        
        assert spectrum.shape == (200,)
        assert np.max(spectrum) > 0
    
    def test_generate_mixture_dataset(self):
        """Test synthetic mixture generation."""
        sim = SpectraSimulator(n_wavelengths=150, random_state=42)
        X, y, meta = sim.generate_mixture_dataset(n_samples=50, n_components=3)
        
        assert X.shape == (50, 150)
        assert y.shape == (50, 3)
        assert 'pure_spectra' in meta
        assert meta['pure_spectra'].shape == (3, 150)
    
    def test_noise_models(self):
        """Test different noise models."""
        sim = SpectraSimulator(n_wavelengths=100, random_state=42)
        
        # Add Gaussian noise
        sim.add_noise_model(NoiseModel('gaussian', std=0.01))
        X, y, _ = sim.generate_mixture_dataset(n_samples=10, n_components=2)
        
        # Data should have noise
        assert X.std() > 0
    
    def test_instrument_model(self):
        """Test instrument response."""
        sim = SpectraSimulator(n_wavelengths=100, random_state=42)
        
        instrument = InstrumentModel(
            resolution=2.0,
            baseline_drift=0.05,
            intensity_scale=1.1
        )
        sim.set_instrument_model(instrument)
        
        X, _, _ = sim.generate_mixture_dataset(n_samples=10, n_components=2)
        assert X.mean() != 0  # Baseline drift and scaling applied


class TestDomainShiftGenerator:
    """Test domain shift generators."""
    
    def test_temperature_shift(self):
        """Test temperature-induced domain shift."""
        np.random.seed(42)
        X_original = np.random.rand(20, 100)
        
        shift_gen = DomainShiftGenerator('temperature', magnitude=10.0, random_state=42)
        X_shifted = shift_gen.apply_shift(X_original)
        
        # Should be different but similar
        assert not np.allclose(X_original, X_shifted)
        assert np.corrcoef(X_original.ravel(), X_shifted.ravel())[0, 1] > 0.9
    
    def test_instrument_shift(self):
        """Test instrument-to-instrument transfer."""
        np.random.seed(42)
        X_original = np.random.rand(15, 80)
        
        shift_gen = DomainShiftGenerator('instrument', magnitude=1.0, random_state=42)
        X_shifted = shift_gen.apply_shift(X_original)
        
        assert X_shifted.shape == X_original.shape
        assert not np.array_equal(X_original, X_shifted)


# Active DOE tests
from foodspec.doe.active_design import ActiveDesign, BayesianOptimizer, AcquisitionFunction


class TestActiveDesign:
    """Test active learning design."""
    
    def test_bayesian_optimizer_initialization(self):
        """Test Bayesian optimizer initializes."""
        opt = BayesianOptimizer(acquisition='ei', random_state=42)
        assert opt.acquisition == 'ei'
    
    def test_active_design_suggest(self):
        """Test active design suggests samples."""
        np.random.seed(42)
        X_candidates = np.random.rand(100, 10)
        
        design = ActiveDesign(acquisition='ei', random_state=42)
        X_suggested = design.suggest(X_candidates, n_suggestions=5)
        
        assert X_suggested.shape == (5, 10)
    
    def test_active_design_update(self):
        """Test active design updates with observations."""
        np.random.seed(42)
        X_candidates = np.random.rand(50, 5)
        
        design = ActiveDesign(random_state=42)
        X_batch1 = design.suggest(X_candidates, n_suggestions=3)
        
        # Simulate measurements
        y_batch1 = np.random.rand(3)
        design.update(X_batch1, y_batch1)
        
        # Next batch should use observations
        X_batch2 = design.suggest(X_candidates, n_suggestions=3)
        assert not np.array_equal(X_batch1, X_batch2)


# Knowledge graph tests
from foodspec.knowledge.graph import SpectralKnowledgeGraph, CompoundPeakLink


class TestSpectralKnowledgeGraph:
    """Test spectral knowledge graph."""
    
    def test_kg_initialization(self):
        """Test knowledge graph initializes."""
        kg = SpectralKnowledgeGraph('TestKG')
        assert kg.name == 'TestKG'
    
    def test_add_compound(self):
        """Test adding compounds."""
        kg = SpectralKnowledgeGraph()
        kg.add_compound('glucose', name='D-Glucose', formula='C6H12O6')
        
        assert 'glucose' in kg.compounds
        assert kg.compounds['glucose']['formula'] == 'C6H12O6'
    
    def test_add_link(self):
        """Test adding compound-peak links."""
        kg = SpectralKnowledgeGraph()
        link = CompoundPeakLink(
            compound_id='glucose',
            peak_wavelength=1030.0,
            assignment='C-O stretch',
            confidence=0.95
        )
        kg.add_link(link)
        
        assert 1030.0 in kg.peaks
        assert len(kg.peaks[1030.0]) == 1
    
    def test_query_by_peak(self):
        """Test querying by peak wavelength."""
        kg = SpectralKnowledgeGraph()
        kg.add_link(CompoundPeakLink('glucose', 1030.0))
        kg.add_link(CompoundPeakLink('fructose', 1032.0))
        
        results = kg.query_by_peak(1031.0, tolerance=5.0)
        assert len(results) == 2
    
    def test_export_json(self):
        """Test JSON export."""
        kg = SpectralKnowledgeGraph('FoodKG')
        kg.add_compound('glucose')
        kg.add_link(CompoundPeakLink('glucose', 1030.0))
        
        json_str = kg.to_json()
        assert 'glucose' in json_str
        assert '1030' in json_str
    
    def test_rdf_export(self):
        """Test RDF export."""
        kg = SpectralKnowledgeGraph()
        kg.add_compound('glucose', name='Glucose')
        kg.add_link(CompoundPeakLink('glucose', 1030.0))
        
        rdf_str = kg.to_rdf()
        assert 'fs:glucose' in rdf_str
        assert 'fs:wavelength' in rdf_str


# Research outputs tests
from foodspec.reporting.research_outputs import (
    DatasetCard, ReproducibilityPackage, MethodsSectionGenerator, ResearchOutputGenerator
)


class TestResearchOutputs:
    """Test research output generation."""
    
    def test_dataset_card_creation(self):
        """Test dataset card creation."""
        card = DatasetCard(
            name='TestDataset',
            version='1.0',
            description='Test dataset for validation',
            n_samples=100,
            n_features=200,
            feature_type='NIR spectra',
            target_type='concentration',
        )
        
        assert card.name == 'TestDataset'
        assert card.n_samples == 100
        
        card_dict = card.to_dict()
        assert 'dataset_info' in card_dict
    
    def test_methods_section_generation(self):
        """Test methods section auto-generation."""
        gen = MethodsSectionGenerator()
        
        methods = gen.generate(
            dataset_info={'n_samples': 100, 'n_features': 200},
            preprocessing=['SNV', 'Savitzky-Golay'],
            model_type='PLS',
            model_params={'n_components': 5},
            validation_strategy='5-fold CV',
        )
        
        assert 'SNV' in methods
        assert 'PLS' in methods
        assert '5-fold CV' in methods
    
    def test_reproducibility_package(self):
        """Test reproducibility package creation."""
        card = DatasetCard(
            name='TestDataset', version='1.0', description='Test',
            n_samples=50, n_features=100, feature_type='Raman', target_type='class'
        )
        
        package = ReproducibilityPackage(
            title='Test Study',
            authors=['Author1', 'Author2'],
            date='2026-01-26',
            dataset_cards=[card],
            methods_section='Test methods',
        )
        
        assert package.title == 'Test Study'
        assert len(package.authors) == 2
    
    def test_research_output_generator(self):
        """Test complete research output generation."""
        np.random.seed(42)
        X = np.random.rand(50, 100)
        y = np.random.rand(50)
        
        gen = ResearchOutputGenerator()
        
        card = gen.create_dataset_card(
            X, y,
            name='TestDataset',
            description='Synthetic test data',
            feature_type='NIR spectra',
        )
        
        assert card.n_samples == 50
        assert card.n_features == 100
        assert 0 <= card.missing_data_fraction <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
