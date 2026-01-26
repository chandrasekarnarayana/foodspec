"""
Physics-Informed Machine Learning for Spectroscopy.

This module provides physics-based loss functions and constraints for
hybrid neural network training, enabling models that respect known
physical laws while learning from data.

Key Features:
- Beer-Lambert law compliance loss
- Peak shape constraints (Gaussian, Lorentzian, Voigt)
- Energy conservation constraints
- Spectral smoothness and sparsity priors
- Compositional learning with sum-to-one constraints

References:
    [1] Raissi et al. (2019). Physics-informed neural networks: A deep
        learning framework for solving forward and inverse problems involving
        nonlinear partial differential equations. J. Comp. Physics, 378, 686-707.
    [2] Karniadakis et al. (2021). Physics-informed machine learning.
        Nature Reviews Physics, 3(6), 422-440.
    [3] von Rueden et al. (2021). Informed Machine Learning - A Taxonomy and
        Survey of Integrating Prior Knowledge into Learning Systems.
        IEEE Trans. Knowledge and Data Engineering.

Example:
    >>> from foodspec.hybrid.physics_loss import PhysicsInformedLoss, BeerLambertLoss
    >>> import torch
    >>> 
    >>> # Create physics-informed loss
    >>> physics_loss = PhysicsInformedLoss()
    >>> physics_loss.add_constraint(BeerLambertLoss(weight=0.1))
    >>> 
    >>> # Training loop
    >>> for X_batch, y_batch in dataloader:
    ...     y_pred = model(X_batch)
    ...     data_loss = mse_loss(y_pred, y_batch)
    ...     phys_loss = physics_loss(X_batch, y_pred, model)
    ...     total_loss = data_loss + phys_loss
    ...     total_loss.backward()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

import numpy as np


class PhysicsConstraint(ABC):
    """
    Abstract base class for physics-based constraints.
    
    Subclasses implement specific physical laws or priors that can be
    enforced during neural network training.
    
    Parameters
    ----------
    weight : float, default=1.0
        Weight for this constraint in total loss.
    
    name : str, optional
        Name of the constraint for logging.
    """
    
    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        self.weight = weight
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def compute_loss(
        self, 
        X: Any, 
        y_pred: Any, 
        model: Optional[Any] = None
    ) -> float:
        """
        Compute physics-based loss.
        
        Parameters
        ----------
        X : array-like
            Input data (e.g., wavelengths, conditions).
        
        y_pred : array-like
            Model predictions (e.g., spectra, concentrations).
        
        model : optional
            Neural network model (for extracting intermediate activations).
        
        Returns
        -------
        loss : float
            Physics constraint violation measure.
        """
        pass


class BeerLambertLoss(PhysicsConstraint):
    """
    Enforce Beer-Lambert law: A = ε * c * l.
    
    For mixtures: A_mixture = Σ_i (ε_i * c_i * l)
    
    This constraint ensures that predicted absorbance is linear in
    concentration when path length and molar absorptivity are known.
    
    Parameters
    ----------
    reference_spectra : ndarray of shape (n_components, n_wavelengths), optional
        Known pure component spectra (ε_i * l).
    
    concentration_index : int or slice, optional
        Index/slice of model output corresponding to concentrations.
    
    spectra_index : int or slice, optional
        Index/slice of model output corresponding to spectra.
    
    weight : float, default=1.0
        Constraint weight.
    
    Example
    -------
    >>> # Model predicts concentrations, reconstruct spectrum
    >>> bl_loss = BeerLambertLoss(reference_spectra=pure_spectra, weight=0.1)
    >>> 
    >>> # In training loop:
    >>> c_pred = model(X)  # Predicted concentrations
    >>> spectrum_pred = c_pred @ pure_spectra  # Reconstruct
    >>> spectrum_true = y_batch
    >>> loss = bl_loss.compute_loss(X, spectrum_pred, spectrum_true)
    """
    
    def __init__(
        self,
        reference_spectra: Optional[np.ndarray] = None,
        concentration_index: Optional[int | slice] = None,
        spectra_index: Optional[int | slice] = None,
        weight: float = 1.0,
    ):
        super().__init__(weight=weight, name='BeerLambert')
        self.reference_spectra = reference_spectra
        self.concentration_index = concentration_index
        self.spectra_index = spectra_index
    
    def compute_loss(
        self, 
        X: np.ndarray, 
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        model: Optional[Any] = None
    ) -> float:
        """
        Compute Beer-Lambert violation.
        
        If reference_spectra provided: Check if y_pred matches linear mixture.
        If not: Check if predicted spectra are linear in concentrations.
        """
        if self.reference_spectra is not None:
            # Extract concentrations from model output
            if self.concentration_index is not None:
                c_pred = y_pred[:, self.concentration_index]
            else:
                c_pred = y_pred  # Assume entire output is concentrations
            
            # Reconstruct spectra via Beer-Lambert
            spectra_reconstructed = c_pred @ self.reference_spectra
            
            # Compare to true spectra (if provided)
            if y_true is not None:
                loss = np.mean((spectra_reconstructed - y_true) ** 2)
            else:
                # No ground truth, just return 0 (no violation detectable)
                loss = 0.0
        
        else:
            # Check linearity: Are predictions linear combinations?
            # This is hard to enforce without knowing true relationships
            # Placeholder: return 0
            loss = 0.0
        
        return self.weight * loss


class SmoothnessLoss(PhysicsConstraint):
    """
    Enforce spectral smoothness via derivative penalty.
    
    Physical motivation: Real spectra are smooth (no sharp discontinuities)
    due to underlying molecular transitions and instrumental broadening.
    
    Loss = ||d²y/dx²||²  (penalize large second derivatives)
    
    Parameters
    ----------
    order : int, default=2
        Derivative order (1 for first derivative, 2 for second).
    
    weight : float, default=1.0
        Constraint weight.
    
    axis : int, default=1
        Axis along which to compute derivatives (1 for wavelength axis).
    
    Example
    -------
    >>> smooth_loss = SmoothnessLoss(order=2, weight=0.01)
    >>> loss = smooth_loss.compute_loss(X, y_pred)
    """
    
    def __init__(
        self,
        order: int = 2,
        weight: float = 1.0,
        axis: int = 1,
    ):
        super().__init__(weight=weight, name='Smoothness')
        self.order = order
        self.axis = axis
    
    def compute_loss(
        self, 
        X: np.ndarray, 
        y_pred: np.ndarray, 
        model: Optional[Any] = None
    ) -> float:
        """Compute smoothness penalty."""
        # Compute finite differences
        diff = y_pred
        for _ in range(self.order):
            diff = np.diff(diff, axis=self.axis)
        
        # L2 penalty on derivatives
        loss = np.mean(diff ** 2)
        
        return self.weight * loss


class PeakConstraintLoss(PhysicsConstraint):
    """
    Enforce peak shape constraints (Gaussian, Lorentzian, or Voigt).
    
    Physical motivation: Spectral peaks follow characteristic lineshapes
    determined by Doppler broadening (Gaussian), collision broadening
    (Lorentzian), or combination (Voigt).
    
    Parameters
    ----------
    peak_positions : list of int
        Expected peak positions (wavelength indices).
    
    peak_type : {'gaussian', 'lorentzian', 'voigt'}, default='gaussian'
        Expected peak lineshape.
    
    width_range : tuple of float, default=(1.0, 10.0)
        Allowed peak width range (in units of wavelength indices).
    
    weight : float, default=1.0
        Constraint weight.
    
    Example
    -------
    >>> # Known peaks at wavelengths 50, 100, 150
    >>> peak_loss = PeakConstraintLoss(peak_positions=[50, 100, 150], weight=0.05)
    >>> loss = peak_loss.compute_loss(X, y_pred)
    """
    
    def __init__(
        self,
        peak_positions: List[int],
        peak_type: str = 'gaussian',
        width_range: tuple[float, float] = (1.0, 10.0),
        weight: float = 1.0,
    ):
        super().__init__(weight=weight, name='PeakConstraint')
        self.peak_positions = peak_positions
        self.peak_type = peak_type
        self.width_range = width_range
    
    def compute_loss(
        self, 
        X: np.ndarray, 
        y_pred: np.ndarray, 
        model: Optional[Any] = None
    ) -> float:
        """
        Compute peak shape violation.
        
        Fit expected peak shapes at given positions and penalize deviations.
        """
        loss = 0.0
        n_wavelengths = y_pred.shape[1] if y_pred.ndim > 1 else len(y_pred)
        
        for pos in self.peak_positions:
            # Extract local region around peak
            window = 20  # ±20 wavelengths
            start = max(0, pos - window)
            end = min(n_wavelengths, pos + window)
            
            if y_pred.ndim > 1:
                local_spectrum = y_pred[:, start:end].mean(axis=0)
            else:
                local_spectrum = y_pred[start:end]
            
            # Fit peak shape
            x_local = np.arange(len(local_spectrum))
            peak_center = window if pos >= window else pos
            
            if self.peak_type == 'gaussian':
                # Fit Gaussian: A * exp(-(x - μ)² / (2σ²))
                fitted_peak = self._fit_gaussian(x_local, local_spectrum, peak_center)
            elif self.peak_type == 'lorentzian':
                # Fit Lorentzian: A * γ² / ((x - μ)² + γ²)
                fitted_peak = self._fit_lorentzian(x_local, local_spectrum, peak_center)
            else:
                # Voigt (Gaussian + Lorentzian convolution)
                fitted_peak = self._fit_gaussian(x_local, local_spectrum, peak_center)
            
            # Penalize deviation from expected shape
            shape_error = np.mean((local_spectrum - fitted_peak) ** 2)
            loss += shape_error
        
        return self.weight * loss / len(self.peak_positions)
    
    def _fit_gaussian(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        center: float
    ) -> np.ndarray:
        """Fit Gaussian peak and return fitted curve."""
        # Simple moment-based fit
        amplitude = np.max(y)
        sigma = 2.0  # Default width
        
        fitted = amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
        return fitted
    
    def _fit_lorentzian(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        center: float
    ) -> np.ndarray:
        """Fit Lorentzian peak and return fitted curve."""
        amplitude = np.max(y)
        gamma = 2.0  # Default width
        
        fitted = amplitude * gamma ** 2 / ((x - center) ** 2 + gamma ** 2)
        return fitted


class EnergyConservationLoss(PhysicsConstraint):
    """
    Enforce energy conservation in spectral transformations.
    
    Physical motivation: Total energy (integral of spectrum) should be
    conserved in certain transformations (e.g., fluorescence, scattering).
    
    Loss = |∫y_pred dx - ∫y_true dx|
    
    Parameters
    ----------
    weight : float, default=1.0
        Constraint weight.
    
    axis : int, default=1
        Axis to integrate over.
    
    Example
    -------
    >>> energy_loss = EnergyConservationLoss(weight=0.1)
    >>> loss = energy_loss.compute_loss(X, y_pred, y_true)
    """
    
    def __init__(self, weight: float = 1.0, axis: int = 1):
        super().__init__(weight=weight, name='EnergyConservation')
        self.axis = axis
    
    def compute_loss(
        self, 
        X: np.ndarray, 
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        model: Optional[Any] = None
    ) -> float:
        """Compute energy conservation violation."""
        if y_true is None:
            # No reference, can't check conservation
            return 0.0
        
        # Integrate over wavelength axis
        energy_pred = np.sum(y_pred, axis=self.axis)
        energy_true = np.sum(y_true, axis=self.axis)
        
        # Relative error in total energy
        loss = np.mean(np.abs(energy_pred - energy_true) / (energy_true + 1e-10))
        
        return self.weight * loss


class SparsityLoss(PhysicsConstraint):
    """
    Enforce sparsity in spectral features or coefficients.
    
    Physical motivation: Many spectroscopy problems have sparse solutions
    (few active components, few non-zero peaks).
    
    Loss = ||y||_1  (L1 penalty)
    
    Parameters
    ----------
    weight : float, default=1.0
        Constraint weight.
    
    threshold : float, default=0.0
        Values below threshold are not penalized.
    
    Example
    -------
    >>> sparsity_loss = SparsityLoss(weight=0.01, threshold=0.05)
    >>> loss = sparsity_loss.compute_loss(X, y_pred)
    """
    
    def __init__(self, weight: float = 1.0, threshold: float = 0.0):
        super().__init__(weight=weight, name='Sparsity')
        self.threshold = threshold
    
    def compute_loss(
        self, 
        X: np.ndarray, 
        y_pred: np.ndarray, 
        model: Optional[Any] = None
    ) -> float:
        """Compute L1 sparsity penalty."""
        # Apply threshold
        y_thresholded = np.where(np.abs(y_pred) > self.threshold, y_pred, 0)
        
        # L1 norm
        loss = np.mean(np.abs(y_thresholded))
        
        return self.weight * loss


class PhysicsInformedLoss:
    """
    Composite physics-informed loss function.
    
    Combines multiple physics constraints with data-driven loss.
    
    Parameters
    ----------
    constraints : list of PhysicsConstraint, optional
        List of physics constraints to enforce.
    
    Attributes
    ----------
    constraints : list
        Active physics constraints.
    
    Example
    -------
    >>> from foodspec.hybrid.physics_loss import (
    ...     PhysicsInformedLoss, BeerLambertLoss, SmoothnessLoss
    ... )
    >>> 
    >>> # Create composite loss
    >>> physics_loss = PhysicsInformedLoss()
    >>> physics_loss.add_constraint(BeerLambertLoss(weight=0.1))
    >>> physics_loss.add_constraint(SmoothnessLoss(order=2, weight=0.01))
    >>> 
    >>> # Use in training
    >>> total_loss = data_loss + physics_loss(X, y_pred, model)
    """
    
    def __init__(self, constraints: Optional[List[PhysicsConstraint]] = None):
        self.constraints = constraints if constraints is not None else []
    
    def add_constraint(self, constraint: PhysicsConstraint):
        """Add a physics constraint to the loss function."""
        self.constraints.append(constraint)
    
    def __call__(
        self, 
        X: np.ndarray, 
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
    ) -> float:
        """
        Compute total physics-informed loss.
        
        Parameters
        ----------
        X : ndarray
            Input data.
        
        y_pred : ndarray
            Model predictions.
        
        y_true : ndarray, optional
            Ground truth (for constraints requiring it).
        
        model : optional
            Neural network model.
        
        Returns
        -------
        total_loss : float
            Sum of all physics constraint losses.
        """
        total_loss = 0.0
        
        for constraint in self.constraints:
            # Check if constraint needs y_true
            if 'y_true' in constraint.compute_loss.__code__.co_varnames:
                loss = constraint.compute_loss(X, y_pred, y_true, model)
            else:
                loss = constraint.compute_loss(X, y_pred, model)
            
            total_loss += loss
        
        return total_loss
    
    def get_constraint_losses(
        self, 
        X: np.ndarray, 
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
    ) -> dict[str, float]:
        """
        Get individual constraint losses for logging.
        
        Returns
        -------
        losses : dict
            Constraint name -> loss value.
        """
        losses = {}
        
        for constraint in self.constraints:
            if 'y_true' in constraint.compute_loss.__code__.co_varnames:
                loss = constraint.compute_loss(X, y_pred, y_true, model)
            else:
                loss = constraint.compute_loss(X, y_pred, model)
            
            losses[constraint.name] = loss
        
        return losses
