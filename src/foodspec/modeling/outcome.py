"""Outcome type definitions for modeling flows."""

from __future__ import annotations

from enum import Enum


class OutcomeType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    COUNT = "count"
    SURVIVAL = "survival"


__all__ = ["OutcomeType"]
