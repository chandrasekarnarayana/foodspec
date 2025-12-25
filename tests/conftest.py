"""Pytest configuration for FoodSpec tests.

This file ensures pytest can discover and run tests in the reorganized
test directory structure which mirrors the source code structure.
"""

import sys
import os

# Add the src directory to path so imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Pytest plugins and fixtures
pytest_plugins = []
