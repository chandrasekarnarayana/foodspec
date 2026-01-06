#!/bin/bash
###############################################################################
#
# JOSS Reviewer Check Script
# 
# Minimal reproducibility check for FoodSpec v1.0.0
# - Creates a clean Python environment
# - Installs the package
# - Verifies core imports
# - Runs test suite
# - Builds documentation with strict mode
#
# Usage: bash scripts/joss_reviewer_check.sh
#
###############################################################################

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print with color
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
    exit 1
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

###############################################################################
# STEP 1: Create clean virtual environment
###############################################################################

print_step "Creating clean Python virtual environment..."
VENV_DIR="/tmp/foodspec_joss_check_venv"

if [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR" || print_error "Failed to create virtual environment"
source "$VENV_DIR/bin/activate"
print_success "Virtual environment created at $VENV_DIR"

###############################################################################
# STEP 2: Upgrade pip and install build dependencies
###############################################################################

print_step "Upgrading pip and installing build dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1 || print_error "Failed to upgrade pip"
print_success "pip upgraded"

###############################################################################
# STEP 3: Install FoodSpec and dependencies
###############################################################################

print_step "Installing FoodSpec package..."
cd "$(dirname "$0")/.." || print_error "Failed to change to repo directory"
pip install -e ".[test,docs]" > /dev/null 2>&1 || print_error "Failed to install FoodSpec"
print_success "FoodSpec installed successfully"

###############################################################################
# STEP 4: Minimal import check
###############################################################################

print_step "Running minimal import checks..."

python3 << 'PYEOF' || print_error "Import checks failed"
import sys

# Core imports
try:
    import foodspec
    print(f"✓ foodspec (v{foodspec.__version__})")
except ImportError as e:
    print(f"✗ Failed to import foodspec: {e}")
    sys.exit(1)

try:
    from foodspec import io, preprocess, ml, stats, validation
    print("✓ foodspec.io, preprocess, ml, stats, validation")
except ImportError as e:
    print(f"✗ Failed to import core modules: {e}")
    sys.exit(1)

try:
    from foodspec.io import load_csv_spectra, save_hdf5
    from foodspec.preprocess import baseline_als, normalize_snv
    from foodspec.ml import ClassifierFactory
    print("✓ Key I/O and preprocessing functions available")
except ImportError as e:
    print(f"✗ Failed to import key functions: {e}")
    sys.exit(1)

try:
    import numpy as np
    import pandas as pd
    import scikit_learn as sklearn
    print("✓ NumPy, pandas, scikit-learn available")
except ImportError as e:
    print(f"✗ Failed to import dependencies: {e}")
    sys.exit(1)

print("\nAll imports successful!")
PYEOF

print_success "Import checks passed"

###############################################################################
# STEP 5: Run test suite
###############################################################################

print_step "Running test suite (pytest)..."
print_info "Running: pytest tests/ --tb=short"

if pytest tests/ --tb=short -v 2>&1 | tee /tmp/pytest_output.log; then
    print_success "Test suite passed"
else
    print_error "Test suite failed (see output above)"
fi

###############################################################################
# STEP 6: Build documentation with strict mode
###############################################################################

print_step "Building documentation (mkdocs build --strict)..."
print_info "Running: mkdocs build --strict"

if mkdocs build --strict > /tmp/mkdocs_output.log 2>&1; then
    print_success "Documentation build passed"
else
    print_error "Documentation build failed"
fi

###############################################################################
# SUMMARY
###############################################################################

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  ✓ ALL CHECKS PASSED                          ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║ • Virtual environment created and activated                    ║"
echo "║ • Package installed successfully                               ║"
echo "║ • Core imports verified                                        ║"
echo "║ • Test suite passed                                            ║"
echo "║ • Documentation builds without errors                          ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║                FoodSpec is JOSS-ready!                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
print_info "Clean venv located at: $VENV_DIR"
print_info "To reuse: source $VENV_DIR/bin/activate"
print_info "To remove: rm -rf $VENV_DIR"
echo ""

###############################################################################
# CLEANUP (optional - deactivate venv)
###############################################################################

# Note: We keep the venv active so reviewer can inspect if needed
print_info "Virtual environment remains active for inspection"
print_info "Type 'deactivate' to exit venv when done"
