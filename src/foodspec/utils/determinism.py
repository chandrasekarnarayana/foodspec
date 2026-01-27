"""
FoodSpec Deterministic Execution

Ensures reproducibility through:
1. Global seed management
2. Environment capture
3. Package version snapshots
"""

import json
import logging
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global seed state
_GLOBAL_SEED: Optional[int] = None


# ============================================================================
# Global Seed Management
# ============================================================================

def set_global_seed(seed: int) -> None:
    """
    Set global seed for all RNG systems.
    
    Affects:
    - Python's random module
    - NumPy
    - scikit-learn
    
    Args:
        seed: Integer seed value
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # scikit-learn (if available)
    if SKLEARN_AVAILABLE:
        pass
        # sklearn uses np.random internally, but being explicit

    logger.info(f"Global seed set: {seed}")


def get_global_seed() -> Optional[int]:
    """Get current global seed"""
    return _GLOBAL_SEED


# ============================================================================
# Environment Capture
# ============================================================================

def capture_environment() -> Dict[str, Any]:
    """
    Capture execution environment details.
    
    Returns:
        Dict with:
        - os_name, os_version
        - python_version
        - machine, processor, cpu_count
        - pwd (current working directory)
    """
    return {
        "os": {
            "name": platform.system(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "python": {
            "python_version": platform.python_version(),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "machine": {
            "node": platform.node(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "machine": platform.machine(),
        },
        "working_directory": str(Path.cwd()),
    }


# ============================================================================
# Package Version Capture
# ============================================================================

CRITICAL_PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "scipy",
    "jinja2",
]

FOODSPEC_PACKAGES = [
    "foodspec",
]


def capture_versions() -> Dict[str, Any]:
    """
    Capture version information for all installed packages.
    
    Returns:
        Dict with:
        - critical_packages: {name: version}
        - foodspec_packages: {name: version}
        - other_packages: {name: version}
    """
    versions = {
        "python": platform.python_version(),
        "critical_packages": {},
        "foodspec_packages": {},
        "other_packages": {},
    }

    # Capture critical packages
    for pkg_name in CRITICAL_PACKAGES:
        try:
            module = __import__(pkg_name)
            version = getattr(module, "__version__", "unknown")
            versions["critical_packages"][pkg_name] = version
        except ImportError:
            versions["critical_packages"][pkg_name] = "not installed"

    # Capture FoodSpec packages
    for pkg_name in FOODSPEC_PACKAGES:
        try:
            module = __import__(pkg_name)
            version = getattr(module, "__version__", "unknown")
            versions["foodspec_packages"][pkg_name] = version
        except ImportError:
            versions["foodspec_packages"][pkg_name] = "not installed"

    return versions


# ============================================================================
# Data Fingerprinting
# ============================================================================

def fingerprint_csv(csv_path: Path) -> str:
    """
    Compute deterministic fingerprint of CSV file.
    
    Uses SHA256 hash of file contents.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Hex digest of SHA256
    """
    import hashlib

    sha256 = hashlib.sha256()

    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def fingerprint_protocol(protocol_dict: Dict[str, Any]) -> str:
    """
    Compute deterministic fingerprint of protocol config.
    
    Uses SHA256 hash of JSON representation.
    
    Args:
        protocol_dict: Protocol configuration dict
        
    Returns:
        Hex digest of SHA256
    """
    import hashlib

    # Sort keys for reproducibility
    json_str = json.dumps(protocol_dict, sort_keys=True, default=str)

    sha256 = hashlib.sha256(json_str.encode())
    return sha256.hexdigest()


# ============================================================================
# Reproducibility Report
# ============================================================================

@dataclass
class ReproducibilityReport:
    """Comprehensive reproducibility metadata"""

    seed: Optional[int]
    environment: Dict[str, Any]
    versions: Dict[str, Any]
    data_fingerprint: str
    protocol_fingerprint: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        return {
            "seed": self.seed,
            "environment": self.environment,
            "versions": self.versions,
            "data_fingerprint": self.data_fingerprint,
            "protocol_fingerprint": self.protocol_fingerprint,
            "timestamp": self.timestamp,
        }

    def to_json(self, out_path: Path) -> Path:
        """Save to JSON file"""
        with open(out_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_path


def generate_reproducibility_report(
    seed: Optional[int] = None,
    csv_path: Optional[Path] = None,
    protocol_dict: Optional[Dict[str, Any]] = None,
) -> ReproducibilityReport:
    """
    Generate comprehensive reproducibility report.
    
    Args:
        seed: Random seed (optional)
        csv_path: Path to data CSV (optional)
        protocol_dict: Protocol config dict (optional)
        
    Returns:
        ReproducibilityReport object
    """
    from datetime import datetime

    env = capture_environment()
    vers = capture_versions()

    data_fp = fingerprint_csv(csv_path) if csv_path and csv_path.exists() else "none"
    proto_fp = fingerprint_protocol(protocol_dict) if protocol_dict else "none"

    timestamp = datetime.utcnow().isoformat()

    return ReproducibilityReport(
        seed=seed or get_global_seed(),
        environment=env,
        versions=vers,
        data_fingerprint=data_fp,
        protocol_fingerprint=proto_fp,
        timestamp=timestamp,
    )
