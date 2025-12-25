"""
HDF5 schema versioning with forward/backward compatibility.

Manages schema versions, migration strategies, and compatibility checking.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple


class SchemaVersion(Enum):
    """HDF5 schema versions with compatibility info."""

    V1_0 = "1.0"  # Initial version: basic spectral data + metadata
    V1_1 = "1.1"  # Added: preprocessing history tracking
    V1_2 = "1_2"  # Added: artifact versioning, run record integration
    V2_0 = "2.0"  # Major: streaming support, nested groups for hyperspectral


# Current stable schema version
CURRENT_SCHEMA_VERSION = SchemaVersion.V1_2.value
MIN_SUPPORTED_SCHEMA_VERSION = SchemaVersion.V1_0.value
MAX_SUPPORTED_SCHEMA_VERSION = SchemaVersion.V2_0.value


class CompatibilityLevel(Enum):
    """Compatibility level between schema versions."""

    COMPATIBLE = "compatible"  # Can read and write
    READABLE = "readable"  # Can read but not write (deprecated)
    INCOMPATIBLE = "incompatible"  # Cannot read
    REQUIRES_MIGRATION = "requires_migration"  # Can read with migration


# Compatibility matrix: (file_version, code_version) -> CompatibilityLevel
COMPATIBILITY_MATRIX: Dict[Tuple[str, str], CompatibilityLevel] = {
    # V1.0 compatibility
    ("1.0", "1.0"): CompatibilityLevel.COMPATIBLE,
    ("1.0", "1.1"): CompatibilityLevel.READABLE,
    ("1.0", "1.2"): CompatibilityLevel.REQUIRES_MIGRATION,
    ("1.0", "2.0"): CompatibilityLevel.REQUIRES_MIGRATION,
    # V1.1 compatibility
    ("1.1", "1.0"): CompatibilityLevel.INCOMPATIBLE,  # Newer file format
    ("1.1", "1.1"): CompatibilityLevel.COMPATIBLE,
    ("1.1", "1.2"): CompatibilityLevel.READABLE,
    ("1.1", "2.0"): CompatibilityLevel.REQUIRES_MIGRATION,
    # V1.2 compatibility
    ("1.2", "1.0"): CompatibilityLevel.INCOMPATIBLE,
    ("1.2", "1.1"): CompatibilityLevel.INCOMPATIBLE,
    ("1.2", "1.2"): CompatibilityLevel.COMPATIBLE,
    ("1.2", "2.0"): CompatibilityLevel.READABLE,
    # V2.0 compatibility
    ("2.0", "1.0"): CompatibilityLevel.INCOMPATIBLE,
    ("2.0", "1.1"): CompatibilityLevel.INCOMPATIBLE,
    ("2.0", "1.2"): CompatibilityLevel.INCOMPATIBLE,
    ("2.0", "2.0"): CompatibilityLevel.COMPATIBLE,
}


def get_compatibility_level(
    file_schema_version: str, code_schema_version: str = CURRENT_SCHEMA_VERSION
) -> CompatibilityLevel:
    """Check compatibility between file and code schema versions.

    Parameters
    ----------
    file_schema_version : str
        Schema version in HDF5 file.
    code_schema_version : str, default=CURRENT_SCHEMA_VERSION
        Current code schema version.

    Returns
    -------
    CompatibilityLevel
        Compatibility status.

    Raises
    ------
    ValueError
        If versions are unknown.
    """
    key = (file_schema_version, code_schema_version)

    if key not in COMPATIBILITY_MATRIX:
        raise ValueError(
            f"Unknown version pair: file={file_schema_version}, "
            f"code={code_schema_version}"
        )

    return COMPATIBILITY_MATRIX[key]


def check_schema_compatibility(
    file_schema_version: str,
    code_schema_version: str = CURRENT_SCHEMA_VERSION,
    allow_incompatible: bool = False,
) -> bool:
    """Check if file schema is compatible with code.

    Parameters
    ----------
    file_schema_version : str
        Schema version from file.
    code_schema_version : str, default=CURRENT_SCHEMA_VERSION
        Code schema version.
    allow_incompatible : bool, default=False
        If True, allow loading incompatible versions (risky).

    Returns
    -------
    bool
        True if compatible or allow_incompatible=True.

    Raises
    ------
    RuntimeError
        If incompatible and allow_incompatible=False.
    """
    compatibility = get_compatibility_level(file_schema_version, code_schema_version)

    if compatibility == CompatibilityLevel.COMPATIBLE:
        return True

    if compatibility == CompatibilityLevel.READABLE:
        # Warn user but allow
        import warnings

        warnings.warn(
            f"File uses deprecated schema v{file_schema_version}. "
            f"Recommend updating to v{code_schema_version}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return True

    if compatibility == CompatibilityLevel.REQUIRES_MIGRATION:
        if allow_incompatible:
            import warnings

            warnings.warn(
                f"File schema v{file_schema_version} requires migration to "
                f"v{code_schema_version}. Loading with allow_incompatible=True "
                f"may cause data loss or errors.",
                UserWarning,
                stacklevel=2,
            )
            return True
        else:
            raise RuntimeError(
                f"File schema v{file_schema_version} is incompatible with "
                f"code schema v{code_schema_version}. Schema migration required. "
                f"Set allow_incompatible=True to proceed at risk of data loss."
            )

    if compatibility == CompatibilityLevel.INCOMPATIBLE:
        if allow_incompatible:
            import warnings

            warnings.warn(
                f"File schema v{file_schema_version} is incompatible with "
                f"code schema v{code_schema_version}. Loading with "
                f"allow_incompatible=True is unsafe.",
                UserWarning,
                stacklevel=2,
            )
            return True
        else:
            raise RuntimeError(
                f"File schema v{file_schema_version} is incompatible with "
                f"code schema v{code_schema_version}. Upgrade FoodSpec or "
                f"regenerate the HDF5 file."
            )

    return False


def migrate_schema_v1_0_to_v1_1(group) -> None:
    """Migrate HDF5 group from schema v1.0 to v1.1 (add preprocessing history).

    Parameters
    ----------
    group : h5py.Group
        HDF5 group to migrate (in-place modification).
    """
    if "preprocessing_history" not in group:
        # Add empty preprocessing history group
        hist_group = group.create_group("preprocessing_history")
        hist_group.attrs["version"] = "1.1"
        hist_group.attrs["description"] = "Preprocessing steps applied to spectra"


def migrate_schema_v1_1_to_v1_2(group) -> None:
    """Migrate HDF5 group from schema v1.1 to v1.2 (add artifact versioning).

    Parameters
    ----------
    group : h5py.Group
        HDF5 group to migrate.
    """
    # Add artifact versioning metadata if not present
    if "artifact_version" not in group.attrs:
        group.attrs["artifact_version"] = "1.0"

    if "foodspec_version" not in group.attrs:
        group.attrs["foodspec_version"] = "1.0.0"


def migrate_schema_v1_2_to_v2_0(group) -> None:
    """Migrate HDF5 group from schema v1.2 to v2.0 (add streaming support).

    Parameters
    ----------
    group : h5py.Group
        HDF5 group to migrate.

    Note
    ----
    This is a major version change. Should backup original file first.
    """
    # Add streaming metadata
    if "streaming_capable" not in group.attrs:
        group.attrs["streaming_capable"] = True

    # Create chunked structure for hyperspectral data if present
    if "hyperspectral_data" in group:
        hsi_group = group["hyperspectral_data"]
        if "chunk_info" not in hsi_group:
            chunk_group = hsi_group.create_group("chunk_info")
            chunk_group.attrs["chunk_height"] = 64
            chunk_group.attrs["chunk_width"] = 64


def migrate_schema(
    group,
    from_version: str,
    to_version: str = CURRENT_SCHEMA_VERSION,
) -> None:
    """Migrate HDF5 group schema from one version to another.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group to migrate.
    from_version : str
        Starting schema version.
    to_version : str, default=CURRENT_SCHEMA_VERSION
        Target schema version.

    Raises
    ------
    ValueError
        If migration path not supported.
    """
    migration_path = _get_migration_path(from_version, to_version)

    for step_from, step_to in migration_path:
        if step_from == "1.0" and step_to == "1.1":
            migrate_schema_v1_0_to_v1_1(group)
        elif step_from == "1.1" and step_to == "1.2":
            migrate_schema_v1_1_to_v1_2(group)
        elif step_from == "1.2" and step_to == "2.0":
            migrate_schema_v1_2_to_v2_0(group)
        else:
            raise ValueError(
                f"No migration path from schema v{step_from} to v{step_to}"
            )

    # Update schema version
    group.attrs["schema_version"] = to_version


def _get_migration_path(from_version: str, to_version: str) -> list:
    """Get sequence of migration steps.

    Parameters
    ----------
    from_version : str
        Starting version.
    to_version : str
        Target version.

    Returns
    -------
    list of tuples
        Migration steps [(from, to), ...].
    """
    versions = ["1.0", "1.1", "1.2", "2.0"]

    try:
        start_idx = versions.index(from_version)
        end_idx = versions.index(to_version)
    except ValueError:
        raise ValueError(
            f"Unknown version in migration: from={from_version}, to={to_version}"
        )

    if start_idx >= end_idx:
        raise ValueError(
            f"Cannot migrate backwards: {from_version} → {to_version}"
        )

    return [(versions[i], versions[i + 1]) for i in range(start_idx, end_idx)]


__all__ = [
    "SchemaVersion",
    "CompatibilityLevel",
    "CURRENT_SCHEMA_VERSION",
    "MIN_SUPPORTED_SCHEMA_VERSION",
    "MAX_SUPPORTED_SCHEMA_VERSION",
    "get_compatibility_level",
    "check_schema_compatibility",
    "migrate_schema",
]
