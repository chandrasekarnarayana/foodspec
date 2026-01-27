"""
Vendor format support matrix and comprehensive block type testing.

Documents which OPUS and SPC block types are supported, known limitations,
and provides validation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class BlockTypeSupportEntry:
    """Document support for a specific block type."""

    block_type: str
    description: str
    supported: bool
    tested: bool
    known_limitations: Optional[str] = None
    min_version: Optional[str] = None
    max_version: Optional[str] = None


# OPUS Block Type Support Matrix
OPUS_BLOCK_TYPES_SUPPORTED: Dict[str, BlockTypeSupportEntry] = {
    "AB": BlockTypeSupportEntry(
        block_type="AB",
        description="Absorption data (main spectrum)",
        supported=True,
        tested=True,
        min_version="1.0",
    ),
    "BA": BlockTypeSupportEntry(
        block_type="BA",
        description="Background spectrum",
        supported=True,
        tested=True,
        min_version="1.0",
    ),
    "BC": BlockTypeSupportEntry(
        block_type="BC",
        description="Beam combiner setup",
        supported=False,
        tested=False,
        known_limitations="Rarely used; not prioritized",
    ),
    "CH": BlockTypeSupportEntry(
        block_type="CH",
        description="Channel information (detector, laser)",
        supported=True,
        tested=True,
        min_version="1.0",
    ),
    "DX": BlockTypeSupportEntry(
        block_type="DX",
        description="Data blocks (spectral data)",
        supported=True,
        tested=True,
        min_version="1.0",
    ),
    "FX": BlockTypeSupportEntry(
        block_type="FX",
        description="Fourier parameters",
        supported=True,
        tested=True,
        min_version="1.0",
    ),
    "GX": BlockTypeSupportEntry(
        block_type="GX",
        description="Grams/SPC format conversion",
        supported=False,
        tested=False,
        known_limitations="Legacy format; rarely encountered",
    ),
    "HX": BlockTypeSupportEntry(
        block_type="HX",
        description="History/comments",
        supported=True,
        tested=False,
        known_limitations="Parsed but not validated against spectral data",
    ),
    "IN": BlockTypeSupportEntry(
        block_type="IN",
        description="Instrument name and info",
        supported=True,
        tested=True,
        min_version="1.0",
    ),
    "LX": BlockTypeSupportEntry(
        block_type="LX",
        description="Layout/file structure info",
        supported=False,
        tested=False,
        known_limitations="Internal metadata; not used for spectral import",
    ),
    "OB": BlockTypeSupportEntry(
        block_type="OB",
        description="Objects (markers, annotations)",
        supported=False,
        tested=False,
        known_limitations="Not implemented; rarely critical for analysis",
    ),
    "OP": BlockTypeSupportEntry(
        block_type="OP",
        description="Optical path information",
        supported=True,
        tested=False,
        known_limitations="Parsed but not used in spectral processing",
    ),
    "PA": BlockTypeSupportEntry(
        block_type="PA",
        description="Parameters",
        supported=True,
        tested=True,
        min_version="1.0",
    ),
    "PE": BlockTypeSupportEntry(
        block_type="PE",
        description="Peak data",
        supported=False,
        tested=False,
        known_limitations="Proprietary format; use peak_extraction instead",
    ),
    "RX": BlockTypeSupportEntry(
        block_type="RX",
        description="Resampled data",
        supported=True,
        tested=False,
        known_limitations="Supported but resampling params not validated",
    ),
    "SX": BlockTypeSupportEntry(
        block_type="SX",
        description="Sample information",
        supported=True,
        tested=True,
        min_version="1.0",
    ),
    "TM": BlockTypeSupportEntry(
        block_type="TM",
        description="Timestamp/time information",
        supported=True,
        tested=True,
        min_version="1.0",
    ),
}

# SPC Block Type Support Matrix
SPC_BLOCK_TYPES_SUPPORTED: Dict[str, BlockTypeSupportEntry] = {
    "data": BlockTypeSupportEntry(
        block_type="data",
        description="Spectral intensity data",
        supported=True,
        tested=True,
        min_version="1.0",
    ),
    "x_axis": BlockTypeSupportEntry(
        block_type="x_axis",
        description="X-axis (wavenumber/wavelength) data",
        supported=True,
        tested=True,
        min_version="1.0",
    ),
    "log_data": BlockTypeSupportEntry(
        block_type="log_data",
        description="Logarithmic data (transmittance → absorbance)",
        supported=True,
        tested=True,
        min_version="1.0",
    ),
    "interferogram": BlockTypeSupportEntry(
        block_type="interferogram",
        description="Raw interferogram (pre-FFT)",
        supported=False,
        tested=False,
        known_limitations="Raw interferogram processing not implemented",
    ),
    "sample_info": BlockTypeSupportEntry(
        block_type="sample_info",
        description="Sample metadata and comments",
        supported=True,
        tested=False,
        known_limitations="Metadata parsed but validation incomplete",
    ),
    "timestamp": BlockTypeSupportEntry(
        block_type="timestamp",
        description="Acquisition date/time",
        supported=True,
        tested=True,
        min_version="1.0",
    ),
}


def get_opus_support_summary() -> str:
    """Get a human-readable OPUS block type support summary.

    Returns:
        A multi-line string summarizing supported, untested, and unsupported
        OPUS block types with known limitations.
    """
    supported = [k for k, v in OPUS_BLOCK_TYPES_SUPPORTED.items() if v.supported and v.tested]
    supported_untested = [k for k, v in OPUS_BLOCK_TYPES_SUPPORTED.items() if v.supported and not v.tested]
    unsupported = [k for k, v in OPUS_BLOCK_TYPES_SUPPORTED.items() if not v.supported]

    return f"""
OPUS Block Type Support Summary:
- Fully Supported & Tested: {", ".join(supported) or "None"}
- Supported but Untested: {", ".join(supported_untested) or "None"}
- Unsupported: {", ".join(unsupported) or "None"}

Known Limitations:
{
        chr(10).join(
            f"  - {k}: {v.known_limitations}" for k, v in OPUS_BLOCK_TYPES_SUPPORTED.items() if v.known_limitations
        )
        or "  None"
    }
"""


def get_spc_support_summary() -> str:
    """Get a human-readable SPC block type support summary.

    Returns:
        A multi-line string summarizing supported, untested, and unsupported
        SPC block types with known limitations.
    """
    supported = [k for k, v in SPC_BLOCK_TYPES_SUPPORTED.items() if v.supported and v.tested]
    supported_untested = [k for k, v in SPC_BLOCK_TYPES_SUPPORTED.items() if v.supported and not v.tested]
    unsupported = [k for k, v in SPC_BLOCK_TYPES_SUPPORTED.items() if not v.supported]

    return f"""
SPC Block Type Support Summary:
- Fully Supported & Tested: {", ".join(supported) or "None"}
- Supported but Untested: {", ".join(supported_untested) or "None"}
- Unsupported: {", ".join(unsupported) or "None"}

Known Limitations:
{
        chr(10).join(
            f"  - {k}: {v.known_limitations}" for k, v in SPC_BLOCK_TYPES_SUPPORTED.items() if v.known_limitations
        )
        or "  None"
    }
"""


def validate_opus_blocks(present_blocks: Set[str]) -> Dict[str, bool]:
    """Validate which OPUS blocks are safe to import.

    Args:
        present_blocks: Block types detected in a file.

    Returns:
        A mapping `{block_type: is_safe_to_import}`.
    """
    return {
        block: OPUS_BLOCK_TYPES_SUPPORTED[block].supported
        for block in present_blocks
        if block in OPUS_BLOCK_TYPES_SUPPORTED
    }


def validate_spc_blocks(present_blocks: Set[str]) -> Dict[str, bool]:
    """Validate which SPC blocks are safe to import.

    Args:
        present_blocks: Block types detected in a file.

    Returns:
        A mapping `{block_type: is_safe_to_import}`.
    """
    return {
        block: SPC_BLOCK_TYPES_SUPPORTED[block].supported
        for block in present_blocks
        if block in SPC_BLOCK_TYPES_SUPPORTED
    }


def get_untested_blocks_opus(present_blocks: Set[str]) -> List[str]:
    """List present OPUS blocks that are supported but untested.

    Args:
        present_blocks: Block types detected in a file.

    Returns:
        A list of block identifiers.
    """
    return [
        block
        for block in present_blocks
        if block in OPUS_BLOCK_TYPES_SUPPORTED
        and OPUS_BLOCK_TYPES_SUPPORTED[block].supported
        and not OPUS_BLOCK_TYPES_SUPPORTED[block].tested
    ]


def get_untested_blocks_spc(present_blocks: Set[str]) -> List[str]:
    """List present SPC blocks that are supported but untested.

    Args:
        present_blocks: Block types detected in a file.

    Returns:
        A list of block identifiers.
    """
    return [
        block
        for block in present_blocks
        if block in SPC_BLOCK_TYPES_SUPPORTED
        and SPC_BLOCK_TYPES_SUPPORTED[block].supported
        and not SPC_BLOCK_TYPES_SUPPORTED[block].tested
    ]


__all__ = [
    "BlockTypeSupportEntry",
    "OPUS_BLOCK_TYPES_SUPPORTED",
    "SPC_BLOCK_TYPES_SUPPORTED",
    "get_opus_support_summary",
    "get_spc_support_summary",
    "validate_opus_blocks",
    "validate_spc_blocks",
    "get_untested_blocks_opus",
    "get_untested_blocks_spc",
]
