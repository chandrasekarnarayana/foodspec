"""Bridge from spectral features to chemical meaning.

Provides a small curated library of wavenumber-to-chemical meaning entries and
helpers that generate short textual explanations for FeatureSpec objects
(e.g., "Increase near 1745 cm^-1 suggests C=O stretch").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from foodspec.features.specs import FeatureSpec


@dataclass(frozen=True)
class ChemicalMeaning:
    center: float
    window: float
    meaning: str
    modality: str = "any"  # raman, ftir, nir, or any

    def matches(self, wn: float, modality: str, tolerance: float) -> bool:
        if self.modality != "any" and modality not in (self.modality, "any", None):
            return False
        return abs(wn - self.center) <= max(self.window, tolerance)


DEFAULT_CHEMICAL_LIBRARY: List[ChemicalMeaning] = [
    ChemicalMeaning(1745.0, 20.0, "C=O stretch (esters/lipids, oxidation markers)", modality="any"),
    ChemicalMeaning(1650.0, 20.0, "Amide I / C=C stretching (proteins, unsaturation)", modality="any"),
    ChemicalMeaning(1600.0, 15.0, "Aromatic ring C=C stretching (phenyl groups)", modality="raman"),
    ChemicalMeaning(1450.0, 15.0, "CH2 bending (lipid chains, saturation)", modality="any"),
    ChemicalMeaning(1370.0, 15.0, "CH3 bending (lipids/proteins)", modality="any"),
    ChemicalMeaning(1265.0, 15.0, "=C-H in-plane / unsaturation (lipids)", modality="raman"),
    ChemicalMeaning(1170.0, 20.0, "C-O-C stretching (carbohydrates, esters)", modality="any"),
    ChemicalMeaning(1125.0, 15.0, "C-C stretching (lipids/carbohydrates)", modality="raman"),
    ChemicalMeaning(1080.0, 15.0, "C-O stretching (polysaccharides, phosphates)", modality="ftir"),
    ChemicalMeaning(1003.0, 8.0, "Phenylalanine ring breathing (protein marker)", modality="raman"),
    ChemicalMeaning(970.0, 12.0, "C=C stretching / trans double bonds", modality="any"),
    ChemicalMeaning(860.0, 12.0, "C-H bending (out-of-plane, aromatic)", modality="any"),
    ChemicalMeaning(720.0, 10.0, "(CH2)n rocking (long-chain lipids, triglycerides)", modality="ftir"),
]


def find_chemical_meanings(
    wavenumber: float,
    *,
    modality: str = "any",
    tolerance: float = 20.0,
    library: Optional[Sequence[ChemicalMeaning]] = None,
    top_n: int = 3,
) -> List[str]:
    """Return likely chemical meanings for a wavenumber.

    Parameters
    ----------
    wavenumber : float
        Center wavenumber to interpret.
    modality : str, optional
        Modality filter (raman/ftir/nir/any). Defaults to "any".
    tolerance : float, optional
        Additional half-window to accept matches even when outside the
        entry's nominal window.
    library : sequence of ChemicalMeaning, optional
        Override the default meaning library.
    top_n : int, optional
        Number of matches to return sorted by proximity.
    """

    lib = list(library or DEFAULT_CHEMICAL_LIBRARY)
    modality = modality or "any"
    matches: List[tuple[float, ChemicalMeaning]] = []
    for entry in lib:
        if entry.modality != "any" and modality not in (entry.modality, "any"):
            continue
        if entry.matches(wavenumber, modality, tolerance):
            distance = abs(wavenumber - entry.center)
            matches.append((distance, entry))
    matches.sort(key=lambda m: m[0])
    return [m[1].meaning for m in matches[:top_n]]


def explain_feature_spec(
    spec: "FeatureSpec",
    *,
    direction: str = "increase",
    modality: Optional[str] = None,
    tolerance: float = 20.0,
    library: Optional[Sequence[ChemicalMeaning]] = None,
) -> str:
    """Generate short interpretation text for a FeatureSpec.

    Examples
    --------
    >>> explain_feature_spec(FeatureSpec(name="c=o", ftype="peak", regions=[(1740, 1750)]))
    'Increase near 1745 cm^-1 suggests C=O stretch (esters/lipids, oxidation markers).'
    """

    modality = modality or spec.constraints.get("modality") or "any"
    direction_word = direction.capitalize()

    if spec.ftype in {"band", "peak"} and spec.regions:
        parts: List[str] = []
        for lo, hi in spec.regions:
            center = (float(lo) + float(hi)) / 2.0
            meanings = find_chemical_meanings(center, modality=modality, tolerance=tolerance, library=library)
            if meanings:
                first = meanings[0]
                extras = f"; also consider {', '.join(meanings[1:])}" if len(meanings) > 1 else ""
                parts.append(f"{direction_word} near {center:.0f} cm^-1 suggests {first}{extras}.")
            else:
                parts.append(f"{direction_word} near {center:.0f} cm^-1 (no library match).")
        return " ".join(parts)

    if spec.ftype in {"ratio", "index"} and spec.formula:
        return f"{direction_word} in {spec.name} reflects formula {spec.formula}."

    return f"{direction_word} in {spec.name} (no interpretation available)."


def explain_feature_set(
    specs: Iterable["FeatureSpec"],
    *,
    direction: str = "increase",
    modality: Optional[str] = None,
    tolerance: float = 20.0,
    library: Optional[Sequence[ChemicalMeaning]] = None,
) -> Dict[str, str]:
    """Map feature names to interpretation text for a collection of specs."""

    return {
        spec.name: explain_feature_spec(
            spec,
            direction=direction,
            modality=modality or spec.constraints.get("modality") or "any",
            tolerance=tolerance,
            library=library,
        )
        for spec in specs
    }


__all__ = [
    "ChemicalMeaning",
    "DEFAULT_CHEMICAL_LIBRARY",
    "find_chemical_meanings",
    "explain_feature_spec",
    "explain_feature_set",
]
