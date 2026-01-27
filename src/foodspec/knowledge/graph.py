"""Spectral Knowledge Graph for semantic representation.

Represents compound-peak relationships and metadata as a knowledge graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CompoundPeakLink:
    """Link between compound and spectral peak."""
    compound_id: str
    peak_wavelength: float
    peak_intensity: Optional[float] = None
    assignment: Optional[str] = None  # e.g., "C-H stretch"
    confidence: float = 1.0
    references: List[str] = field(default_factory=list)


@dataclass
class MetadataOntology:
    """Ontology for spectral metadata."""
    instrument_type: str
    measurement_conditions: Dict[str, float]
    sample_preparation: Optional[str] = None
    date: Optional[str] = None
    operator: Optional[str] = None


class SpectralKnowledgeGraph:
    """Knowledge graph for spectroscopy.
    
    Stores relationships between:
    - Compounds and their characteristic peaks
    - Peaks and their chemical assignments
    - Samples and their metadata
    
    Parameters
    ----------
    name : str, default='SpectralKG'
        Name of the knowledge graph.
    
    Example
    -------
    >>> from foodspec.knowledge import SpectralKnowledgeGraph, CompoundPeakLink
    >>> 
    >>> kg = SpectralKnowledgeGraph('FoodSpecKG')
    >>> kg.add_compound('glucose', peaks=[1030, 1080, 1150])
    >>> kg.add_link(CompoundPeakLink('glucose', 1080, assignment='C-O stretch'))
    >>> 
    >>> # Query
    >>> compounds = kg.query_by_peak(1080, tolerance=5)
    """

    def __init__(self, name: str = 'SpectralKG'):
        self.name = name
        self.compounds: Dict[str, Dict] = {}
        self.peaks: Dict[float, List[CompoundPeakLink]] = {}
        self.metadata: Dict[str, MetadataOntology] = {}
        self.links: List[CompoundPeakLink] = []

    def add_compound(
        self,
        compound_id: str,
        name: Optional[str] = None,
        formula: Optional[str] = None,
        peaks: Optional[List[float]] = None,
    ):
        """Add a compound to the knowledge graph."""
        self.compounds[compound_id] = {
            'name': name or compound_id,
            'formula': formula,
            'peaks': peaks or [],
        }

    def add_link(self, link: CompoundPeakLink):
        """Add a compound-peak link."""
        self.links.append(link)

        if link.peak_wavelength not in self.peaks:
            self.peaks[link.peak_wavelength] = []
        self.peaks[link.peak_wavelength].append(link)

    def query_by_peak(
        self,
        wavelength: float,
        tolerance: float = 1.0,
    ) -> List[CompoundPeakLink]:
        """Query compounds by peak wavelength."""
        results = []
        for peak_wl, links in self.peaks.items():
            if abs(peak_wl - wavelength) <= tolerance:
                results.extend(links)
        return results

    def query_by_compound(self, compound_id: str) -> List[CompoundPeakLink]:
        """Query peaks for a compound."""
        return [link for link in self.links if link.compound_id == compound_id]

    def to_rdf(self) -> str:
        """Export to RDF/Turtle format (simplified)."""
        rdf_lines = [
            "@prefix fs: <http://foodspec.org/kg#> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            ""
        ]

        # Compounds
        for cid, info in self.compounds.items():
            rdf_lines.append(f"fs:{cid} rdf:type fs:Compound ;")
            rdf_lines.append(f'    fs:name "{info["name"]}" .')

        # Links
        for link in self.links:
            rdf_lines.append(f"fs:{link.compound_id} fs:hasPeak [")
            rdf_lines.append(f"    fs:wavelength {link.peak_wavelength} ;")
            if link.assignment:
                rdf_lines.append(f'    fs:assignment "{link.assignment}" ;')
            rdf_lines.append(f"    fs:confidence {link.confidence} ] .")

        return "\n".join(rdf_lines)

    def to_json(self) -> str:
        """Export to JSON."""
        data = {
            'name': self.name,
            'compounds': self.compounds,
            'links': [
                {
                    'compound': link.compound_id,
                    'wavelength': link.peak_wavelength,
                    'intensity': link.peak_intensity,
                    'assignment': link.assignment,
                    'confidence': link.confidence,
                    'references': link.references,
                }
                for link in self.links
            ],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> SpectralKnowledgeGraph:
        """Load from JSON."""
        data = json.loads(json_str)
        kg = cls(name=data['name'])

        for cid, info in data['compounds'].items():
            kg.add_compound(cid, name=info.get('name'), formula=info.get('formula'))

        for link_data in data['links']:
            link = CompoundPeakLink(
                compound_id=link_data['compound'],
                peak_wavelength=link_data['wavelength'],
                peak_intensity=link_data.get('intensity'),
                assignment=link_data.get('assignment'),
                confidence=link_data.get('confidence', 1.0),
                references=link_data.get('references', []),
            )
            kg.add_link(link)

        return kg
