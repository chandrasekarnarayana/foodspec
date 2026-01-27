"""Journal preset metadata for rapid compliance checks."""
from __future__ import annotations

from typing import Dict, List

JOURNAL_PRESETS: Dict[str, Dict] = {
    "nature": {
        "word_limit": 2000,
        "figure_limit": 6,
        "citation_style": "numbered superscript",
        "data_policy": "Data availability statement mandatory; code availability encouraged.",
        "stats_expectations": "Report exact p-values, effect sizes, and correction for multiplicity.",
        "sections": ["Abstract", "Introduction", "Methods", "Results", "Discussion"],
    },
    "science": {
        "word_limit": 4500,
        "figure_limit": 6,
        "citation_style": "numbered",
        "data_policy": "Include materials and methods in supplementary; data/code availability statements required.",
        "stats_expectations": "State sample sizes, replicates, and statistical tests with assumptions.",
        "sections": ["Abstract", "Introduction", "Results", "Discussion", "Materials and Methods"],
    },
    "ieee-tmi": {
        "word_limit": 9000,
        "figure_limit": 10,
        "citation_style": "IEEE numeric",
        "data_policy": "Data sharing encouraged; describe anonymization and ethics approvals.",
        "stats_expectations": "Report confidence intervals, training/validation splits, and ablation detail.",
        "sections": ["Abstract", "Introduction", "Related Work", "Methods", "Results", "Discussion"],
    },
}


def list_journal_presets() -> List[str]:
    return sorted(JOURNAL_PRESETS.keys())


def get_journal_preset(name: str) -> Dict:
    key = name.lower()
    if key not in JOURNAL_PRESETS:
        raise KeyError(f"Unknown journal preset: {name}")
    return JOURNAL_PRESETS[key].copy()
