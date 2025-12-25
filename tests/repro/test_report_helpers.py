"""Tests for publication assistance helpers."""

from foodspec.report.checklist import ChecklistItem, checklist_score, default_checklist, render_checklist
from foodspec.report.captions import generate_caption, panel_labels
from foodspec.report.journals import get_journal_preset, list_journal_presets
from foodspec.report.methods import MethodsConfig, generate_methods_text, methods_sections
from foodspec.report.stats_notes import statistical_justification


def test_methods_text_contains_components():
    cfg = MethodsConfig(
        dataset="olive oil",
        sample_size=120,
        target="authenticity",
        modality="Raman",
        instruments=["LabRAM HR Evolution"],
        preprocessing=["SNV", "baseline correction"],
        models=["RandomForest"],
        metrics=["accuracy", "roc_auc"],
        calibration="Platt scaling",
        drift_monitoring="PSI and KL divergence",
        random_seed=7,
    )
    text = generate_methods_text(cfg)
    assert "120" in text
    assert "RandomForest" in text
    assert "SNV" in text
    # sections helper returns structured blocks
    sections = methods_sections(cfg)
    assert "Preprocessing" in sections


def test_caption_templates_and_panels():
    cap = generate_caption(
        "roc_curve",
        dataset="oil test",
        modality="Raman",
        highlights=["AUC=0.92"],
        stats="95% CI 0.88-0.96",
        panels=panel_labels(2),
    )
    assert "ROC" in cap or "Receiver" in cap
    assert "AUC" in cap
    assert "95%" in cap


def test_statistical_justification_has_alpha():
    note = statistical_justification(
        test="mann-whitney",
        design="two-group, unpaired",
        alpha=0.01,
        correction="holm-bonferroni",
        effect_size="Cliff's delta",
    )
    assert "0.010" in note
    assert "Mann-Whitney" in note


def test_checklist_render_and_score():
    items = default_checklist()
    items[0].status = True
    md = render_checklist(items[:2])
    assert "[x]" in md
    assert checklist_score(items[:1]) == 1.0


def test_journal_presets():
    preset_names = list_journal_presets()
    assert "nature" in preset_names
    nature = get_journal_preset("Nature")
    assert nature["word_limit"] > 0
    assert "Methods" in " ".join(nature["sections"])
