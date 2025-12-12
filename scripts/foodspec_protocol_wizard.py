#!/usr/bin/env python3
"""
Friendly Protocol Wizard for FoodSpec (lightweight stub for UX).
Guides: Load → Validate → Configure → Estimate → Execute → Review.
Includes onboarding text and safe-default messaging for non-experts.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from PyQt5 import QtCore, QtWidgets


class Breadcrumb(QtWidgets.QWidget):
    def __init__(self, labels: List[str]):
        super().__init__()
        self.labels = labels
        self.current = 0
        self.layout = QtWidgets.QHBoxLayout(self)
        self._render()

    def _render(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, label in enumerate(self.labels):
            dot = QtWidgets.QLabel("●")
            dot.setStyleSheet("color: green;" if i == self.current else "color: gray;")
            text = QtWidgets.QLabel(label)
            font = text.font()
            font.setBold(i == self.current)
            text.setFont(font)
            box = QtWidgets.QHBoxLayout()
            box.addWidget(dot)
            box.addWidget(text)
            w = QtWidgets.QWidget()
            w.setLayout(box)
            self.layout.addWidget(w)
            if i < len(self.labels) - 1:
                self.layout.addWidget(QtWidgets.QLabel("➜"))

    def set_current(self, idx: int):
        self.current = idx
        self._render()


class ProtocolWizard(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FoodSpec Protocol Wizard")
        self.resize(900, 600)
        self.labels = [
            "Load Data & Protocol",
            "Validate",
            "Configure",
            "Estimate",
            "Execute",
            "Review",
        ]
        self.breadcrumb = Breadcrumb(self.labels)

        self.stack = QtWidgets.QStackedWidget()
        self.page_load = self._page_load()
        self.page_validate = self._page_message(
            "Validate", "Run validation to check required columns, class counts, and protocol/library compatibility."
        )
        self.page_config = self._page_message(
            "Configure",
            "Review preprocessing/validation choices. Defaults are safe; advanced overrides live in the cockpit.",
        )
        self.page_estimate = self._page_message(
            "Estimate runtime", "Small CSVs run in seconds; HSI or multi-input runs may take longer."
        )
        self.page_execute = self._page_message(
            "Execute", "Click Run to execute the protocol. Progress and cancel controls are in the cockpit."
        )
        self.page_review = self._page_message(
            "Review outputs", "Open the run folder for report.txt/html, figures/, tables/, metadata.json, index.json."
        )
        for page in [
            self.page_load,
            self.page_validate,
            self.page_config,
            self.page_estimate,
            self.page_execute,
            self.page_review,
        ]:
            self.stack.addWidget(page)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.breadcrumb)
        layout.addWidget(self.stack)
        nav = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("Back")
        self.next_btn = QtWidgets.QPushButton("Next")
        self.prev_btn.clicked.connect(self.prev_step)
        self.next_btn.clicked.connect(self.next_step)
        nav.addStretch()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)

    def _page_load(self):
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        intro = QtWidgets.QLabel(
            "This wizard will preprocess → harmonize → analyze → report.\n"
            "Small datasets typically run in seconds; larger ones may take minutes.\n"
            "See docs/quickstart_protocol.md for expected columns and examples."
        )
        intro.setWordWrap(True)
        self.protocol_combo = QtWidgets.QComboBox()
        proto_dir = Path(__file__).resolve().parents[1] / "examples" / "protocols"
        presets = {
            "Typical edible oil discrimination": "EdibleOil_Classification_v1.yml",
            "Thermal stability tracking": "EdibleOil_Heating_Stability_v1.yml",
            "Oil vs chips matrix comparison": "Chips_vs_Oil_MatrixEffects_v1.yml",
        }
        for label, fname in presets.items():
            p = proto_dir / fname
            if p.exists():
                self.protocol_combo.addItem(f"{label} ({fname})", p)
        self.dataset_list = QtWidgets.QListWidget()
        form.addRow(intro)
        form.addRow("Protocol preset/file (defaults are safe for new users)", self.protocol_combo)
        form.addRow("Datasets", self.dataset_list)
        return w

    def _page_message(self, title: str, text: str) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        lbl = QtWidgets.QLabel(f"<b>{title}</b><br/>{text}")
        lbl.setWordWrap(True)
        v.addWidget(lbl)
        v.addStretch()
        return w

    def next_step(self):
        idx = self.stack.currentIndex()
        if idx < self.stack.count() - 1:
            self.stack.setCurrentIndex(idx + 1)
            self.breadcrumb.set_current(idx + 1)
        self._update_nav()

    def prev_step(self):
        idx = self.stack.currentIndex()
        if idx > 0:
            self.stack.setCurrentIndex(idx - 1)
            self.breadcrumb.set_current(idx - 1)
        self._update_nav()

    def _update_nav(self):
        idx = self.stack.currentIndex()
        self.prev_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < self.stack.count() - 1)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = ProtocolWizard()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover
    main()
