#!/usr/bin/env python3
"""
PyQt GUI for the FoodSpec Raman workflow (RQ1–RQ14)
---------------------------------------------------

A desktop UI wrapper around scripts/raman_workflow_foodspec.py:
- File pickers for oil CSV (required) and chips CSV (optional)
- Configurable run name, output root, oil/heating columns, baseline params
- Non-blocking execution with status log and result folder shortcut

Launch:
    python scripts/raman_workflow_qt.py

Dependencies:
    pip install PyQt5
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from scripts.raman_workflow_foodspec import Config, run_workflow


class Worker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)
    errored = QtCore.pyqtSignal(str)

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg

    def run(self):
        try:
            self.progress.emit("Starting workflow...")
            results_dir = run_workflow(self.cfg)
            self.progress.emit("Workflow complete.")
            self.finished.emit(str(results_dir))
        except Exception as exc:  # pragma: no cover - UI handler
            self.errored.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FoodSpec Raman Workflow (PyQt)")
        self.resize(720, 520)
        self.worker: Worker | None = None

        self.input_field = QtWidgets.QLineEdit()
        self.input_field.setPlaceholderText("Path to wide-format Raman CSV")
        self.browse_input = QtWidgets.QPushButton("Browse")
        self.browse_input.clicked.connect(self.choose_input)

        self.chips_field = QtWidgets.QLineEdit()
        self.chips_field.setPlaceholderText("Optional chips ratiometric CSV (for heating analysis)")
        self.browse_chips = QtWidgets.QPushButton("Browse")
        self.browse_chips.clicked.connect(self.choose_chips)

        self.run_name_field = QtWidgets.QLineEdit("raman_qt_run")
        self.output_root_field = QtWidgets.QLineEdit("results")
        self.oil_col_field = QtWidgets.QLineEdit("Oil_Name")
        self.heat_col_field = QtWidgets.QLineEdit("Heating_Stage")
        self.baseline_lambda_field = QtWidgets.QDoubleSpinBox()
        self.baseline_lambda_field.setRange(1, 1e7)
        self.baseline_lambda_field.setValue(1e5)
        self.baseline_lambda_field.setDecimals(0)
        self.baseline_p_field = QtWidgets.QDoubleSpinBox()
        self.baseline_p_field.setRange(0.000001, 1.0)
        self.baseline_p_field.setDecimals(6)
        self.baseline_p_field.setValue(0.01)
        self.savgol_field = QtWidgets.QSpinBox()
        self.savgol_field.setRange(3, 101)
        self.savgol_field.setSingleStep(2)
        self.savgol_field.setValue(5)

        self.run_button = QtWidgets.QPushButton("Run workflow")
        self.run_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.run_button.clicked.connect(self.start_run)

        self.status_label = QtWidgets.QLabel("Idle")
        self.results_link = QtWidgets.QLabel("")
        self.results_link.setOpenExternalLinks(True)
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(200)

        form = QtWidgets.QFormLayout()
        form.addRow("Oil CSV", self._with_hbox(self.input_field, self.browse_input))
        form.addRow("Chips CSV (optional)", self._with_hbox(self.chips_field, self.browse_chips))
        form.addRow("Run name", self.run_name_field)
        form.addRow("Output root", self.output_root_field)
        form.addRow("Oil column", self.oil_col_field)
        form.addRow("Heating column", self.heat_col_field)
        form.addRow("Baseline lambda", self.baseline_lambda_field)
        form.addRow("Baseline p", self.baseline_p_field)
        form.addRow("Savitzky-Golay window", self.savgol_field)

        top_widget = QtWidgets.QWidget()
        main_vbox = QtWidgets.QVBoxLayout(top_widget)
        main_vbox.addLayout(form)
        main_vbox.addWidget(self.run_button)
        main_vbox.addWidget(self.status_label)
        main_vbox.addWidget(self.results_link)
        main_vbox.addWidget(QtWidgets.QLabel("Log"))
        main_vbox.addWidget(self.log, stretch=1)
        self.setCentralWidget(top_widget)

    @staticmethod
    def _with_hbox(widget_left, widget_right):
        box = QtWidgets.QHBoxLayout()
        box.addWidget(widget_left)
        box.addWidget(widget_right)
        container = QtWidgets.QWidget()
        container.setLayout(box)
        return container

    def choose_input(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Raman CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.input_field.setText(path)

    def choose_chips(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Chips CSV (optional)", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.chips_field.setText(path)

    def append_log(self, msg: str):
        self.log.append(msg)
        self.log.ensureCursorVisible()

    def start_run(self):
        input_path = Path(self.input_field.text().strip())
        if not input_path.exists():
            QtWidgets.QMessageBox.warning(self, "Input missing", "Provide a valid oil CSV path.")
            return

        chips_text = self.chips_field.text().strip()
        chips_path = Path(chips_text) if chips_text else None
        if chips_path and not chips_path.exists():
            QtWidgets.QMessageBox.information(
                self,
                "Chips CSV not found",
                "Chips heating analysis will be skipped because the file was not found.",
            )
            chips_path = None

        cfg = Config(
            input_csv=str(input_path),
            chips_csv=str(chips_path) if chips_path else None,
            run_name=self.run_name_field.text().strip() or "raman_qt_run",
            output_root=Path(self.output_root_field.text().strip() or "results"),
            oil_col=self.oil_col_field.text().strip() or "Oil_Name",
            heating_col=self.heat_col_field.text().strip() or "Heating_Stage",
            baseline_lambda=float(self.baseline_lambda_field.value()),
            baseline_p=float(self.baseline_p_field.value()),
            savgol_window=int(self.savgol_field.value()),
        )

        self.run_button.setEnabled(False)
        self.status_label.setText("Running...")
        self.append_log(f"Launching workflow with input={cfg.input_csv}")

        self.worker = Worker(cfg)
        self.worker.progress.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.errored.connect(self.on_error)
        self.worker.start()

    def on_finished(self, results_dir: str):
        self.status_label.setText("Done")
        safe_path = Path(results_dir).resolve()
        self.results_link.setText(f'<a href="file://{safe_path}">Open results folder</a>')
        self.append_log(f"Results saved to: {safe_path}")
        self.run_button.setEnabled(True)

    def on_error(self, err: str):
        self.status_label.setText("Error")
        self.append_log(f"Error: {err}")
        QtWidgets.QMessageBox.critical(self, "Workflow error", err)
        self.run_button.setEnabled(True)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon.fromTheme("document-open"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
