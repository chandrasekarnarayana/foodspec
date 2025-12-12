#!/usr/bin/env python3
"""
FoodSpec Protocol Cockpit (PyQt)
--------------------------------

Lightweight desktop UI to browse YAML protocols, validate data, and run
FoodSpec's ProtocolRunner without blocking the main thread. Adds:
- Protocol browser with description/requirements preview.
- Validation dialog (errors/warnings).
- Run history with links to output folders.
- Simple cancel flag for long runs.

Launch:
    python scripts/foodspec_protocol_cockpit.py
Requires:
    pip install PyQt5
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from foodspec.protocol_engine import ProtocolConfig, ProtocolRunner, validate_protocol
from foodspec.logging_utils import setup_logging
from modeling_gui.project_manager import Project, ProjectEntry

PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "examples" / "protocols"
HISTORY_PATH = Path.home() / ".foodspec_gui_history.json"


def load_history() -> List[dict]:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except Exception:
            return []
    return []


def save_history(entries: List[dict]) -> None:
    try:
        HISTORY_PATH.write_text(json.dumps(entries, indent=2))
    except Exception:
        pass


class ProtocolBrowserWidget(QtWidgets.QWidget):
    protocolSelected = QtCore.pyqtSignal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.list = QtWidgets.QListWidget()
        self.desc = QtWidgets.QTextEdit()
        self.desc.setReadOnly(True)
        self.refresh()
        self.list.currentItemChanged.connect(self.on_selection)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Protocols"))
        layout.addWidget(self.list)
        layout.addWidget(QtWidgets.QLabel("Details"))
        layout.addWidget(self.desc)

    def refresh(self):
        self.list.clear()
        for p in sorted(PROTOCOL_DIR.glob("*.yml")):
            self.list.addItem(str(p))

    def on_selection(self, item):
        if not item:
            return
        path = Path(item.text())
        try:
            cfg = ProtocolConfig.from_file(path)
            detail = f"{cfg.name} (v{cfg.version})\n\n{cfg.description}\n\nWhen to use:\n{cfg.when_to_use}\n\nExpected columns:\n{json.dumps(cfg.expected_columns, indent=2)}"
        except Exception as exc:  # pragma: no cover - UI feedback
            detail = f"Failed to load: {exc}"
        self.desc.setPlainText(detail)
        self.protocolSelected.emit(path)


class ProtocolWorker(QtCore.QThread):
    finishedWithResult = QtCore.pyqtSignal(object, object)
    errored = QtCore.pyqtSignal(str)

    def __init__(self, runner: ProtocolRunner, datasets, output_dir: Path, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.datasets = datasets
        self.output_dir = output_dir
        self._cancel = False

    def request_cancel(self):
        self._cancel = True
        try:
            self.runner.request_cancel()
        except Exception:
            pass

    def run(self):
        try:
            res = self.runner.run(self.datasets)
            # Save bundle
            self.runner.save_outputs(res, self.output_dir)
            # Auto-publish bundle (figure panel + narrative)
            try:
                from foodspec.narrative import save_markdown_bundle

                publish_dir = self.output_dir / "publish"
                publish_dir.mkdir(parents=True, exist_ok=True)
                save_markdown_bundle(self.output_dir, publish_dir, fig_limit=6, include_all=False)
            except Exception:
                pass
            self.finishedWithResult.emit(self.runner, res)
        except Exception as exc:  # pragma: no cover - UI feedback
            self.errored.emit(str(exc))


class FoodSpecProtocolCockpit(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FoodSpec Protocol Cockpit")
        self.resize(1200, 720)
        # Logger writes to run.log in run_dir when available
        self.logger = setup_logging()

        self.browser = ProtocolBrowserWidget()
        self.browser.protocolSelected.connect(self.set_protocol_path)
        self.protocol_path: Optional[Path] = None

        self.data_field = QtWidgets.QLineEdit()
        self.btn_browse = QtWidgets.QPushButton("Browse data")
        self.btn_browse.clicked.connect(self.choose_data)

        self.validate_btn = QtWidgets.QPushButton("Validate data & protocol")
        self.run_btn = QtWidgets.QPushButton("Run analysis")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)

        self.validate_btn.clicked.connect(self.validate_protocol_dialog)
        self.run_btn.clicked.connect(self.run_protocol)

        self.cancel_btn.clicked.connect(self.cancel_run)
        self.worker: Optional[ProtocolWorker] = None

        self.plan_box = QtWidgets.QTextEdit()
        self.plan_box.setReadOnly(True)
        self.status = QtWidgets.QLabel("Idle")
        self.history_list = QtWidgets.QListWidget()
        self.load_history_ui()

        # Project controls
        self.project = Project(name="default")
        self.project_list = QtWidgets.QListWidget()
        self.btn_add_entry = QtWidgets.QPushButton("Add dataset to project")
        self.btn_add_entry.clicked.connect(self.add_project_entry)
        self.btn_save_project = QtWidgets.QPushButton("Save project")
        self.btn_save_project.clicked.connect(self.save_project)
        self.btn_load_project = QtWidgets.QPushButton("Load project")
        self.btn_load_project.clicked.connect(self.load_project)
        self.history_list.itemDoubleClicked.connect(self.open_history_item)

        controls = QtWidgets.QFormLayout()
        controls.addRow("Data", self._hbox(self.data_field, self.btn_browse))
        controls.addRow("Validate / Run", self._hbox(self.validate_btn, self.run_btn, self.cancel_btn))

        right = QtWidgets.QVBoxLayout()
        right.addLayout(controls)
        right.addWidget(QtWidgets.QLabel("Run plan / Parameters (safe defaults)"))
        right.addWidget(self.plan_box)
        right.addWidget(QtWidgets.QLabel("Project datasets"))
        right.addWidget(self.project_list)
        right.addWidget(self._hbox(self.btn_add_entry, self.btn_save_project, self.btn_load_project))
        right.addWidget(QtWidgets.QLabel("Run history"))
        right.addWidget(self.history_list)
        right.addWidget(QtWidgets.QLabel("Status"))
        right.addWidget(self.status)
        right.addWidget(self.progress)

        # Prediction hook controls
        self.model_path_field = QtWidgets.QLineEdit()
        self.model_path_field.setPlaceholderText("Frozen model path prefix (.json/.pkl)")
        btn_browse_model = QtWidgets.QPushButton("Browse model")
        btn_browse_model.clicked.connect(self.choose_model)
        btn_apply_model = QtWidgets.QPushButton("Apply model to dataset")
        btn_apply_model.clicked.connect(self.apply_model_to_dataset)
        self.validation_combo = QtWidgets.QComboBox()
        self.validation_combo.addItems(["standard (default 5-fold CV)", "batch_aware", "group_stratified"])
        right.addWidget(QtWidgets.QLabel("Apply frozen model"))
        right.addWidget(self._hbox(self.model_path_field, btn_browse_model, btn_apply_model))
        right.addWidget(QtWidgets.QLabel("Validation strategy"))
        right.addWidget(self.validation_combo)

        splitter = QtWidgets.QSplitter()
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.addWidget(self.browser)
        splitter.addWidget(left_widget)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 600])

        main = QtWidgets.QVBoxLayout(self)
        main.addWidget(splitter)
        # HSI view area
        self.hsi_tabs = QtWidgets.QTabWidget()
        self.hsi_img_label = QtWidgets.QLabel("No HSI loaded")
        self.hsi_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.hsi_roi_table = QtWidgets.QTableWidget()
        self.hsi_roi_table.setColumnCount(3)
        self.hsi_roi_table.setHorizontalHeaderLabels(["ROI", "Pixels", "Notes"])
        self.hsi_tabs.addTab(self.hsi_img_label, "HSI Projection / Labels")
        self.hsi_tabs.addTab(self.hsi_roi_table, "ROI Summary")
        main.addWidget(self.hsi_tabs)
        # Review summary (shows run status and run folder link when available)
        self.review_label = QtWidgets.QLabel("No run selected yet.")
        self.review_link = QtWidgets.QLabel("")
        self.review_link.setOpenExternalLinks(True)
        main.addWidget(self.review_label)
        main.addWidget(self.review_link)

    def _hbox(self, *widgets):
        box = QtWidgets.QHBoxLayout()
        for w in widgets:
            box.addWidget(w)
        container = QtWidgets.QWidget()
        container.setLayout(box)
        return container

    # --------- History ----------
    def load_history_ui(self):
        self.history_list.clear()
        for entry in load_history():
            text = f"{entry.get('ts','?')} | {entry.get('protocol','?')} | {entry.get('status','?')} | {entry.get('run_dir','')}"
            self.history_list.addItem(text)

    def append_history(self, protocol: str, status: str, run_dir: str):
        hist = load_history()
        hist.insert(0, {"ts": QtCore.QDateTime.currentDateTime().toString(QtCore.Qt.ISODate), "protocol": protocol, "status": status, "run_dir": run_dir})
        save_history(hist[:50])
        self.load_history_ui()

    def open_history_item(self, item):
        text = item.text()
        parts = text.split("|")
        run_dir = parts[-1].strip() if parts else ""
        if run_dir and Path(run_dir).exists():
            QtWidgets.QMessageBox.information(self, "Run folder", f"Run folder:\n{run_dir}")

    # --------- Protocol selection ----------
    def set_protocol_path(self, path: Path):
        self.protocol_path = path
        try:
            cfg = ProtocolConfig.from_file(path)
            steps = "\n".join([f"- {s.get('type')}" for s in cfg.steps])
            pp_summary = ""
            for step in cfg.steps:
                if step.get("type") == "preprocess":
                    params = step.get("params", {})
                    pp_summary = f"Baseline={params.get('baseline_method','als')} | Smooth={params.get('smoothing_method','savgol')} | Norm={params.get('normalization','reference')}"
            self.plan_box.setPlainText(
                f"Protocol: {cfg.name} (v{cfg.version})\n\nPreprocessing: {pp_summary or 'n/a'}\n\nSteps:\n{steps}\n\nTip: keep defaults for a defensible starting analysis."
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Protocol load error", f"Could not load protocol.\n\n{exc}\n\nCheck expected columns in quickstart_protocol.md.")
            self.plan_box.setPlainText(f"Failed to load protocol.")

    def choose_data(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select CSV/HDF5", str(Path.cwd()), "Data (*.csv *.h5 *.hdf5)")
        if fname:
            self.data_field.setText(fname)

    # --------- Validation ----------
    def validate_protocol_dialog(self):
        if not self.protocol_path or not self.data_field.text():
            QtWidgets.QMessageBox.warning(self, "Missing input", "Select protocol and data first. See quickstart_protocol.md for expected columns.")
            return
        cfg = ProtocolConfig.from_file(self.protocol_path)
        df = pd.read_csv(self.data_field.text())
        diag = validate_protocol(cfg, df)
        msg = ""
        if diag["errors"]:
            msg += "Errors:\n- " + "\n- ".join(diag["errors"]) + "\n"
        if diag["warnings"]:
            msg += "\nWarnings:\n- " + "\n- ".join(diag["warnings"])
        if not msg:
            msg = "Validation passed."
        icon = QtWidgets.QMessageBox.Warning if diag["errors"] else QtWidgets.QMessageBox.Information
        QtWidgets.QMessageBox(icon, "Validation", msg, parent=self).show()
        if diag["errors"]:
            self.status.setText("Validation errors; fix inputs.")
        else:
            self.status.setText("Validation passed; ready to run.")

    # --------- Run ----------
    def run_protocol(self):
        if not self.protocol_path or (not self.data_field.text() and not self.project.entries):
            QtWidgets.QMessageBox.warning(self, "Missing input", "Select protocol and data or add project datasets. See quickstart_protocol.md for guidance.")
            return
        cfg = ProtocolConfig.from_file(self.protocol_path)
        cfg.validation_strategy = self.validation_combo.currentText()
        # Use project entries if defined, else single file
        inputs = []
        if self.project.entries:
            inputs = [Path(e.path) for e in self.project.entries]
        else:
            inputs = [Path(self.data_field.text())]
        # Pre-validate first input if it's a CSV
        try:
            first_df = pd.read_csv(inputs[0]) if inputs[0].suffix.lower() in {".csv"} else None
            if first_df is not None:
                diag = validate_protocol(cfg, first_df)
                if diag["errors"]:
                    QtWidgets.QMessageBox.critical(self, "Validation errors", "\n".join(diag["errors"]))
                    return
        except Exception:
            pass
        datasets = []
        for p in inputs:
            if p.suffix.lower() in {".h5", ".hdf5"}:
                try:
                    datasets.append(pd.read_hdf(p))
                except Exception:
                    datasets.append(p)
            else:
                datasets.append(pd.read_csv(p))
        runner = ProtocolRunner(cfg)
        # output directory
        out_dir = Path("protocol_runs_gui") / f"{cfg.name}_{inputs[0].stem}"
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        self.worker = ProtocolWorker(runner, datasets, out_dir)
        self.worker.finishedWithResult.connect(self.on_finished)
        self.worker.errored.connect(self.on_error)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status.setText("Running...")
        self.progress.setVisible(True)
        self.worker.start()

    def cancel_run(self):
        if self.worker:
            self.worker.request_cancel()
        self.status.setText("Cancel requested...")

    def on_finished(self, runner, result):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status.setText("Done")
        self.progress.setVisible(False)
        run_dir = getattr(result, "run_dir", None) or str(Path("protocol_runs_gui") / f"{runner.config.name}_{Path(self.data_field.text()).stem}")
        self.append_history(self.protocol_path.name if self.protocol_path else "unknown", "success", str(run_dir))
        # Show summary with report/publish links
        publish_dir = Path(run_dir) / "publish"
        msg = f"Run finished.\nRun dir: {run_dir}"
        if publish_dir.exists():
            msg += f"\nPublished bundle: {publish_dir}"
        self.review_label.setText("Latest run summary:")
        self.review_link.setText(f"<a href='{Path(run_dir).resolve().as_uri()}'>Open run folder</a>")
        QtWidgets.QMessageBox.information(self, "Run complete", msg)
        self.load_hsi_from_run(run_dir)

    def on_error(self, msg: str):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status.setText("Errored")
        self.progress.setVisible(False)
        self.append_history(self.protocol_path.name if self.protocol_path else "unknown", "error", "")
        QtWidgets.QMessageBox.critical(self, "Run error", msg)

    # --------- HSI view helpers ----------
    def load_hsi_from_run(self, run_dir: str):
        if not run_dir:
            return
        run_path = Path(run_dir)
        label_path = run_path / "hsi" / "label_map.npy"
        if label_path.exists():
            try:
                import numpy as np
                labels = np.load(label_path)
                import matplotlib.pyplot as plt
                import io
                fig, ax = plt.subplots()
                im = ax.imshow(labels, cmap="tab20")
                ax.axis("off")
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                plt.close(fig)
                pixmap = QtGui.QPixmap()
                pixmap.loadFromData(buf.getvalue())
                self.hsi_img_label.setPixmap(pixmap)
            except Exception:
                self.hsi_img_label.setText("Failed to load HSI labels.")
        roi_dir = run_path / "hsi"
        rows = []
        if roi_dir.exists():
            for npy in roi_dir.glob("label_*.npy"):
                try:
                    import numpy as np
                    mask = np.load(npy)
                    rows.append((npy.stem, mask.sum(), ""))
                except Exception:
                    continue
        self.hsi_roi_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QtWidgets.QTableWidgetItem(str(val))
                self.hsi_roi_table.setItem(i, j, item)

    # --------- Prediction hooks ----------
    def choose_model(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select frozen model", str(Path.cwd()), "Model (*.json)")
        if fname:
            self.model_path_field.setText(fname.replace(".json", ""))

    def apply_model_to_dataset(self):
        if not self.model_path_field.text():
            QtWidgets.QMessageBox.warning(self, "Missing model", "Select a frozen model first.")
            return
        if not self.project.entries:
            QtWidgets.QMessageBox.warning(self, "Missing dataset", "Add a dataset to apply the model to.")
            return
        try:
            from foodspec.model_lifecycle import FrozenModel
            model = FrozenModel.load(Path(self.model_path_field.text()))
            df = pd.read_csv(self.project.entries[0].path)
            preds = model.predict(df)
            out_path = Path(self.project.entries[0].path).with_suffix(".preds.csv")
            preds.to_csv(out_path, index=False)
            QtWidgets.QMessageBox.information(self, "Prediction complete", f"Predictions saved to {out_path}")
        except Exception as exc:  # pragma: no cover - UI path
            QtWidgets.QMessageBox.critical(self, "Prediction error", str(exc))

    # --------- Project helpers ----------
    def add_project_entry(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Add dataset", str(Path.cwd()), "Data (*.csv *.h5 *.hdf5)")
        if not fname:
            return
        entry = ProjectEntry(path=fname)
        self.project.entries.append(entry)
        self.refresh_project_list()

    def refresh_project_list(self):
        self.project_list.clear()
        for e in self.project.entries:
            self.project_list.addItem(f"{e.path} | instr={e.instrument} | batch={e.batch} | matrix={e.matrix} | role={e.role}")

    def save_project(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", str(Path.home() / "foodspec_project.json"), "JSON (*.json)")
        if not fname:
            return
        self.project.save(Path(fname))

    def load_project(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load project", str(Path.home()), "JSON (*.json)")
        if not fname:
            return
        self.project = Project.load(Path(fname))
        self.refresh_project_list()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = FoodSpecProtocolCockpit()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
