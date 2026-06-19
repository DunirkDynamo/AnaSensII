"""Pipeline tab for the AnaSens2 Qt GUI."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from PySide6.QtCore import QProcess, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


@dataclass(slots=True)
class PipelineRunRequest:
    """Capture the user-selected pipeline options for future native execution."""

    input_folder: str
    analysis_mode: str
    cut_start_points: int
    cut_end_points: int
    smooth_window_length: int
    smooth_polyorder: int
    baseline_algo: str
    baseline_snip_expected_fwhm: int
    baseline_snip_window_size: int
    baseline_moving_average_window_size: int
    baseline_moving_average_forward: bool
    baseline_moving_average_max_iter: int
    pairplot: bool
    drift_2d: bool
    drift_1d: bool
    overlay_scans: bool


@dataclass(slots=True)
class PreprocessSettings:
    """Editable preprocessing settings for cut, smooth, and baseline stages."""

    cut_start_points: int = 0
    cut_end_points: int = 0
    smooth_window_length: int = 17
    smooth_polyorder: int = 3
    baseline_algo: str = "snip"
    baseline_snip_expected_fwhm: int = 35
    baseline_snip_window_size: int = 10
    baseline_moving_average_window_size: int = 10
    baseline_moving_average_forward: bool = True
    baseline_moving_average_max_iter: int = 1000


class PreprocessSettingsDialog(QDialog):
    """Modal editor for preprocessing stage parameters."""

    def __init__(self, settings: PreprocessSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pipeline Preprocessing Settings")
        self.resize(560, 380)

        layout = QVBoxLayout(self)
        description = QLabel(
            "Customize preprocessing parameters for each stage in the analysis pipeline."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()

        self.cut_start_spin = QSpinBox(self)
        self.cut_start_spin.setRange(0, 500000)
        self.cut_start_spin.setValue(settings.cut_start_points)
        form.addRow("Cut data: points from beginning", self.cut_start_spin)

        self.cut_end_spin = QSpinBox(self)
        self.cut_end_spin.setRange(0, 500000)
        self.cut_end_spin.setValue(settings.cut_end_points)
        form.addRow("Cut data: points from end", self.cut_end_spin)

        self.smooth_window_spin = QSpinBox(self)
        self.smooth_window_spin.setRange(3, 20001)
        self.smooth_window_spin.setSingleStep(2)
        self.smooth_window_spin.setValue(settings.smooth_window_length)
        form.addRow("Savgol smoothing: window length", self.smooth_window_spin)

        self.smooth_polyorder_spin = QSpinBox(self)
        self.smooth_polyorder_spin.setRange(1, 100)
        self.smooth_polyorder_spin.setValue(settings.smooth_polyorder)
        form.addRow("Savgol smoothing: polyorder", self.smooth_polyorder_spin)

        self.baseline_algo_combo = QComboBox(self)
        self.baseline_algo_combo.addItem("pybaselines - snip", "snip")
        self.baseline_algo_combo.addItem("Ekidna moving average", "moving_average")
        current_index = self.baseline_algo_combo.findData(settings.baseline_algo)
        if current_index >= 0:
            self.baseline_algo_combo.setCurrentIndex(current_index)
        form.addRow("Baseline subtraction: method", self.baseline_algo_combo)
        self.baseline_algo_combo.currentIndexChanged.connect(self._sync_baseline_control_state)

        self.baseline_snip_window_spin = QSpinBox(self)
        self.baseline_snip_window_spin.setRange(1, 20001)
        self.baseline_snip_window_spin.setValue(settings.baseline_snip_window_size)
        form.addRow("SNIP: baseline window size", self.baseline_snip_window_spin)

        self.baseline_snip_fwhm_spin = QSpinBox(self)
        self.baseline_snip_fwhm_spin.setRange(1, 100000)
        self.baseline_snip_fwhm_spin.setValue(settings.baseline_snip_expected_fwhm)
        form.addRow("SNIP: expected FWHM", self.baseline_snip_fwhm_spin)

        self.baseline_ma_window_spin = QSpinBox(self)
        self.baseline_ma_window_spin.setRange(1, 20001)
        self.baseline_ma_window_spin.setValue(settings.baseline_moving_average_window_size)
        form.addRow("Moving average: window size", self.baseline_ma_window_spin)

        self.baseline_ma_forward_combo = QComboBox(self)
        self.baseline_ma_forward_combo.addItems(["True", "False"])
        self.baseline_ma_forward_combo.setCurrentText("True" if settings.baseline_moving_average_forward else "False")
        form.addRow("Moving average: forward", self.baseline_ma_forward_combo)

        self.baseline_ma_max_iter_spin = QSpinBox(self)
        self.baseline_ma_max_iter_spin.setRange(1, 1000000)
        self.baseline_ma_max_iter_spin.setValue(settings.baseline_moving_average_max_iter)
        form.addRow("Moving average: max_iter", self.baseline_ma_max_iter_spin)

        self._update_baseline_control_state(str(self.baseline_algo_combo.currentData()))

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings(self) -> PreprocessSettings:
        """Read edited settings from controls with basic consistency constraints."""
        window_length = self.smooth_window_spin.value()
        if window_length % 2 == 0:
            window_length += 1

        polyorder = self.smooth_polyorder_spin.value()
        if polyorder >= window_length:
            polyorder = window_length - 1

        return PreprocessSettings(
            cut_start_points=self.cut_start_spin.value(),
            cut_end_points=self.cut_end_spin.value(),
            smooth_window_length=window_length,
            smooth_polyorder=max(1, polyorder),
            baseline_algo=str(self.baseline_algo_combo.currentData()),
            baseline_snip_expected_fwhm=self.baseline_snip_fwhm_spin.value(),
            baseline_snip_window_size=self.baseline_snip_window_spin.value(),
            baseline_moving_average_window_size=self.baseline_ma_window_spin.value(),
            baseline_moving_average_forward=self.baseline_ma_forward_combo.currentText() == "True",
            baseline_moving_average_max_iter=self.baseline_ma_max_iter_spin.value(),
        )

    def _sync_baseline_control_state(self) -> None:
        self._update_baseline_control_state(str(self.baseline_algo_combo.currentData()))

    def _update_baseline_control_state(self, algo: str) -> None:
        is_snip = algo == "snip"
        self.baseline_snip_window_spin.setEnabled(is_snip)
        self.baseline_snip_fwhm_spin.setEnabled(is_snip)
        self.baseline_ma_window_spin.setEnabled(not is_snip)
        self.baseline_ma_forward_combo.setEnabled(not is_snip)
        self.baseline_ma_max_iter_spin.setEnabled(not is_snip)


class PipelineTab(QWidget):
    """Collect pipeline settings and launch the native analysis pipeline."""

    runRequested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._preprocess_settings = PreprocessSettings()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("Pipeline")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        input_card = self._make_card()
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(16, 16, 16, 16)
        input_layout.setSpacing(12)

        folder_label = QLabel("Input folder")
        folder_label.setObjectName("sectionTitle")
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText(r"Select a folder containing voltammogram CSV files")
        self.folder_button = QPushButton("Browse")
        self.folder_button.clicked.connect(self._choose_folder)

        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(self.folder_button)
        input_layout.addWidget(folder_label)
        input_layout.addLayout(folder_row)

        mode_label = QLabel("Analysis mode")
        mode_label.setObjectName("sectionTitle")
        input_layout.addWidget(mode_label)
        self.single_mode = QRadioButton("Single measurement")
        self.batch_mode = QRadioButton("Batch analysis")
        self.single_mode.setChecked(True)
        input_layout.addWidget(self.single_mode)
        input_layout.addWidget(self.batch_mode)

        preprocess_label = QLabel("Preprocessing parameters")
        preprocess_label.setObjectName("sectionTitle")
        input_layout.addWidget(preprocess_label)

        preprocess_row = QHBoxLayout()
        self.preprocess_button = QPushButton("Configure Preprocessing...")
        self.preprocess_button.clicked.connect(self._open_preprocess_settings)
        preprocess_row.addWidget(self.preprocess_button)
        self.preprocess_load_button = QPushButton("Load From JSON...")
        self.preprocess_load_button.clicked.connect(self._load_preprocess_from_json)
        preprocess_row.addWidget(self.preprocess_load_button)
        preprocess_row.addStretch(1)
        input_layout.addLayout(preprocess_row)

        self.preprocess_summary = QLabel("")
        self.preprocess_summary.setObjectName("subtleBody")
        self.preprocess_summary.setWordWrap(True)
        input_layout.addWidget(self.preprocess_summary)
        self._refresh_preprocess_summary()

        viz_card = self._make_card()
        viz_layout = QVBoxLayout(viz_card)
        viz_layout.setContentsMargins(16, 16, 16, 16)
        viz_layout.setSpacing(10)

        viz_title = QLabel("Visualization options")
        viz_title.setObjectName("sectionTitle")
        viz_layout.addWidget(viz_title)

        self.pairplot_check = QCheckBox("Pairplot of extracted parameters")
        self.drift_2d_check = QCheckBox("2D peak-coordinate drift")
        self.drift_1d_check = QCheckBox("1D peak-coordinate drift")
        self.overlay_check = QCheckBox("Overlay tagged sweeps")
        self.pairplot_check.setChecked(True)
        self.drift_2d_check.setChecked(True)
        self.drift_1d_check.setChecked(True)
        self.overlay_check.setChecked(True)
        viz_layout.addWidget(self.pairplot_check)
        viz_layout.addWidget(self.drift_2d_check)
        viz_layout.addWidget(self.drift_1d_check)
        viz_layout.addWidget(self.overlay_check)

        top_cards_row = QHBoxLayout()
        top_cards_row.setSpacing(12)
        top_cards_row.addWidget(input_card, 1)
        top_cards_row.addWidget(viz_card, 1)

        layout.addLayout(top_cards_row)

        action_row = QHBoxLayout()
        self.run_button = QPushButton("Run Full Pipeline")
        self.run_button.clicked.connect(self._run_pipeline)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._clear_form)
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.clear_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("subtleBody")
        layout.addWidget(self.status_label)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Run output will appear here.")
        layout.addWidget(self.log_output, 1)

    def request_summary(self) -> PipelineRunRequest:
        """Collect the current UI state into a lightweight request object."""
        return PipelineRunRequest(
            input_folder=self.folder_edit.text().strip(),
            analysis_mode="single" if self.single_mode.isChecked() else "batch",
            cut_start_points=self._preprocess_settings.cut_start_points,
            cut_end_points=self._preprocess_settings.cut_end_points,
            smooth_window_length=self._preprocess_settings.smooth_window_length,
            smooth_polyorder=self._preprocess_settings.smooth_polyorder,
            baseline_algo=self._preprocess_settings.baseline_algo,
            baseline_snip_expected_fwhm=self._preprocess_settings.baseline_snip_expected_fwhm,
            baseline_snip_window_size=self._preprocess_settings.baseline_snip_window_size,
            baseline_moving_average_window_size=self._preprocess_settings.baseline_moving_average_window_size,
            baseline_moving_average_forward=self._preprocess_settings.baseline_moving_average_forward,
            baseline_moving_average_max_iter=self._preprocess_settings.baseline_moving_average_max_iter,
            pairplot=self.pairplot_check.isChecked(),
            drift_2d=self.drift_2d_check.isChecked(),
            drift_1d=self.drift_1d_check.isChecked(),
            overlay_scans=self.overlay_check.isChecked(),
        )

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select voltammogram folder")
        if folder:
            self.folder_edit.setText(folder)
            self.status_label.setText(f"Selected folder: {folder}")

    def _clear_form(self) -> None:
        self.folder_edit.clear()
        self.single_mode.setChecked(True)
        self._preprocess_settings = PreprocessSettings()
        self._refresh_preprocess_summary()
        self.pairplot_check.setChecked(True)
        self.drift_2d_check.setChecked(True)
        self.drift_1d_check.setChecked(True)
        self.overlay_check.setChecked(True)
        self.status_label.setText("Ready.")
        self.log_output.clear()

    def _run_pipeline(self) -> None:
        request = self.request_summary()
        self.runRequested.emit(request)

    def _open_preprocess_settings(self) -> None:
        dialog = PreprocessSettingsDialog(self._preprocess_settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._preprocess_settings = dialog.settings()
            self._refresh_preprocess_summary()

    def _load_preprocess_from_json(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Preprocessing Settings JSON",
            "",
            "JSON files (*.json);;All files (*.*)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._preprocess_settings = self._settings_from_json_payload(payload, self._preprocess_settings)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Preprocessing JSON", str(exc))
            return

        self._refresh_preprocess_summary()
        self.status_label.setText(f"Loaded preprocessing settings from: {file_path}")

    def _settings_from_json_payload(
        self,
        payload: Any,
        current: PreprocessSettings,
    ) -> PreprocessSettings:
        if not isinstance(payload, dict):
            raise ValueError("JSON root must be an object.")

        source = payload
        nested = payload.get("preprocess")
        if isinstance(nested, dict):
            source = nested

        def pick(*keys: str):
            for key in keys:
                if key in source:
                    return source[key]
            return None

        cut_start_value = pick(
            "cut_start_points",
            "cut_start",
            "cut_beginning_points",
            "cut_leading_points",
            "cutoff_points",
        )
        cut_end_value = pick("cut_end_points", "cut_end", "cut_trailing_points")
        smooth_window_value = pick(
            "smooth_window_length",
            "savgol_window_length",
            "smoothing_window_length",
            "window_length",
        )
        smooth_polyorder_value = pick(
            "smooth_polyorder",
            "savgol_polyorder",
            "smoothing_polyorder",
            "polyorder",
        )
        baseline_source = source.get("baseline") if isinstance(source.get("baseline"), dict) else None
        baseline_algo = current.baseline_algo

        snip_source = baseline_source.get("snip") if isinstance(baseline_source, dict) and isinstance(baseline_source.get("snip"), dict) else None
        moving_average_source = (
            baseline_source.get("moving_average")
            if isinstance(baseline_source, dict) and isinstance(baseline_source.get("moving_average"), dict)
            else None
        )

        baseline_snip_fwhm_value = None
        baseline_snip_window_value = None
        baseline_ma_window_value = None
        baseline_ma_forward_value = None
        baseline_ma_max_iter_value = None

        if snip_source is not None:
            baseline_snip_fwhm_value = snip_source.get("baseline_expected_fwhm", snip_source.get("expected_fwhm"))
            baseline_snip_window_value = snip_source.get("baseline_window_size", snip_source.get("window_size"))
        else:
            baseline_snip_fwhm_value = pick("baseline_snip_expected_fwhm", "baseline_expected_fwhm", "expected_fwhm", "baseline_fwhm")
            baseline_snip_window_value = pick("baseline_snip_window_size", "baseline_window_size", "baseline_window", "snip_window_size")

        if moving_average_source is not None:
            baseline_ma_window_value = moving_average_source.get("window_size", moving_average_source.get("baseline_window_size"))
            baseline_ma_forward_value = moving_average_source.get("forward")
            baseline_ma_max_iter_value = moving_average_source.get("max_iter")
        else:
            baseline_ma_window_value = pick(
                "baseline_moving_average_window_size",
                "moving_average_window_size",
                "moving_average_window",
            )
            baseline_ma_forward_value = pick("baseline_moving_average_forward", "moving_average_forward")
            baseline_ma_max_iter_value = pick("baseline_moving_average_max_iter", "moving_average_max_iter")

        cut_start = current.cut_start_points if cut_start_value is None else max(0, int(cut_start_value))
        cut_end = current.cut_end_points if cut_end_value is None else max(0, int(cut_end_value))
        smooth_window = (
            current.smooth_window_length if smooth_window_value is None else max(3, int(smooth_window_value))
        )
        if smooth_window % 2 == 0:
            smooth_window += 1

        smooth_polyorder = (
            current.smooth_polyorder if smooth_polyorder_value is None else max(1, int(smooth_polyorder_value))
        )
        if smooth_polyorder >= smooth_window:
            smooth_polyorder = smooth_window - 1

        baseline_snip_fwhm = (
            current.baseline_snip_expected_fwhm
            if baseline_snip_fwhm_value is None
            else max(1, int(baseline_snip_fwhm_value))
        )
        baseline_snip_window = (
            current.baseline_snip_window_size
            if baseline_snip_window_value is None
            else max(1, int(baseline_snip_window_value))
        )
        baseline_ma_window = (
            current.baseline_moving_average_window_size
            if baseline_ma_window_value is None
            else max(1, int(baseline_ma_window_value))
        )
        baseline_ma_forward = (
            current.baseline_moving_average_forward
            if baseline_ma_forward_value is None
            else self._coerce_json_bool(baseline_ma_forward_value)
        )
        baseline_ma_max_iter = (
            current.baseline_moving_average_max_iter
            if baseline_ma_max_iter_value is None
            else max(1, int(baseline_ma_max_iter_value))
        )

        return PreprocessSettings(
            cut_start_points=cut_start,
            cut_end_points=cut_end,
            smooth_window_length=smooth_window,
            smooth_polyorder=smooth_polyorder,
            baseline_algo=baseline_algo,
            baseline_snip_expected_fwhm=baseline_snip_fwhm,
            baseline_snip_window_size=baseline_snip_window,
            baseline_moving_average_window_size=baseline_ma_window,
            baseline_moving_average_forward=baseline_ma_forward,
            baseline_moving_average_max_iter=baseline_ma_max_iter,
        )

    def _coerce_json_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        raise ValueError(f"Could not interpret boolean value: {value!r}")

    def _refresh_preprocess_summary(self) -> None:
        settings = self._preprocess_settings
        if settings.baseline_algo == "moving_average":
            baseline_summary = "Baseline(moving_average window/forward/max_iter): {}/{}/{}".format(
                settings.baseline_moving_average_window_size,
                settings.baseline_moving_average_forward,
                settings.baseline_moving_average_max_iter,
            )
        else:
            baseline_summary = "Baseline(snip window/FWHM): {}/{}".format(
                settings.baseline_snip_window_size,
                settings.baseline_snip_expected_fwhm,
            )

        self.preprocess_summary.setText(
            "Cut(start/end): {}/{} points | Savgol(window/polyorder): {}/{} | {}".format(
                settings.cut_start_points,
                settings.cut_end_points,
                settings.smooth_window_length,
                settings.smooth_polyorder,
                baseline_summary,
            )
        )

    def _make_card(self) -> QFrame:
        card = QFrame(self)
        card.setObjectName("card")
        card.setFrameShape(QFrame.Shape.StyledPanel)
        return card

    def append_log(self, text: str) -> None:
        """Append process output to the visible log."""
        if text:
            self.log_output.appendPlainText(text.rstrip())

    def set_running_state(self, running: bool) -> None:
        """Update button state while a pipeline process is active."""
        self.run_button.setEnabled(not running)
        self.folder_button.setEnabled(not running)
        self.preprocess_button.setEnabled(not running)
        self.preprocess_load_button.setEnabled(not running)
        self.clear_button.setEnabled(not running)
        self.status_label.setText("Running..." if running else "Ready.")
