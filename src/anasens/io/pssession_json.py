"""PSTrace pssession parsing and JSON-like extraction utilities."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SessionTextReader:
    """Read .pssession text with fallback encodings."""

    def __init__(self, encodings: tuple[str, ...] | None = None) -> None:
        self.encodings = encodings or ("utf-16", "utf-8-sig", "utf-8")

    def read_text(self, path: Path) -> str:
        """Read full text using configured encoding fallbacks."""
        last_error: Exception | None = None

        for encoding in self.encodings:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeError as exc:
                last_error = exc

        raise UnicodeError(
            f"Could not decode {path} with encodings {self.encodings}: {last_error}"
        )

    def read_first_line_utf16_style(self, path: Path) -> str:
        """Read the first line with utf-16 to match existing PSTrace conversion behavior."""
        with path.open("r", encoding="utf-16") as handle:
            lines = handle.readlines()

        if not lines:
            return ""

        return lines[0]


class SessionParser:
    """Parse raw pssession text into Python objects."""

    def parse(self, raw_text: str) -> Any:
        """Parse text via JSON first, then literal_eval fallback."""
        cleaned = raw_text.strip().lstrip("\ufeff")

        parsed = self._try_json(cleaned)
        if parsed is not None:
            return parsed

        normalized = (
            cleaned.replace("true", "True")
            .replace("false", "False")
            .replace("null", "None")
        )
        return ast.literal_eval(normalized)

    def parse_module_style(self, raw_text: str) -> Any:
        """Parse using legacy module-style eval semantics for compatibility."""
        normalized = raw_text.rstrip("\ufeff").replace("true", "True").replace("false", "False")
        parsed = eval(normalized)
        return dict(parsed)

    @staticmethod
    def _try_json(text: str) -> Any | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None


class MeasurementExtractor:
    """Extract canonical measurement payloads from parsed pssession data."""

    schema_version = "1.0"

    def build_payload(self, parsed: Any, source_path: Path) -> dict[str, Any]:
        root = self._ensure_dict(parsed, source_path)
        measurements_raw = self._pick(root, "measurements", "Measurements", default=[])
        if not isinstance(measurements_raw, list):
            measurements_raw = []

        measurements: list[dict[str, Any]] = []
        for idx, measurement in enumerate(measurements_raw):
            if not isinstance(measurement, dict):
                continue
            measurements.append(self._extract_single_measurement(measurement, idx))

        return {
            "schema_version": self.schema_version,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "file_metadata": self._build_file_metadata(source_path),
            "measurement_count": len(measurements),
            "measurements": measurements,
        }

    @staticmethod
    def _pick(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in d:
                return d[key]
        return default

    @staticmethod
    def _ensure_dict(parsed: Any, source_path: Path) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            raise ValueError(f"Parsed pssession content is not a dictionary in {source_path}")
        return parsed

    def _extract_single_measurement(self, measurement: dict[str, Any], index: int) -> dict[str, Any]:
        title = self._pick(measurement, "title", "Title")
        method_text = self._pick(measurement, "method", "Method")

        dataset = self._pick(measurement, "dataset", "DataSet", default={})
        values = self._pick(dataset, "values", "Values", default=[])
        potential, current, potential_unit, current_unit = self._extract_value_arrays(values)

        curves_raw = self._pick(measurement, "curves", "Curves", default=[])
        curves = self._extract_curves(curves_raw)

        record: dict[str, Any] = {
            "measurement_index": index,
            "title": title,
            "method": method_text,
            "time": self._extract_time_from_method(method_text),
            "scan_rate": self._extract_scan_rate(method_text),
            "units": {
                "potential": potential_unit,
                "current": current_unit,
            },
            "point_count": len(potential),
            "potential": potential,
            "current": current,
        }

        if curves:
            record["curve_count"] = len(curves)
            record["curves"] = curves

        return record

    def _extract_value_arrays(self, values_section: Any) -> tuple[list[float], list[float], str | None, str | None]:
        if not isinstance(values_section, list) or len(values_section) < 3:
            return [], [], None, None

        potential_section = values_section[1]
        current_section = values_section[2]

        potential_datavalues = self._pick(potential_section, "datavalues", "DataValues", default=[])
        current_datavalues = self._pick(current_section, "datavalues", "DataValues", default=[])

        potential = [self._to_float(self._pick(point, "v", "V", default=0.0)) for point in potential_datavalues]
        current = [self._to_float(self._pick(point, "v", "V", default=0.0)) for point in current_datavalues]

        potential_unit = self._pick(self._pick(potential_section, "unit", "Unit", default={}), "s", "S")
        current_unit = self._pick(self._pick(current_section, "unit", "Unit", default={}), "s", "S")

        return potential, current, potential_unit, current_unit

    def _extract_curves(self, curves_section: Any) -> list[dict[str, Any]]:
        curves: list[dict[str, Any]] = []
        if not isinstance(curves_section, list):
            return curves

        for idx, curve in enumerate(curves_section):
            if not isinstance(curve, dict):
                continue

            x_array = self._pick(curve, "XAxisDataArray", "xaxisdataarray", default={})
            y_array = self._pick(curve, "YAxisDataArray", "yaxisdataarray", default={})
            x_datavalues = self._pick(x_array, "DataValues", "datavalues", default=[])
            y_datavalues = self._pick(y_array, "DataValues", "datavalues", default=[])

            potential = [self._to_float(self._pick(point, "V", "v", default=0.0)) for point in x_datavalues]
            current = [self._to_float(self._pick(point, "V", "v", default=0.0)) for point in y_datavalues]
            curves.append(
                {
                    "curve_index": idx,
                    "point_count": len(potential),
                    "potential": potential,
                    "current": current,
                }
            )

        return curves

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _extract_scan_rate(method_text: Any) -> float | None:
        if not isinstance(method_text, str) or not method_text:
            return None

        for line in method_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            if line.startswith("SCAN_RATE="):
                raw = line.split("=", 1)[1]
                try:
                    return float(raw)
                except ValueError:
                    return None
        return None

    @staticmethod
    def _extract_time_from_method(method_text: Any) -> str | None:
        if not isinstance(method_text, str) or not method_text:
            return None

        lines = [line.strip() for line in method_text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
        if len(lines) >= 2:
            return lines[1].lstrip("#")
        return None

    @staticmethod
    def _build_file_metadata(source_path: Path) -> dict[str, str]:
        stat = source_path.stat()
        return {
            "source_file": str(source_path),
            "source_name": source_path.name,
            "source_stem": source_path.stem,
            "source_suffix": source_path.suffix,
            "file_modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }


class PSSessionConversionService:
    """Convert pssession files to canonical extracted payload dictionaries."""

    def __init__(
        self,
        reader: SessionTextReader | None = None,
        parser: SessionParser | None = None,
        extractor: MeasurementExtractor | None = None,
    ) -> None:
        self.reader = reader or SessionTextReader()
        self.parser = parser or SessionParser()
        self.extractor = extractor or MeasurementExtractor()

    def parse_file(self, input_path: Path) -> dict[str, Any]:
        """Parse one pssession file into extracted payload structure."""
        try:
            first_line = self.reader.read_first_line_utf16_style(input_path)
            parsed = self.parser.parse_module_style(first_line)
        except Exception:
            raw_text = self.reader.read_text(input_path)
            parsed = self.parser.parse(raw_text)

        return self.extractor.build_payload(parsed, input_path)
