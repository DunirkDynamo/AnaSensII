"""Convert PSTrace .pssession files into JSON payloads.

Usage patterns:
1. Run with no arguments:
   - Opens a file navigator to select one .pssession file.
   - Writes JSON beside the selected file.

2. Run with --input:
   - If --input points to one .pssession file, writes one JSON file.
   - If --input points to a folder, converts all .pssession files in that folder.
"""

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from typing import Any, Optional


class SessionTextReader:
    """Read .pssession text using fallback encodings."""

    def __init__(self, encodings: Optional[tuple[str, ...]] = None) -> None:
        # PSTrace commonly writes utf-16, so try that first.
        self.encodings = encodings or ("utf-16", "utf-8-sig", "utf-8")

    def read_text(self, path: Path) -> str:
        """Return decoded text from a pssession file.

        Raises:
            UnicodeError: If none of the fallback encodings can decode content.
        """
        last_error: Optional[Exception] = None

        for encoding in self.encodings:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeError as exc:
                last_error = exc

        raise UnicodeError(
            f"Could not decode {path} with encodings {self.encodings}: {last_error}"
        )

    def read_first_line_utf16_style(self, path: Path) -> str:
        """Read first-line payload using module.py-compatible behavior."""
        with open(path, "r", encoding="utf-16") as handle:
            lines = handle.readlines()

        if not lines:
            return ""

        return lines[0]


class SessionParser:
    """Parse PSTrace session text into Python objects."""

    def parse(self, raw_text: str) -> Any:
        """Parse session text.

        Strategy:
        1. Try strict JSON first.
        2. If that fails, normalize boolean/null literals and use literal_eval.
        """
        cleaned = self._clean_text(raw_text)

        parsed = self._try_json(cleaned)
        if parsed is not None:
            return parsed

        normalized = self._normalize_for_literal_eval(cleaned)
        return ast.literal_eval(normalized)

    def parse_module_style(self, raw_text: str) -> Any:
        """Parse text using module.py-style normalization and eval semantics."""
        normalized = (
            raw_text.rstrip("\ufeff")
            .replace("true", "True")
            .replace("false", "False")
        )
        parsed = eval(normalized)
        return dict(parsed)

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """Remove outer whitespace and potential UTF BOM."""
        return raw_text.strip().lstrip("\ufeff")

    @staticmethod
    def _try_json(text: str) -> Optional[Any]:
        """Attempt json.loads and return None if parsing fails."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_for_literal_eval(text: str) -> str:
        """Normalize JSON-style literals so literal_eval can parse them."""
        return (
            text.replace("true", "True")
            .replace("false", "False")
            .replace("null", "None")
        )


class MeasurementExtractor:
    """Extract metadata and measurement arrays from parsed session payload."""

    schema_version = "1.0"

    def build_payload(self, parsed: Any, source_path: Path) -> dict[str, Any]:
        """Build JSON-serializable payload for one .pssession file."""
        root = self._ensure_dict(parsed, source_path)
        measurements_raw = self._get_measurements(root)
        measurements = self._extract_all_measurements(measurements_raw)

        return {
            "schema_version": self.schema_version,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "file_metadata": self._build_file_metadata(source_path),
            "measurement_count": len(measurements),
            "measurements": measurements,
        }

    @staticmethod
    def _pick(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
        """Return first matching key from a dictionary with mixed key styles."""
        for key in keys:
            if key in d:
                return d[key]
        return default

    def _ensure_dict(self, parsed: Any, source_path: Path) -> dict[str, Any]:
        """Validate parsed payload type."""
        if not isinstance(parsed, dict):
            raise ValueError(f"Parsed pssession content is not a dict in {source_path}")
        return parsed

    def _get_measurements(self, root: dict[str, Any]) -> list[Any]:
        """Fetch top-level measurements list with case-insensitive support."""
        measurements = self._pick(root, "measurements", "Measurements", default=[])
        return measurements if isinstance(measurements, list) else []

    def _extract_all_measurements(self, measurements_raw: list[Any]) -> list[dict[str, Any]]:
        """Extract each measurement entry into normalized output objects."""
        extracted: list[dict[str, Any]] = []

        for idx, measurement in enumerate(measurements_raw):
            if not isinstance(measurement, dict):
                continue
            extracted.append(self._extract_single_measurement(measurement, idx))

        return extracted

    def _extract_single_measurement(
        self,
        measurement: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        """Extract one measurement's metadata and point arrays."""
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

        # If this file stores multiple curves inside a measurement, preserve them.
        if curves:
            record["curve_count"] = len(curves)
            record["curves"] = curves

        return record

    def _extract_value_arrays(
        self,
        values_section: Any,
    ) -> tuple[list[float], list[float], Optional[str], Optional[str]]:
        """Extract potential/current arrays from DataSet.Values blocks."""
        if not isinstance(values_section, list) or len(values_section) < 3:
            return [], [], None, None

        potential_section = values_section[1]
        current_section = values_section[2]

        potential_datavalues = self._pick(
            potential_section,
            "datavalues",
            "DataValues",
            default=[],
        )
        current_datavalues = self._pick(
            current_section,
            "datavalues",
            "DataValues",
            default=[],
        )

        potential = [self._to_float(self._pick(point, "v", "V", default=0.0)) for point in potential_datavalues]
        current = [self._to_float(self._pick(point, "v", "V", default=0.0)) for point in current_datavalues]

        potential_unit = self._pick(
            self._pick(potential_section, "unit", "Unit", default={}),
            "s",
            "S",
        )
        current_unit = self._pick(
            self._pick(current_section, "unit", "Unit", default={}),
            "s",
            "S",
        )

        return potential, current, potential_unit, current_unit

    def _extract_curves(self, curves_section: Any) -> list[dict[str, Any]]:
        """Extract curve arrays from Curves blocks (multi-curve sessions)."""
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
        """Convert possible numeric value to float safely."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _extract_scan_rate(self, method_text: Any) -> Optional[float]:
        """Extract SCAN_RATE from method text if present."""
        if not isinstance(method_text, str) or not method_text:
            return None

        for line in self._split_method_lines(method_text):
            if line.startswith("SCAN_RATE="):
                raw = line.split("=", 1)[1]
                try:
                    return float(raw)
                except ValueError:
                    return None

        return None

    def _extract_time_from_method(self, method_text: Any) -> Optional[str]:
        """Extract time/date-like second line from method text."""
        if not isinstance(method_text, str) or not method_text:
            return None

        lines = [line.strip() for line in self._split_method_lines(method_text) if line.strip()]
        if len(lines) >= 2:
            return lines[1].lstrip("#")

        return None

    @staticmethod
    def _split_method_lines(method_text: str) -> list[str]:
        """Normalize line endings and split method text."""
        return method_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    @staticmethod
    def _build_file_metadata(source_path: Path) -> dict[str, str]:
        """Build file provenance metadata."""
        stat = source_path.stat()
        return {
            "source_file": str(source_path),
            "source_name": source_path.name,
            "source_stem": source_path.stem,
            "source_suffix": source_path.suffix,
            "file_modified_utc": datetime.fromtimestamp(
                stat.st_mtime,
                tz=timezone.utc,
            ).isoformat(),
        }


class JsonWriter:
    """Write JSON payloads to disk."""

    def write(self, payload: dict[str, Any], output_path: Path) -> None:
        """Serialize payload as pretty JSON and write UTF-8 file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class PSSessionConversionService:
    """Coordinate read -> parse -> extract -> write steps."""

    def __init__(
        self,
        reader: SessionTextReader,
        parser: SessionParser,
        extractor: MeasurementExtractor,
        writer: JsonWriter,
    ) -> None:
        self.reader = reader
        self.parser = parser
        self.extractor = extractor
        self.writer = writer

    def convert_file(self, input_path: Path, output_path: Path) -> None:
        """Convert one .pssession file to one JSON file."""
        # Primary path mirrors module.py parsing behavior.
        raw_text = self.reader.read_first_line_utf16_style(input_path)
        try:
            parsed = self.parser.parse_module_style(raw_text)
        except Exception:
            # Fallback parser supports files that deviate from module.py assumptions.
            raw_text = self.reader.read_text(input_path)
            parsed = self.parser.parse(raw_text)

        payload = self.extractor.build_payload(parsed, input_path)
        self.writer.write(payload, output_path)

    def convert_folder(self, input_dir: Path, output_dir: Path) -> list[Path]:
        """Convert every .pssession file in a folder."""
        outputs: list[Path] = []

        for pssession_path in sorted(input_dir.glob("*.pssession")):
            output_path = output_dir / f"{pssession_path.stem}.json"
            self.convert_file(pssession_path, output_path)
            outputs.append(output_path)

        return outputs


class CliApp:
    """CLI entry class.

    Default behavior:
    - No --input provided: open a file navigator to select one .pssession file.
    - Save output JSON in same location as selected file.
    """

    def __init__(self, service: PSSessionConversionService) -> None:
        self.service = service

    def run(self) -> None:
        """Parse args and execute conversion mode."""
        args = self._parse_args()

        # Requested UX: running the script can pop a file picker for one pssession.
        if not args.input:
            self._run_gui_single_file_flow()
            return

        input_path = Path(args.input)
        self._validate_input_path(input_path)

        if input_path.is_file():
            self._run_file_conversion(input_path, args.output)
            return

        self._run_folder_conversion(input_path, args.output)

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        """Build CLI parser."""
        parser = argparse.ArgumentParser(
            description=(
                "Convert PSTrace .pssession files into JSON with metadata and measurement arrays."
            )
        )
        parser.add_argument(
            "--input",
            required=False,
            help="Path to a .pssession file or folder containing .pssession files.",
        )
        parser.add_argument(
            "--output",
            required=False,
            help=(
                "Output path. For single-file input: JSON path. "
                "For folder input: output directory. "
                "If no --input is provided, this argument is ignored and output is saved next to the selected file."
            ),
        )
        return parser.parse_args()

    @staticmethod
    def _validate_input_path(input_path: Path) -> None:
        """Validate input path existence."""
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

    def _run_gui_single_file_flow(self) -> None:
        """Open file picker, convert selected file, save JSON beside it."""
        selected_file = self._select_pssession_file()

        if selected_file is None:
            print("No file selected. Conversion cancelled.")
            return

        output_path = selected_file.with_suffix(".json")
        self.service.convert_file(selected_file, output_path)
        print(f"Wrote: {output_path}")

    @staticmethod
    def _select_pssession_file() -> Optional[Path]:
        """Open OS file picker and return selected pssession path."""
        root = Tk()
        root.withdraw()

        selected_path = askopenfilename(
            title="Select a PSTrace pssession file",
            filetypes=[("PSTrace Session", "*.pssession"), ("All files", "*.*")],
        )

        root.destroy()

        if not selected_path:
            return None

        return Path(selected_path)

    def _run_file_conversion(self, input_path: Path, output_arg: Optional[str]) -> None:
        """Convert one explicit input file."""
        if input_path.suffix.lower() != ".pssession":
            raise ValueError("Single file input must use the .pssession extension")

        output_path = Path(output_arg) if output_arg else input_path.with_suffix(".json")
        self.service.convert_file(input_path, output_path)
        print(f"Wrote: {output_path}")

    def _run_folder_conversion(self, input_path: Path, output_arg: Optional[str]) -> None:
        """Convert all pssession files in an input folder."""
        output_dir = Path(output_arg) if output_arg else input_path.parent / f"{input_path.name}_json"
        outputs = self.service.convert_folder(input_path, output_dir)
        print(f"Converted {len(outputs)} files to: {output_dir}")


def build_default_app() -> CliApp:
    """Create fully wired application instance."""
    reader = SessionTextReader()
    parser = SessionParser()
    extractor = MeasurementExtractor()
    writer = JsonWriter()
    service = PSSessionConversionService(reader, parser, extractor, writer)
    return CliApp(service)


def main() -> None:
    """Module entrypoint."""
    app = build_default_app()
    app.run()


if __name__ == "__main__":
    main()
