"""Extract pssession key structure and save it as JSON.

This utility reads PSTrace .pssession files and outputs only the schema-like
key structure (no measurement values), so you can inspect available content.
"""

import argparse
import ast
import json
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from typing import Any, Optional


class PSSessionTextLoader:
    """Load raw .pssession text using common fallback encodings."""

    def __init__(self, encodings: Optional[tuple[str, ...]] = None) -> None:
        self.encodings = encodings or ("utf-16", "utf-8-sig", "utf-8")

    def load(self, path: Path) -> str:
        """Read file text, trying multiple encodings in order."""
        last_error: Optional[Exception] = None
        for encoding in self.encodings:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeError as exc:
                last_error = exc

        raise UnicodeError(
            f"Could not decode {path} with encodings {self.encodings}: {last_error}"
        )

    def load_first_line_utf16_style(self, path: Path) -> str:
        """Load first-line payload using module.py-compatible behavior.

        module.py treats the pssession payload as a single-line record and reads
        the first line only. We mirror that behavior here.
        """
        with open(path, "r", encoding="utf-16") as handle:
            lines = handle.readlines()

        if not lines:
            return ""

        return lines[0]


class PSSessionObjectParser:
    """Parse raw pssession text into a Python object."""

    def parse(self, raw_text: str) -> Any:
        """Parse pssession text with JSON first, then literal_eval fallback."""
        text = raw_text.strip().lstrip("\ufeff")

        # Many pssession exports are close to JSON. Use JSON parser first.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback for JSON-like but non-standard text.
        normalized = (
            text.replace("true", "True")
            .replace("false", "False")
            .replace("null", "None")
        )
        return ast.literal_eval(normalized)

    def parse_module_style(self, raw_text: str) -> Any:
        """Parse with module.py-style normalization and eval semantics.

        module.py behavior:
        1. Strip FEFF from the first-line string.
        2. Replace true/false with Python True/False.
        3. Run eval and cast to dict.
        """
        normalized = (
            raw_text.rstrip("\ufeff")
            .replace("true", "True")
            .replace("false", "False")
        )

        # Keep close behavior to module.py by evaluating the normalized payload.
        parsed = eval(normalized)
        return dict(parsed)


class KeyStructureExtractor:
    """Extract key paths and key tree from nested objects without values."""

    def extract(self, payload: Any) -> dict[str, Any]:
        """Return key-only representation for dictionaries/lists."""
        key_paths: set[str] = set()
        key_tree = self._build_key_tree(payload, prefix="", key_paths=key_paths)

        return {
            "key_paths": sorted(key_paths),
            "key_tree": key_tree,
        }

    def _build_key_tree(self, node: Any, prefix: str, key_paths: set[str]) -> Any:
        """Recursively build key tree while collecting flattened key paths.

        Rules:
        - Dict keys are preserved exactly.
        - Lists use [] marker, and we merge all item key structures.
        - Primitive values are represented by null in output JSON.
        """
        if isinstance(node, dict):
            output: dict[str, Any] = {}
            for key, value in node.items():
                current_path = f"{prefix}.{key}" if prefix else key
                key_paths.add(current_path)
                output[key] = self._build_key_tree(value, current_path, key_paths)
            return output

        if isinstance(node, list):
            list_path = f"{prefix}[]" if prefix else "[]"
            key_paths.add(list_path)

            if not node:
                return []

            # Merge key structure across all list items so we retain union of keys.
            merged = self._merge_list_item_structures(
                [self._build_key_tree(item, list_path, key_paths) for item in node]
            )
            return [merged]

        # Primitive node: no further keys below this point.
        return None

    def _merge_list_item_structures(self, items: list[Any]) -> Any:
        """Merge list-item structures into one representative structure."""
        if not items:
            return None

        # If items are dict-like trees, merge their keys recursively.
        if all(isinstance(item, dict) for item in items):
            merged_dict: dict[str, Any] = {}
            all_keys = set().union(*(item.keys() for item in items))
            for key in all_keys:
                merged_dict[key] = self._merge_list_item_structures(
                    [item[key] for item in items if key in item]
                )
            return merged_dict

        # If items are list-like trees, merge first representative element.
        if all(isinstance(item, list) for item in items):
            sub_items = [sub for item in items for sub in item]
            return [self._merge_list_item_structures(sub_items)] if sub_items else []

        # Mixed or primitive structures: return null as terminal marker.
        return None


class JsonOutputWriter:
    """Write extracted key structures to JSON files."""

    def write(self, output_path: Path, payload: dict[str, Any]) -> None:
        """Serialize payload as pretty UTF-8 JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class PSSessionKeysService:
    """Orchestrate load -> parse -> extract -> write for pssession key dumps."""

    def __init__(
        self,
        loader: PSSessionTextLoader,
        parser: PSSessionObjectParser,
        extractor: KeyStructureExtractor,
        writer: JsonOutputWriter,
    ) -> None:
        self.loader = loader
        self.parser = parser
        self.extractor = extractor
        self.writer = writer

    def process_file(self, input_path: Path, output_path: Path) -> None:
        """Extract key structure from one pssession file and write JSON."""
        # Primary path uses module.py-compatible first-line + eval parsing.
        raw_text = self.loader.load_first_line_utf16_style(input_path)
        try:
            parsed = self.parser.parse_module_style(raw_text)
        except Exception:
            # Fallback to broader parser when files are not module.py-compatible.
            raw_text = self.loader.load(input_path)
            parsed = self.parser.parse(raw_text)

        extracted = self.extractor.extract(parsed)

        payload = {
            "source_file": str(input_path),
            "source_name": input_path.name,
            "key_path_count": len(extracted["key_paths"]),
            "key_paths": extracted["key_paths"],
            "key_tree": extracted["key_tree"],
        }
        self.writer.write(output_path, payload)

    def process_folder(self, input_dir: Path, output_dir: Path) -> list[Path]:
        """Extract key structures for every pssession file in a folder."""
        outputs: list[Path] = []
        for input_path in sorted(input_dir.glob("*.pssession")):
            output_path = output_dir / f"{input_path.stem}_keys.json"
            self.process_file(input_path, output_path)
            outputs.append(output_path)
        return outputs


class CliApplication:
    """Command-line wrapper for key extraction utility."""

    def __init__(self, service: PSSessionKeysService) -> None:
        self.service = service

    def run(self) -> None:
        """Parse arguments and execute extraction."""
        args = self._parse_args()

        # Match the same workflow as pssession_to_json.py:
        # if no input is provided, open a file navigator for one .pssession file.
        if not args.input:
            self._run_gui_single_file_flow()
            return

        input_path = Path(args.input)

        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if input_path.is_file():
            if input_path.suffix.lower() != ".pssession":
                raise ValueError("Single file input must be a .pssession file")

            output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_keys.json")
            self.service.process_file(input_path, output_path)
            print(f"Wrote: {output_path}")
            return

        output_dir = Path(args.output) if args.output else input_path.parent / f"{input_path.name}_keys"
        outputs = self.service.process_folder(input_path, output_dir)
        print(f"Processed {len(outputs)} files. Output folder: {output_dir}")

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        """Build and parse CLI arguments."""
        parser = argparse.ArgumentParser(
            description="Extract pssession key structure and save it to JSON."
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
                "Output path. For file input: output JSON path. "
                "For folder input: output directory."
            ),
        )
        return parser.parse_args()

    def _run_gui_single_file_flow(self) -> None:
        """Open a file picker, then extract key structure for selected file."""
        selected_file = self._select_pssession_file()

        if selected_file is None:
            print("No file selected. Extraction cancelled.")
            return

        output_path = selected_file.with_name(f"{selected_file.stem}_keys.json")
        self.service.process_file(selected_file, output_path)
        print(f"Wrote: {output_path}")

    @staticmethod
    def _select_pssession_file() -> Optional[Path]:
        """Show native file dialog and return selected .pssession path."""
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


def build_app() -> CliApplication:
    """Construct app with default dependency wiring."""
    service = PSSessionKeysService(
        loader=PSSessionTextLoader(),
        parser=PSSessionObjectParser(),
        extractor=KeyStructureExtractor(),
        writer=JsonOutputWriter(),
    )
    return CliApplication(service)


def main() -> None:
    """Entrypoint."""
    app = build_app()
    app.run()


if __name__ == "__main__":
    main()
