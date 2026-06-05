from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .patterns import CompiledPattern, match_path, parse_timestamp

logger = logging.getLogger(__name__)


class FileMatcher:
    """Scan files below a root directory and match them with a compiled pattern."""

    def __init__(self, directory: str | Path, regex_pattern: CompiledPattern):
        if not isinstance(regex_pattern, CompiledPattern):
            raise TypeError("regex_pattern must be created by check_pattern()")
        self.directory = directory
        self.regex_pattern = regex_pattern

    def get_files(self) -> List[str]:
        file_list: list[str] = []
        root = Path(self.directory).expanduser()
        logger.info("Searching for files in %s", root)
        for path in root.rglob("*"):
            if path.is_file():
                file_list.append(str(path.resolve()))
        logger.info("Finish. %d files found in %s", len(file_list), root)
        return file_list

    def match_files(
        self,
        file_paths: Optional[List[str]] = None,
        num_threads: int = 1,
    ) -> List[Dict]:
        if file_paths is None:
            file_paths = self.get_files()
        logger.info("Start file pattern matching...")

        if num_threads <= 1:
            results = [self._match_file(path) for path in file_paths]
        else:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(executor.map(self._match_file, file_paths))

        matched = [result for result in results if result]
        logger.info("%d files matched.", len(matched))
        return matched

    def _match_file(self, file_path: str) -> Optional[Dict]:
        try:
            fields = match_path(self.regex_pattern, file_path)
            if fields is None:
                return None
            fields["time"] = parse_timestamp(fields)
            if fields["time"] is None:
                return None
            fields["path"] = Path(file_path).expanduser().resolve().as_posix()
            return fields
        except Exception as exc:
            logger.warning("Failed to match file %s: %s", file_path, exc)
            return None


class FileFilter:
    """Filter matched file dictionaries with simple list/range criteria."""

    def __init__(self, criteria: Optional[Dict[str, dict]] = None, num_threads: int = 1):
        self.raw_criteria = criteria or {}
        self.list_criteria: Dict[str, List[Any]] = {}
        self.range_criteria: Dict[str, List[tuple[Any, Any]]] = {}
        self.type_map: Dict[str, Optional[str]] = {}
        self.num_threads = max(1, num_threads)
        self._parse_criteria()

    def _parse_criteria(self) -> None:
        for field_name, cfg in self.raw_criteria.items():
            if not isinstance(cfg, dict) or "type" not in cfg or "value" not in cfg:
                raise ValueError(f"Filter '{field_name}' must contain 'type' and 'value'.")
            filter_type = cfg["type"]
            values = cfg["value"]
            self.type_map[field_name] = cfg.get("data_type")
            if filter_type == "list":
                self._parse_list_criteria(field_name, values)
            elif filter_type == "range":
                self._parse_range_criteria(field_name, values)
            else:
                raise ValueError(f"Filter '{field_name}' has unsupported type '{filter_type}'.")

    def _parse_list_criteria(self, field_name: str, values: Any) -> None:
        if not isinstance(values, (list, tuple, set)):
            raise ValueError(f"Filter '{field_name}' list value must be a sequence.")
        self.list_criteria[field_name] = list(values)

    def _parse_range_criteria(self, field_name: str, values: Any) -> None:
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Filter '{field_name}' range value must be a sequence.")
        if len(values) % 2 != 0:
            raise ValueError(f"Filter '{field_name}' range value must contain pairs.")
        self.range_criteria[field_name] = [
            (values[index], values[index + 1])
            for index in range(0, len(values), 2)
        ]

    def filter_files(self, file_list: List[Dict]) -> List[Dict]:
        if not file_list:
            return []
        if self.num_threads <= 1:
            valid_flags = [self._is_valid_file(item) for item in file_list]
        else:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                valid_flags = list(executor.map(self._is_valid_file, file_list))
        filtered = [
            file_info for file_info, is_valid in zip(file_list, valid_flags) if is_valid
        ]
        logger.info("filtering finished, %d files passed.", len(filtered))
        return filtered

    def _is_valid_file(self, file_info: Dict) -> bool:
        return self._passes_list_criteria(file_info) and self._passes_range_criteria(file_info)

    def _passes_list_criteria(self, file_info: Dict) -> bool:
        for field_name, valid_values in self.list_criteria.items():
            if field_name not in file_info:
                return False
            if not self._has_declared_type(field_name, file_info[field_name]):
                return False
            if file_info[field_name] not in valid_values:
                return False
        return True

    def _passes_range_criteria(self, file_info: Dict) -> bool:
        for field_name, pairs in self.range_criteria.items():
            if field_name not in file_info:
                return False
            if not self._has_declared_type(field_name, file_info[field_name]):
                return False
            if not any(start <= file_info[field_name] <= end for start, end in pairs):
                return False
        return True

    def _has_declared_type(self, field_name: str, value: Any) -> bool:
        declared_type = self.type_map.get(field_name)
        if declared_type is None:
            return True
        if declared_type == "datetime":
            return isinstance(value, datetime)
        if declared_type in ("float", "int", "numeric"):
            try:
                float(value)
                return True
            except (TypeError, ValueError):
                return False
        if declared_type == "str":
            return isinstance(value, str)
        return True
