from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.exceptions import CorruptDatasetFileError, DatasetCaseNotFoundError

# RCAEval on-disk filenames.
# Synthetic fixtures currently use metrics.json.
# The real Hugging Face RCAEval layout uses metrics.parquet.
_METRICS_JSON_FILENAME = "metrics.json"
_METRICS_PARQUET_FILENAME = "metrics.parquet"
_LOGS_FILENAME = "logs.csv"
_TRACES_FILENAME = "traces.csv"
_INJECT_TIME_FILENAME = "inject_time.txt"


def _metrics_json_to_dataframe(raw: object, case_id: str) -> pd.DataFrame:
    """
    Convert parsed metrics.json content into a pandas DataFrame with a
    'time' column. Accepts either:
      - a list of record dicts, each containing a 'time' key, or
      - a dict mapping column name -> list of values, where one key is
        'time'.
    """
    if isinstance(raw, list):
        df = pd.DataFrame(raw)
    elif isinstance(raw, dict):
        df = pd.DataFrame(raw)
    else:
        raise CorruptDatasetFileError(
            case_id,
            _METRICS_JSON_FILENAME,
            f"expected JSON list or object, got {type(raw).__name__}",
        )

    if "time" not in df.columns:
        raise CorruptDatasetFileError(
            case_id,
            _METRICS_JSON_FILENAME,
            "no 'time' column after parsing",
        )

    return df


class RCAEvalRawCase:
    """
    Thin, lazy, format-only reader for a single RCAEval case directory.

    Supports the current synthetic JSON layout and the real RCAEval
    Hugging Face Parquet layout. This class knows only the raw file formats
    and does not know about opaque case IDs, ground truth, or RCA logic.

    Each file is read from disk only on first access to its corresponding
    method, and cached afterwards.
    """

    def __init__(self, case_dir: Path):
        self._case_dir = Path(case_dir)
        self._local_id = self._case_dir.name  # real dir name; NOT opaque ID

        if not self._case_dir.is_dir():
            raise DatasetCaseNotFoundError(self._local_id)

        self._metrics: Optional[pd.DataFrame] = None
        self._logs: Optional[pd.DataFrame] = None
        self._traces: Optional[pd.DataFrame] = None
        self._inject_time: Optional[float] = None

    @property
    def case_dir(self) -> Path:
        return self._case_dir

    def has_metrics(self) -> bool:
        """Return True when either supported metrics file exists."""
        return (
            (self._case_dir / _METRICS_PARQUET_FILENAME).is_file()
            or (self._case_dir / _METRICS_JSON_FILENAME).is_file()
        )

    def has_logs(self) -> bool:
        return (self._case_dir / _LOGS_FILENAME).is_file()

    def has_traces(self) -> bool:
        return (self._case_dir / _TRACES_FILENAME).is_file()

    def has_inject_time(self) -> bool:
        """Return True when inject_time.txt exists."""
        return (self._case_dir / _INJECT_TIME_FILENAME).is_file()

    def load_metrics(self) -> pd.DataFrame:
        """
        Load and cache metrics.

        Prefer the real RCAEval metrics.parquet format when present.
        Fall back to metrics.json for the existing synthetic fixtures.
        Both formats require a 'time' column.
        """
        if self._metrics is not None:
            return self._metrics

        parquet_path = self._case_dir / _METRICS_PARQUET_FILENAME
        json_path = self._case_dir / _METRICS_JSON_FILENAME

        # Real RCAEval format: metrics.parquet
        if parquet_path.is_file():
            try:
                df = pd.read_parquet(parquet_path)
            except (OSError, ImportError, ValueError) as exc:
                raise CorruptDatasetFileError(
                    self._local_id,
                    _METRICS_PARQUET_FILENAME,
                    str(exc),
                ) from exc

            if "time" not in df.columns:
                raise CorruptDatasetFileError(
                    self._local_id,
                    _METRICS_PARQUET_FILENAME,
                    "no 'time' column",
                )

            self._metrics = df
            return self._metrics

        # Existing synthetic format: metrics.json
        if json_path.is_file():
            try:
                raw = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CorruptDatasetFileError(
                    self._local_id,
                    _METRICS_JSON_FILENAME,
                    str(exc),
                ) from exc

            self._metrics = _metrics_json_to_dataframe(raw, self._local_id)
            return self._metrics

        raise CorruptDatasetFileError(
            self._local_id,
            _METRICS_PARQUET_FILENAME,
            "neither metrics.parquet nor metrics.json exists",
        )

    def load_logs(self) -> pd.DataFrame:
        """Load and cache logs.csv (RE2/RE3 only) as a DataFrame."""
        if self._logs is not None:
            return self._logs

        path = self._case_dir / _LOGS_FILENAME
        if not path.is_file():
            raise CorruptDatasetFileError(
                self._local_id,
                _LOGS_FILENAME,
                "file is missing",
            )

        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise CorruptDatasetFileError(
                self._local_id,
                _LOGS_FILENAME,
                str(exc),
            ) from exc

        if "time" not in df.columns:
            raise CorruptDatasetFileError(
                self._local_id,
                _LOGS_FILENAME,
                "no 'time' column",
            )

        self._logs = df
        return self._logs

    def load_traces(self) -> pd.DataFrame:
        """Load and cache traces.csv (RE2/RE3 only) as a DataFrame."""
        if self._traces is not None:
            return self._traces

        path = self._case_dir / _TRACES_FILENAME
        if not path.is_file():
            raise CorruptDatasetFileError(
                self._local_id,
                _TRACES_FILENAME,
                "file is missing",
            )

        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise CorruptDatasetFileError(
                self._local_id,
                _TRACES_FILENAME,
                str(exc),
            ) from exc

        if "time" not in df.columns:
            raise CorruptDatasetFileError(
                self._local_id,
                _TRACES_FILENAME,
                "no 'time' column",
            )

        self._traces = df
        return self._traces

    def load_inject_time(self) -> float:
        """Load and cache inject_time.txt as a Unix timestamp (float)."""
        if self._inject_time is not None:
            return self._inject_time

        path = self._case_dir / _INJECT_TIME_FILENAME
        if not path.is_file():
            raise CorruptDatasetFileError(
                self._local_id,
                _INJECT_TIME_FILENAME,
                "file is missing",
            )

        try:
            text = path.read_text(encoding="utf-8").strip()
            value = float(text)
        except (OSError, ValueError) as exc:
            raise CorruptDatasetFileError(
                self._local_id,
                _INJECT_TIME_FILENAME,
                str(exc),
            ) from exc

        self._inject_time = value
        return self._inject_time