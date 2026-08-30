from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, get_args

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DatasetCaseNotFoundError,
    DatasetError,
    DatasetNotConfiguredError,
)
from app.schemas.incident import CaseMetadata, DatasetSuite, System
from app.schemas.telemetry import ModalityAvailability
from app.services.datasets.rcaeval.case_id_registry import CaseIdRegistry
from app.services.datasets.rcaeval.raw_loader import RCAEvalRawCase

logger = logging.getLogger(__name__)

_KNOWN_SUITES = get_args(DatasetSuite)  # ("RE1", "RE2", "RE3")
_KNOWN_SYSTEMS = get_args(System)  # ("online-boutique", "sock-shop", "train-ticket")

# Real Hugging Face RCAEval flat-layout prefix:
#   re1ob_... -> RE1 / online-boutique
#   re1ss_... -> RE1 / sock-shop
#   re1tt_... -> RE1 / train-ticket
#
# The prefix is deliberately separated from the rest of the directory name
# because everything after the first underscore may encode fault information.
_FLAT_CASE_PREFIX_RE = re.compile(
    r"^re(?P<suite>[123])(?P<system_code>[^_]+)_.+$",
)

_SYSTEM_CODE_MAP: Dict[str, System] = {
    "ob": "online-boutique",
    "ss": "sock-shop",
    "tt": "train-ticket",
}


class _DiscoveredCase:
    """Internal record pairing a case's real directory with safe metadata."""

    __slots__ = ("real_dir_name", "case_dir", "suite", "system")

    def __init__(
        self,
        real_dir_name: str,
        case_dir: Path,
        suite: DatasetSuite,
        system: System,
    ):
        self.real_dir_name = real_dir_name
        self.case_dir = case_dir
        self.suite = suite
        self.system = system


def _parse_system_from_dir_name(real_dir_name: str) -> System:
    """
    Parse the system token used by the existing synthetic/nested fixture
    layout:

        {system}_{service}_{fault}_{instance}

    Only the first token is examined. The remainder may contain
    ground-truth information and is deliberately ignored.
    """
    first_token = real_dir_name.split("_", 1)[0]

    if first_token not in _KNOWN_SYSTEMS:
        raise DatasetError(
            f"Cannot determine system for RCAEval case directory "
            f"{real_dir_name!r}: leading token {first_token!r} is not one "
            f"of {_KNOWN_SYSTEMS}."
        )

    return first_token  # type: ignore[return-value]


def _parse_flat_case_name(
    real_dir_name: str,
) -> Optional[tuple[DatasetSuite, System]]:
    """
    Parse the real Hugging Face RCAEval flat-layout case name.

    Examples:
        re1ob_adservice_cpu_1
        re1ss_carts_cpu_1
        re1tt_ts-auth-service_cpu_1

    Returns:
        (suite, system) for a recognized prefix.

    Returns None when the directory name does not look like a flat
    RCAEval case at all.

    Raises:
        DatasetError when the directory looks like a flat RCAEval case
        but contains an unknown system code. Unknown codes are logged and
        skipped by the caller rather than guessed.
    """
    match = _FLAT_CASE_PREFIX_RE.match(real_dir_name)
    if match is None:
        return None

    suite_value = f"RE{match.group('suite')}"
    system_code = match.group("system_code")

    if suite_value not in _KNOWN_SUITES:
        return None

    system = _SYSTEM_CODE_MAP.get(system_code)
    if system is None:
        raise DatasetError(
            f"Unrecognized RCAEval system code {system_code!r} in "
            f"flat-layout case directory {real_dir_name!r}."
        )

    return suite_value, system  # type: ignore[return-value]


class RCAEvalDiscovery:
    """
    Scans RCAEVAL_DATA_DIR for RCAEval case directories and exposes them
    as safe, opaque-ID-only CaseMetadata.

    Supports both layouts currently used by this project:

    1. Synthetic nested fixture layout:

        {RCAEVAL_DATA_DIR}/
            RE1/
                {system}_{service}_{fault}_{instance}/
                    metrics.json
                    inject_time.txt
            RE2/
                ...

    2. Real Hugging Face RCAEval flat layout:

        {RCAEVAL_DATA_DIR}/
            re1ob_adservice_cpu_1/
                metrics.parquet
                inject_time.txt
            re1ss_carts_cpu_1/
                metrics.parquet
                inject_time.txt
            re1tt_ts-auth-service_cpu_1/
                metrics.parquet
                inject_time.txt

    Discovery never exposes the real directory name through CaseMetadata.
    The real name is retained only internally for CaseIdRegistry/path
    resolution.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._registry: Optional[CaseIdRegistry] = None
        self._cases_by_real_name: Dict[str, _DiscoveredCase] = {}
        self._scanned = False

    def _add_nested_cases(
        self,
        suite: DatasetSuite,
        suite_dir: Path,
        discovered: Dict[str, _DiscoveredCase],
    ) -> None:
        """Scan one synthetic/nested suite directory."""
        for case_dir in sorted(suite_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            real_dir_name = case_dir.name

            try:
                system = _parse_system_from_dir_name(real_dir_name)
            except DatasetError as exc:
                logger.warning("%s Skipping directory.", exc)
                continue

            discovered[real_dir_name] = _DiscoveredCase(
                real_dir_name=real_dir_name,
                case_dir=case_dir,
                suite=suite,
                system=system,
            )

    def _add_flat_case(
        self,
        case_dir: Path,
        discovered: Dict[str, _DiscoveredCase],
    ) -> None:
        """
        Inspect one top-level directory for the real flat RCAEval format.

        A directory is considered a candidate only when it has both:
        - a supported metrics file
        - inject_time.txt

        This avoids treating unrelated directories such as '.cache'
        as RCAEval cases.
        """
        real_dir_name = case_dir.name
        raw_case = RCAEvalRawCase(case_dir)

        if not raw_case.has_metrics() or not raw_case.has_inject_time():
            return

        try:
            parsed = _parse_flat_case_name(real_dir_name)
        except DatasetError as exc:
            logger.warning("%s Skipping directory.", exc)
            return

        if parsed is None:
            return

        suite, system = parsed

        discovered[real_dir_name] = _DiscoveredCase(
            real_dir_name=real_dir_name,
            case_dir=case_dir,
            suite=suite,
            system=system,
        )

    def _scan(self) -> None:
        if self._scanned:
            return

        data_dir = self._settings.rcaeval_data_dir_path
        if not data_dir.is_dir():
            raise DatasetNotConfiguredError()

        discovered: Dict[str, _DiscoveredCase] = {}

        # Support the existing synthetic nested layout.
        for suite in _KNOWN_SUITES:
            suite_dir = data_dir / suite
            if not suite_dir.is_dir():
                continue

            self._add_nested_cases(
                suite=suite,  # type: ignore[arg-type]
                suite_dir=suite_dir,
                discovered=discovered,
            )

        # Support the real Hugging Face flat layout.
        for entry in sorted(data_dir.iterdir(), key=lambda p: p.name):
            if not entry.is_dir():
                continue

            # Suite directories were already handled above.
            if entry.name in _KNOWN_SUITES:
                continue

            self._add_flat_case(
                case_dir=entry,
                discovered=discovered,
            )

        if not discovered:
            raise DatasetNotConfiguredError()

        self._registry = CaseIdRegistry(list(discovered.keys()))
        self._cases_by_real_name = discovered
        self._scanned = True

    def list_cases(
        self,
        suite: Optional[DatasetSuite] = None,
        system: Optional[System] = None,
        max_cases: Optional[int] = None,
    ) -> List[CaseMetadata]:
        """
        Return CaseMetadata for discovered cases, optionally filtered by
        suite/system and capped by max_cases or the configured development
        limit.

        Results are sorted by opaque case_id.
        """
        self._scan()
        assert self._registry is not None

        effective_cap = (
            max_cases
            if max_cases is not None
            else self._settings.RCAEVAL_DEV_MODE_MAX_CASES
        )

        metadata_list: List[CaseMetadata] = []

        for real_name, discovered_case in self._cases_by_real_name.items():
            if suite is not None and discovered_case.suite != suite:
                continue

            if system is not None and discovered_case.system != system:
                continue

            raw_case = RCAEvalRawCase(discovered_case.case_dir)
            case_id = self._registry.opaque_id_for(real_name)

            metadata_list.append(
                CaseMetadata(
                    case_id=case_id,
                    dataset_suite=discovered_case.suite,
                    system=discovered_case.system,
                    available_modalities=ModalityAvailability(
                        metrics=raw_case.has_metrics(),
                        logs=raw_case.has_logs(),
                        traces=raw_case.has_traces(),
                    ),
                )
            )

        metadata_list.sort(key=lambda m: m.case_id)

        if effective_cap is not None:
            metadata_list = metadata_list[:effective_cap]

        return metadata_list

    def get_case_dir(self, case_id: str) -> Path:
        """
        Resolve an opaque case_id to its internal real on-disk directory.

        The returned path must never be exposed in inference-facing
        responses, prompts, or API output because its name may encode
        ground-truth information.
        """
        self._scan()
        assert self._registry is not None

        real_name = self._registry.real_directory_name_for(case_id)
        discovered_case = self._cases_by_real_name.get(real_name)

        if discovered_case is None:
            raise DatasetCaseNotFoundError(case_id)

        return discovered_case.case_dir