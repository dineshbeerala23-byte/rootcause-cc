from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, get_args

from app.core.config import Settings, get_settings
from app.core.exceptions import DatasetCaseNotFoundError, DatasetError, DatasetNotConfiguredError
from app.schemas.incident import CaseMetadata, DatasetSuite, System
from app.schemas.telemetry import ModalityAvailability
from app.services.datasets.rcaeval.case_id_registry import CaseIdRegistry
from app.services.datasets.rcaeval.raw_loader import RCAEvalRawCase

_KNOWN_SUITES = get_args(DatasetSuite)  # ("RE1", "RE2", "RE3")
_KNOWN_SYSTEMS = get_args(System)  # ("online-boutique", "sock-shop", "train-ticket")


class _DiscoveredCase:
    """Internal record pairing a case's real directory with its parsed, safe metadata."""

    __slots__ = ("real_dir_name", "case_dir", "suite", "system")

    def __init__(self, real_dir_name: str, case_dir: Path, suite: DatasetSuite, system: System):
        self.real_dir_name = real_dir_name
        self.case_dir = case_dir
        self.suite = suite
        self.system = system


def _parse_system_from_dir_name(real_dir_name: str) -> System:
    """
    Extract the system token from a real RCAEval case directory name of
    the form "{system}_{service}_{fault}_{instance}". Only the leading
    system token is safe to expose (which of the three demo apps this
    case belongs to) — everything after the first underscore encodes
    the faulted service/fault type (ground truth) and must never be
    parsed out here or exposed via CaseMetadata. Discovery only ever
    looks at this one leading token; it never inspects the remainder.
    """
    first_token = real_dir_name.split("_", 1)[0]
    if first_token not in _KNOWN_SYSTEMS:
        raise DatasetError(
            f"Cannot determine system for RCAEval case directory "
            f"{real_dir_name!r}: leading token {first_token!r} is not one "
            f"of {_KNOWN_SYSTEMS}."
        )
    return first_token  # type: ignore[return-value]


class RCAEvalDiscovery:
    """
    Scans RCAEVAL_DATA_DIR for RCAEval case directories and exposes them
    as safe, opaque-ID-only CaseMetadata — never the real directory name.

    ASSUMED on-disk layout (RCAEval has not been downloaded in this
    project yet, so this mirrors the manifest's stated directory-naming
    convention rather than a layout confirmed against real downloaded
    data — flagging this rather than guessing silently):

        {RCAEVAL_DATA_DIR}/
            RE1/
                {system}_{service}_{fault}_{instance}/
                    metrics.json
                    inject_time.txt
            RE2/
                {system}_{service}_{fault}_{instance}/
                    metrics.json
                    logs.csv
                    traces.csv
                    inject_time.txt
            RE3/
                ...

    Only RE1/RE2/RE3 top-level folders are scanned; anything else under
    RCAEVAL_DATA_DIR is ignored. If the real download layout differs
    (extra nesting, differently named suite folders), this scanning
    logic — and only this file — will need updating once real data is
    available.

    Case discovery / registry construction is lazy: nothing is scanned
    until the first call to list_cases() or get_case_dir(), and the
    result is cached for the lifetime of this instance. Listing itself
    only calls RCAEvalRawCase.has_metrics()/has_logs()/has_traces()
    (cheap existence checks) — it never loads metrics/logs/traces/
    inject_time contents, preserving the lazy-loading principle from
    Part 3.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._registry: Optional[CaseIdRegistry] = None
        self._cases_by_real_name: Dict[str, _DiscoveredCase] = {}
        self._scanned = False

    def _scan(self) -> None:
        if self._scanned:
            return

        data_dir = self._settings.rcaeval_data_dir_path
        if not data_dir.is_dir():
            raise DatasetNotConfiguredError()

        discovered: Dict[str, _DiscoveredCase] = {}

        for suite in _KNOWN_SUITES:
            suite_dir = data_dir / suite
            if not suite_dir.is_dir():
                continue

            for case_dir in sorted(suite_dir.iterdir()):
                if not case_dir.is_dir():
                    continue

                real_dir_name = case_dir.name
                system = _parse_system_from_dir_name(real_dir_name)
                discovered[real_dir_name] = _DiscoveredCase(
                    real_dir_name=real_dir_name,
                    case_dir=case_dir,
                    suite=suite,  # type: ignore[arg-type]
                    system=system,
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
        suite/system, capped at max_cases if given, else at
        RCAEVAL_DEV_MODE_MAX_CASES from settings (if that's set and
        max_cases was not explicitly passed). No cap is applied if
        neither is set.

        Result is always sorted by opaque case_id, so a cap from either
        source is deterministic across calls.
        """
        self._scan()
        assert self._registry is not None

        effective_cap = (
            max_cases if max_cases is not None else self._settings.RCAEVAL_DEV_MODE_MAX_CASES
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
        Internal/adapter-facing only (Part 5): resolve an opaque case_id
        to its real on-disk directory, so the adapter can open it via
        RCAEvalRawCase. The returned Path's name is the real RCAEval
        directory name and may encode ground-truth information (faulted
        service / fault type) — callers must never surface this Path, or
        any part of it, in an inference-facing response, prompt, or
        NormalizedIncident. Carries the same restriction as
        CaseIdRegistry.real_directory_name_for, which it delegates to.
        """
        self._scan()
        assert self._registry is not None

        real_name = self._registry.real_directory_name_for(case_id)
        discovered_case = self._cases_by_real_name.get(real_name)
        if discovered_case is None:
            # Unreachable if the registry and scan cache stay in sync;
            # guarded anyway rather than returning a bad path silently.
            raise DatasetCaseNotFoundError(case_id)
        return discovered_case.case_dir