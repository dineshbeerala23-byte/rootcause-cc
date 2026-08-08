from __future__ import annotations

from typing import Dict, List

from app.core.exceptions import DatasetCaseNotFoundError

_OPAQUE_ID_PREFIX = "case-"
_OPAQUE_ID_WIDTH = 4


class CaseIdRegistry:
    """
    Maps real RCAEval directory names (which encode the fault answer,
    e.g. "sock-shop_cartservice_cpu_1") to opaque, safe-to-expose IDs
    (e.g. "case-0001"), and back.

    This is the only place in the codebase, outside the raw loader
    itself, that is allowed to hold real directory names. The reverse
    lookup (`real_directory_name_for`) is internal-use only.

    ID assignment is deterministic: real directory names are sorted
    and numbered in that order.
    """

    def __init__(self, real_directory_names: List[str]):
        unique_names = sorted(set(real_directory_names))

        self._opaque_to_real: Dict[str, str] = {}
        self._real_to_opaque: Dict[str, str] = {}

        for index, real_name in enumerate(unique_names, start=1):
            opaque_id = f"{_OPAQUE_ID_PREFIX}{index:0{_OPAQUE_ID_WIDTH}d}"
            self._opaque_to_real[opaque_id] = real_name
            self._real_to_opaque[real_name] = opaque_id

    def all_case_ids(self) -> List[str]:
        """Return all opaque case IDs known to this registry, sorted."""
        return sorted(self._opaque_to_real.keys())

    def opaque_id_for(self, real_directory_name: str) -> str:
        """
        Look up the opaque case ID for a real directory name.

        Note: no opaque ID exists yet if this lookup fails, so the real
        directory name is passed to DatasetCaseNotFoundError as the
        identifier — it's the only one available at this call site.
        """
        try:
            return self._real_to_opaque[real_directory_name]
        except KeyError as exc:
            raise DatasetCaseNotFoundError(real_directory_name) from exc

    def real_directory_name_for(self, case_id: str) -> str:
        """
        Internal-use-only reverse lookup: opaque case ID -> real
        directory name. Callers must not surface this value in any
        inference-facing schema, API response, or LLM prompt.
        """
        try:
            return self._opaque_to_real[case_id]
        except KeyError as exc:
            raise DatasetCaseNotFoundError(case_id) from exc