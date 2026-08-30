# ============================================================
# app/services/reasoning/llm_reasoner.py  (NEW)
# ============================================================
"""
LLM-assisted rationale annotation for RCA candidates.

Per approved design: in this milestone the LLM layer is NOT permitted to
re-rank, add, or remove candidates. It may only attach a `rationale`
string to candidates that already came from DeterministicCandidateGenerator.
The deterministic ranking (order, scores, supporting_evidence, graph_support)
is copied through unchanged.

On any failure -- provider error, unparseable/invalid JSON, or a rationale
key that doesn't match one of the given candidates -- this returns the
original deterministic result unchanged plus an explanatory note. It never
raises out to the caller.

Never receives raw telemetry or anything ground-truth-adjacent; only the
already-bounded RCACandidateResult, EvidenceBundle, and ServiceGraph.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, Set

from pydantic import BaseModel, ValidationError

from app.schemas.evidence import EvidenceBundle
from app.schemas.graph import ServiceGraph
from app.schemas.reasoning import RCACandidateResult
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fences(raw: str) -> str:
    return _CODE_FENCE_RE.sub("", raw).strip()


SYSTEM_PROMPT = """You are an assistant explaining root-cause candidates for a \
distributed-systems incident. You will be given a fixed, already-ranked list of \
candidate services and the evidence supporting each one. Your ONLY job is to write a \
short rationale (1-3 sentences) for EACH candidate, explaining why the evidence and \
topology point to it.

You must NOT change the ranking, add a new candidate, remove a candidate, or reference \
any service not already present in the candidate list you were given.

Respond with ONLY a single JSON object -- no markdown, no code fences, no commentary -- \
matching exactly this shape:

{
  "rationales": {
    "<service-name>": "<1-3 sentence rationale>"
  }
}

Every key in "rationales" MUST exactly match a service name from the candidate list you \
were given. Do not include any other keys.
"""


class _RationalePayload(BaseModel):
    rationales: Dict[str, str]


class LLMRationaleAnnotator:
    def __init__(self, llm_provider: LLMProvider):
        self._llm_provider = llm_provider

    async def annotate(
        self,
        result: RCACandidateResult,
        evidence: EvidenceBundle,
        graph: ServiceGraph,
    ) -> RCACandidateResult:
        if not result.candidates:
            return result

        valid_services = {c.service for c in result.candidates}
        user_prompt = self._build_user_prompt(result, graph)

        try:
            raw_output = await self._llm_provider.complete(SYSTEM_PROMPT, user_prompt)
        except Exception as exc:  # noqa: BLE001 - vendor/provider errors caught broadly, by design
            logger.warning("LLM rationale annotation failed: %s", exc)
            return self._with_note(result, f"LLM rationale generation failed: {exc}")

        try:
            payload = self._parse_output(raw_output, valid_services)
        except ValueError as exc:
            logger.warning("LLM rationale output invalid: %s", exc)
            return self._with_note(result, f"LLM rationale output invalid: {exc}")

        annotated_candidates = [
            candidate.model_copy(update={"rationale": payload.rationales.get(candidate.service)})
            for candidate in result.candidates
        ]

        return result.model_copy(
            update={"method": "llm-assisted", "candidates": annotated_candidates}
        )

    @staticmethod
    def _build_user_prompt(result: RCACandidateResult, graph: ServiceGraph) -> str:
        lines = [f"case_id: {result.case_id}", f"system: {graph.system}", "candidates:"]
        for candidate in result.candidates:
            lines.append(
                f"- service: {candidate.service}, score: {candidate.score:.4f}, "
                f"supporting_evidence: {candidate.supporting_evidence}, "
                f"graph_support: {candidate.graph_support}"
            )
        if graph.notes:
            lines.append(f"graph_notes: {graph.notes}")
        return "\n".join(lines)

    @staticmethod
    def _parse_output(raw_output: str, valid_services: Set[str]) -> _RationalePayload:
        if not raw_output or not raw_output.strip():
            raise ValueError("empty response from LLM provider")

        cleaned = _strip_code_fences(raw_output)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"could not parse JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("response was not a JSON object")

        try:
            payload = _RationalePayload(**data)
        except ValidationError as exc:
            raise ValueError(f"unexpected response shape: {exc}") from exc

        unknown = set(payload.rationales.keys()) - valid_services
        if unknown:
            raise ValueError(f"rationale referenced unknown service(s): {sorted(unknown)}")

        return payload

    @staticmethod
    def _with_note(result: RCACandidateResult, note: str) -> RCACandidateResult:
        return result.model_copy(update={"notes": [*result.notes, note]})