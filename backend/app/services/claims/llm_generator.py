# ============================================================
# app/services/claims/llm_generator.py  (NEW)
# ============================================================
"""
LLM-assisted claim generation for RCA candidates.

Structurally parallel to app/services/reasoning/llm_reasoner.py.

Per approved design: the LLM authors natural-language claim statements
about candidates that already exist in a given RCACandidateResult. It may
NOT invent a candidate service that isn't already in the candidate list,
and it may NOT cite evidence outside the specific candidate's own
supporting_evidence list. Any claim violating either rule is dropped
(not the whole result) and a note is appended.

On any failure -- provider error, unparseable/invalid JSON, or a response
shape that doesn't validate -- this returns an empty ClaimGenerationResult
plus an explanatory note. It never raises out to the caller.

Only receives the already-bounded RCACandidateResult, EvidenceBundle
(column_name + score only, never raw summary stats), and ServiceGraph.
Never receives raw telemetry or anything ground-truth-adjacent.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Set

from pydantic import BaseModel, ValidationError

from app.schemas.claims import Claim, ClaimGenerationResult
from app.schemas.evidence import EvidenceBundle
from app.schemas.graph import ServiceGraph
from app.schemas.reasoning import RCACandidateResult
from app.services.claims.base import ClaimGenerator
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fences(raw: str) -> str:
    return _CODE_FENCE_RE.sub("", raw).strip()


SYSTEM_PROMPT = """You are an assistant that writes evidence-grounded claims about \
root-cause candidates for a distributed-systems incident. You will be given a fixed, \
already-ranked list of candidate services, and for each candidate the evidence column \
names and scores that support it.

For each candidate, write one or more short claims (1-2 sentences each). Each claim \
MUST cite one or more evidence column names, and every cited column name MUST come \
from that SPECIFIC candidate's own evidence list -- never from another candidate's \
evidence, and never a column name you invent.

You must NOT reference any service not already present in the candidate list you were \
given. You must NOT invent evidence column names.

Respond with ONLY a single JSON object -- no markdown, no code fences, no commentary -- \
matching exactly this shape:

{
  "claims": [
    {
      "claim_id": "<short unique id, e.g. c1>",
      "candidate_service": "<must exactly match a candidate service name>",
      "statement": "<1-2 sentence claim>",
      "cited_evidence": ["<column_name>", "..."]
    }
  ]
}

Every "candidate_service" MUST exactly match a service name from the candidate list. \
Every entry in "cited_evidence" MUST exactly match an evidence column name given for \
that candidate. Do not include any other keys.
"""


class _ClaimPayloadItem(BaseModel):
    claim_id: str
    candidate_service: str
    statement: str
    cited_evidence: List[str] = []


class _ClaimPayload(BaseModel):
    claims: List[_ClaimPayloadItem] = []


class LLMClaimGenerator(ClaimGenerator):
    def __init__(self, llm_provider: LLMProvider):
        self._llm_provider = llm_provider

    async def generate(
        self,
        candidate_result: RCACandidateResult,
        evidence: EvidenceBundle,
        graph: ServiceGraph,
    ) -> ClaimGenerationResult:
        notes: List[str] = []

        if not candidate_result.candidates:
            return ClaimGenerationResult(
                case_id=candidate_result.case_id,
                as_of=evidence.as_of,
                method="llm-generated",
                claims=[],
                notes=["No candidates available; no claims generated."],
            )

        valid_evidence_by_service: Dict[str, Set[str]] = {
            candidate.service: set(candidate.supporting_evidence)
            for candidate in candidate_result.candidates
        }

        user_prompt = self._build_user_prompt(candidate_result, graph)

        try:
            raw_output = await self._llm_provider.complete(SYSTEM_PROMPT, user_prompt)
        except Exception as exc:  # noqa: BLE001 - vendor/provider errors caught broadly, by design
            logger.warning("LLM claim generation failed: %s", exc)
            return ClaimGenerationResult(
                case_id=candidate_result.case_id,
                as_of=evidence.as_of,
                method="llm-generated",
                claims=[],
                notes=[f"LLM claim generation failed: {exc}"],
            )

        try:
            payload = self._parse_output(raw_output)
        except ValueError as exc:
            logger.warning("LLM claim generation output invalid: %s", exc)
            return ClaimGenerationResult(
                case_id=candidate_result.case_id,
                as_of=evidence.as_of,
                method="llm-generated",
                claims=[],
                notes=[f"LLM claim generation output invalid: {exc}"],
            )

        claims: List[Claim] = []
        seen_claim_ids: Set[str] = set()
        for item in payload.claims:
            reason = self._reject_reason(item, valid_evidence_by_service, seen_claim_ids)
            if reason is not None:
                notes.append(
                    f"Dropped claim '{item.claim_id}' for service "
                    f"'{item.candidate_service}': {reason}"
                )
                continue
            seen_claim_ids.add(item.claim_id)
            claims.append(
                Claim(
                    claim_id=item.claim_id,
                    candidate_service=item.candidate_service,
                    statement=item.statement,
                    cited_evidence=item.cited_evidence,
                )
            )

        if not claims and not notes:
            notes.append("LLM returned no claims.")

        return ClaimGenerationResult(
            case_id=candidate_result.case_id,
            as_of=evidence.as_of,
            method="llm-generated",
            claims=claims,
            notes=notes,
        )

    @staticmethod
    def _reject_reason(
        item: _ClaimPayloadItem,
        valid_evidence_by_service: Dict[str, Set[str]],
        seen_claim_ids: Set[str],
    ) -> str | None:
        if not item.claim_id:
            return "empty claim_id"
        if item.claim_id in seen_claim_ids:
            return "duplicate claim_id"
        if not item.statement.strip():
            return "empty statement"
        if item.candidate_service not in valid_evidence_by_service:
            return f"unknown candidate service '{item.candidate_service}'"

        allowed = valid_evidence_by_service[item.candidate_service]
        invalid_citations = [c for c in item.cited_evidence if c not in allowed]
        if invalid_citations:
            return (
                f"cited evidence not in candidate's supporting_evidence: "
                f"{sorted(invalid_citations)}"
            )
        return None

    @staticmethod
    def _build_user_prompt(candidate_result: RCACandidateResult, graph: ServiceGraph) -> str:
        lines = [f"case_id: {candidate_result.case_id}", f"system: {graph.system}", "candidates:"]
        for candidate in candidate_result.candidates:
            lines.append(
                f"- service: {candidate.service}, score: {candidate.score:.4f}, "
                f"evidence: {candidate.supporting_evidence}"
            )
        if graph.notes:
            lines.append(f"graph_notes: {graph.notes}")
        return "\n".join(lines)

    @staticmethod
    def _parse_output(raw_output: str) -> _ClaimPayload:
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
            return _ClaimPayload(**data)
        except ValidationError as exc:
            raise ValueError(f"unexpected response shape: {exc}") from exc
