# ============================================================
# app/services/final_result/service.py  (NEW)
# ============================================================
"""
Final Evidence-Grounded RCA Result Assembly.

Pure, deterministic combination of an already-produced RCACandidateResult
and ClaimVerificationResult. Does not call the reasoning layer, the
claims layer, EvidenceRetrievalService, TopologyProvider, or any LLM
provider itself -- it only combines outputs already produced by callers,
same pattern as ClaimVerificationService not calling
EvidenceRetrievalService directly.

No DatasetAdapter, no GroundTruth, no raw telemetry access anywhere in
this module.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from app.schemas.claims import ClaimVerdict, ClaimVerificationResult
from app.schemas.final_result import (
    CandidateGroundingStatus,
    FinalRCAResult,
    GroundedCandidate,
)
from app.schemas.reasoning import RCACandidateResult


class FinalResultAssemblyService:
    def assemble(
        self,
        candidate_result: RCACandidateResult,
        verification_result: ClaimVerificationResult,
    ) -> FinalRCAResult:
        notes: List[str] = list(candidate_result.notes) + list(verification_result.notes)

        if candidate_result.case_id != verification_result.case_id:
            notes.append(
                "case_id mismatch between RCACandidateResult "
                f"('{candidate_result.case_id}') and ClaimVerificationResult "
                f"('{verification_result.case_id}'); using RCACandidateResult.case_id."
            )

        if candidate_result.as_of != verification_result.as_of:
            notes.append(
                "as_of mismatch between RCACandidateResult "
                f"('{candidate_result.as_of}') and ClaimVerificationResult "
                f"('{verification_result.as_of}'); using ClaimVerificationResult.as_of."
            )

        verdicts_by_service: Dict[str, List[ClaimVerdict]] = defaultdict(list)
        for verdict in verification_result.verdicts:
            verdicts_by_service[verdict.candidate_service].append(verdict)

        grounded_candidates: List[GroundedCandidate] = []
        for candidate in candidate_result.candidates:
            candidate_verdicts = verdicts_by_service.get(candidate.service, [])

            supporting = [v.claim_id for v in candidate_verdicts if v.status == "supported"]
            contradicting = [v.claim_id for v in candidate_verdicts if v.status == "unsupported"]
            uncertain = [v.claim_id for v in candidate_verdicts if v.status == "uncertain"]

            status = self._decide_grounding_status(candidate_verdicts)

            grounded_candidates.append(
                GroundedCandidate(
                    candidate=candidate,
                    grounding_status=status,
                    supporting_claim_ids=supporting,
                    contradicting_claim_ids=contradicting,
                    uncertain_claim_ids=uncertain,
                )
            )

        return FinalRCAResult(
            case_id=candidate_result.case_id,
            as_of=verification_result.as_of,
            candidates=grounded_candidates,
            notes=notes,
        )

    @staticmethod
    def _decide_grounding_status(
        candidate_verdicts: List[ClaimVerdict],
    ) -> CandidateGroundingStatus:
        if not candidate_verdicts:
            return "uncertain"
        if any(v.status == "unsupported" for v in candidate_verdicts):
            return "unsupported"
        if all(v.status == "supported" for v in candidate_verdicts):
            return "supported"
        return "uncertain"
