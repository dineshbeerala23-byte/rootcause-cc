# ============================================================
# app/services/claims/verifier.py  (NEW)
# ============================================================
"""
Deterministic, evidence-grounded claim verification.

Structurally parallel to app/services/reasoning/deterministic.py:
consumes only ClaimGenerationResult + EvidenceBundle. No LLM, no
telemetry access, no ground truth, no DatasetAdapter. Deterministic and
fully testable offline.

Independently re-derives which evidence column names exist and which
service each is attributed to directly from EvidenceBundle.items -- it
never trusts a Claim's own cited_evidence as proof of anything, only as
a set of references to check.

No scores or thresholds are used anywhere in this module. Verdicts are
decided purely by set membership/attribution, per the approved semantics:

  - supported:   cited_evidence is non-empty and every citation is found
                  in the bundle, attributed to candidate_service.
  - unsupported: cited_evidence is non-empty and none of the citations
                  are found attributed to candidate_service.
  - uncertain:   partial attribution/match, or cited_evidence is empty.
"""

from __future__ import annotations

from typing import Dict, List

from app.schemas.claims import (
    ClaimGenerationResult,
    ClaimVerdict,
    ClaimVerificationResult,
)
from app.schemas.evidence import EvidenceBundle
from app.services.claims.base import ClaimVerifier


class DeterministicClaimVerifier(ClaimVerifier):
    def verify(
        self,
        claim_result: ClaimGenerationResult,
        evidence: EvidenceBundle,
    ) -> ClaimVerificationResult:
        notes: List[str] = list(claim_result.notes)

        # column_name -> attributed service (None if the item has no
        # service attribution; such items can never support a claim
        # about a specific candidate_service).
        column_to_service: Dict[str, str] = {
            item.column_name: item.service
            for item in evidence.items
            if item.service is not None
        }
        known_columns = {item.column_name for item in evidence.items}

        if not claim_result.claims:
            notes.append("No claims to verify.")
            return ClaimVerificationResult(
                case_id=claim_result.case_id,
                as_of=evidence.as_of,
                verdicts=[],
                notes=notes,
            )

        verdicts: List[ClaimVerdict] = []
        for claim in claim_result.claims:
            matched: List[str] = []
            unmatched: List[str] = []

            for column in claim.cited_evidence:
                attributed_service = column_to_service.get(column)
                if column in known_columns and attributed_service == claim.candidate_service:
                    matched.append(column)
                else:
                    unmatched.append(column)

            status = self._decide_status(matched=matched, unmatched=unmatched)

            verdicts.append(
                ClaimVerdict(
                    claim_id=claim.claim_id,
                    candidate_service=claim.candidate_service,
                    status=status,
                    matched_evidence=matched,
                    unmatched_evidence=unmatched,
                )
            )

        return ClaimVerificationResult(
            case_id=claim_result.case_id,
            as_of=evidence.as_of,
            verdicts=verdicts,
            notes=notes,
        )

    @staticmethod
    def _decide_status(matched: List[str], unmatched: List[str]) -> str:
        if not matched and not unmatched:
            # empty cited_evidence
            return "uncertain"
        if matched and not unmatched:
            return "supported"
        if unmatched and not matched:
            return "unsupported"
        return "uncertain"
