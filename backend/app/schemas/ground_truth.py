# ============================================================
# app/schemas/ground_truth.py  (NEW)
# ============================================================
"""
Ground-truth schema for RCA evaluation.

DELIBERATELY isolated in its own module, separate from app/schemas/incident.py
and app/schemas/telemetry.py. GroundTruth carries the actual answer to an RCA
case (which service failed, what kind of fault). It must never be reachable
from any object that gets built into an LLM prompt, an extraction payload, or
a default API response body.

Import-path rule (enforced by convention now, structurally in Part 5 once the
adapter exists): only evaluation-only code — not inference code, not the
default case-inspection endpoint — should ever import from this module. If
you find yourself importing GroundTruth from anything under
app/services/datasets/rcaeval/adapter.py's inference-facing path, that's a bug.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GroundTruth(BaseModel):
    """
    The actual answer for one RCAEval case. Joined to a NormalizedIncident
    only via its opaque case_id (see app/services/datasets/rcaeval/
    case_id_registry.py, Part 3) — never embedded inside NormalizedIncident
    itself.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(
        ...,
        description=(
            "Opaque case ID (e.g. 'case-0001'), matching the id used in "
            "NormalizedIncident.case_id. Never the raw RCAEval directory name."
        ),
    )
    root_cause_service: str = Field(
        ...,
        description="The service RCAEval injected the fault into. This IS the answer.",
    )
    fault_type: str = Field(
        ...,
        description=(
            "RCAEval fault-type token (e.g. 'cpu', 'mem', 'delay', 'f1'). "
            "This IS the answer — treat with the same care as root_cause_service."
        ),
    )
    root_cause_indicator: Optional[str] = Field(
        default=None,
        description=(
            "Specific metric/log signal RCAEval annotates as the root-cause "
            "indicator for this case, where available."
        ),
    )