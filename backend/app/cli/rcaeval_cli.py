"""
RCAEval dataset CLI (Phase 3, Part 8).

Run with:
    python -m app.cli.rcaeval_cli <command> [options]

Thin by design, same convention as app/api/v1/endpoints/datasets.py: this
module only formats output. All dataset logic lives in
app.services.datasets.rcaeval.*. Same leakage rules as the API layer:
opaque case IDs only, no raw directory names, no ground truth -- this CLI
intentionally has no "ground-truth" or "answer" command.

`analyze` (NEW -- Evidence-Grounded RCA Pipeline API/CLI wiring milestone)
runs the full incident -> FinalRCAResult pipeline via
app.services.pipeline.service.RCAPipelineService, using the exact same
orchestration the API endpoint uses. Additive only; the three original
commands (list/inspect/process) are unchanged.
"""

import argparse
import asyncio
import json
import sys
from typing import Optional

from app.core.config import get_settings
from app.core.exceptions import RootCauseError
from app.schemas.telemetry import TelemetryQuery
from app.services.claims.llm_generator import LLMClaimGenerator
from app.services.claims.service import ClaimVerificationService
from app.services.datasets.base import DatasetAdapter
from app.services.datasets.rcaeval.adapter import RCAEvalAdapter
from app.services.evidence.service import EvidenceRetrievalService
from app.services.final_result.service import FinalResultAssemblyService
from app.services.graph.provider import StaticTopologyProvider
from app.services.llm.groq_provider import GroqProvider
from app.services.pipeline.service import RCAPipelineService
from app.services.reasoning.llm_reasoner import LLMRationaleAnnotator
from app.services.reasoning.service import RCAReasoningService
from app.services.evaluation.service import EvaluationService

def _get_adapter() -> DatasetAdapter:
    return RCAEvalAdapter(settings=get_settings())


def _print_json(obj) -> None:
    if isinstance(obj, list):
        print(json.dumps([json.loads(item.model_dump_json()) for item in obj], indent=2))
    else:
        print(obj.model_dump_json(indent=2))


def cmd_list(args: argparse.Namespace) -> int:
    adapter = _get_adapter()
    cases = adapter.list_cases()

    if args.suite is not None:
        cases = [c for c in cases if c.dataset_suite == args.suite]
    if args.system is not None:
        cases = [c for c in cases if c.system == args.system]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    if not cases:
        print("No cases found.", file=sys.stderr)
        return 0
    _print_json(cases)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    adapter = _get_adapter()
    incident = adapter.get_incident(args.case_id)
    print("--- incident ---")
    _print_json(incident)

    summary = adapter.get_telemetry_summary(args.case_id, incident.diagnostic_window)
    print("--- telemetry summary ---")
    _print_json(summary)
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    adapter = _get_adapter()
    query = TelemetryQuery(service=args.service, max_rows=args.max_rows)
    handle = adapter.get_telemetry(args.case_id, query)

    output = {
        "case_id": args.case_id,
        "metrics": json.loads(handle.metrics().to_json(orient="records")),
        "logs": json.loads(handle.logs().to_json(orient="records")),
        "traces": json.loads(handle.traces().to_json(orient="records")),
    }
    print(json.dumps(output, indent=2))
    return 0


def _build_pipeline_service() -> RCAPipelineService:
    settings = get_settings()
    adapter = _get_adapter()
    evidence_service = EvidenceRetrievalService(adapter)
    topology_provider = StaticTopologyProvider()
    llm_provider = GroqProvider(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        timeout=settings.LLM_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES,
    )
    reasoning_service = RCAReasoningService(
        evidence_service,
        topology_provider,
        rationale_annotator=LLMRationaleAnnotator(llm_provider),
    )
    claims_service = ClaimVerificationService(claim_generator=LLMClaimGenerator(llm_provider))
    final_result_service = FinalResultAssemblyService()

    return RCAPipelineService(
        adapter=adapter,
        evidence_service=evidence_service,
        topology_provider=topology_provider,
        reasoning_service=reasoning_service,
        claims_service=claims_service,
        final_result_service=final_result_service,
        llm_provider=llm_provider,
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    pipeline_service = _build_pipeline_service()
    response = asyncio.run(pipeline_service.analyze(args.case_id, mode=args.mode))
    print(response.model_dump_json(indent=2))
    return 0

def cmd_evaluate(args: argparse.Namespace) -> int:
    adapter = _get_adapter()
    cases = adapter.list_cases()

    if args.suite is not None:
        cases = [c for c in cases if c.dataset_suite == args.suite]
    if args.system is not None:
        cases = [c for c in cases if c.system == args.system]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    if not cases:
        print("No cases found.", file=sys.stderr)
        return 0

    case_ids = [c.case_id for c in cases]
    pipeline_service = _build_pipeline_service()
    evaluation_service = EvaluationService(adapter=adapter, pipeline_service=pipeline_service)

    run_result = asyncio.run(evaluation_service.evaluate_cases(case_ids, mode=args.mode))
    print(run_result.model_dump_json(indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli.rcaeval_cli",
        description="Inspect RCAEval dataset cases (opaque IDs only, no ground truth).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser("list", help="List available cases.")
    p_list.add_argument("--suite", choices=["RE1", "RE2", "RE3"], default=None)
    p_list.add_argument("--system", choices=["online-boutique", "sock-shop", "train-ticket"], default=None)
    p_list.add_argument("--max-cases", type=int, default=None, dest="max_cases")
    p_list.set_defaults(func=cmd_list)

    p_inspect = subparsers.add_parser("inspect", help="Show a normalized incident + telemetry summary.")
    p_inspect.add_argument("case_id", help="Opaque case ID, e.g. case-0001")
    p_inspect.set_defaults(func=cmd_inspect)

    p_process = subparsers.add_parser("process", help="Print bounded telemetry rows for a case.")
    p_process.add_argument("case_id", help="Opaque case ID, e.g. case-0001")
    p_process.add_argument("--service", default=None, help="Filter to one service name.")
    p_process.add_argument("--max-rows", type=int, default=20, dest="max_rows")
    p_process.set_defaults(func=cmd_process)

    p_analyze = subparsers.add_parser(
        "analyze", help="Run the full evidence-grounded RCA pipeline for a case."
    )
    p_analyze.add_argument("case_id", help="Opaque case ID, e.g. case-0001")
    p_analyze.add_argument(
        "--mode",
        choices=["deterministic", "llm-assisted"],
        default="deterministic",
        help="Reasoning mode -- controls candidate rationale only; claim "
        "generation always uses the LLM regardless of this setting.",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    p_evaluate = subparsers.add_parser(
        "evaluate", help="Run evaluation (predicted_rank/top-k/grounding precision-recall) over cases."
    )
    p_evaluate.add_argument("--suite", choices=["RE1", "RE2", "RE3"], default=None)
    p_evaluate.add_argument("--system", choices=["online-boutique", "sock-shop", "train-ticket"], default=None)
    p_evaluate.add_argument("--max-cases", type=int, default=None, dest="max_cases")
    p_evaluate.add_argument(
        "--mode",
        choices=["deterministic", "llm-assisted"],
        default="deterministic",
        help="Reasoning mode used when running the pipeline for each case being evaluated.",
    )
    p_evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RootCauseError as exc:
        print(f"Error: {exc.__class__.__name__}: {exc.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
