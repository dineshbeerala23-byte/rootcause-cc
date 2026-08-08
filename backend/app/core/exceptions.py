# ============================================================
# app/core/exceptions.py  (MODIFIED — full file, Phase 3 additions marked)
# ============================================================
"""
Application-wide error handling.

RootCauseError (and its subclasses) is what application/service code should
raise for expected, domain-specific failure conditions. Later phases (graph
DB, vector store, LLM calls) should raise subclasses of this rather than
letting raw driver exceptions leak out of the API layer.

All handlers return a consistent JSON error shape:
    { "error": str, "message": str, "path": str, "details"?: ... }
"""

import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class RootCauseError(Exception):
    """Base class for all application-specific errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ServiceUnavailableError(RootCauseError):
    """
    Raised when a required downstream dependency (graph DB, vector store,
    LLM provider) cannot be reached.
    """

    def __init__(self, message: str = "A required service is currently unavailable.") -> None:
        super().__init__(message, status.HTTP_503_SERVICE_UNAVAILABLE)


class LLMProviderError(ServiceUnavailableError):
    """
    Raised when the configured LLM provider (Phase 2: Groq) cannot fulfill
    a request — missing configuration, timeout, rate limit, transport
    failure, etc. Caught at the API boundary and turned into a consistent
    503 response rather than leaking a raw SDK exception.
    """

    def __init__(self, message: str = "The LLM provider is currently unavailable.") -> None:
        super().__init__(message)


class ExtractionError(RootCauseError):
    """
    Raised when the LLM provider responded, but its output could not be
    turned into a valid extraction result (empty response, unparseable
    JSON, or a response shape with no salvageable content). Used by
    app.services.extraction_service so malformed model output degrades to
    a clean API error instead of crashing the request.
    """

    def __init__(
        self,
        message: str = "Failed to extract structured data from the provided text.",
    ) -> None:
        super().__init__(message, status.HTTP_502_BAD_GATEWAY)


# --- Dataset foundation (Phase 3: RCAEval and future dataset adapters) ---


class DatasetError(RootCauseError):
    """
    Base class for errors raised by app.services.datasets.* (RCAEval now,
    other dataset adapters later). Subclass this rather than raising
    RootCauseError directly from dataset code, so dataset-layer failures
    are easy to catch/distinguish as a category at the API boundary if we
    ever want to (e.g. a shared "dataset problem" log tag).
    """

    def __init__(
        self,
        message: str = "A dataset operation failed.",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ) -> None:
        super().__init__(message, status_code)


class DatasetNotConfiguredError(ServiceUnavailableError):
    """
    Raised when RCAEVAL_DATA_DIR does not exist or contains no downloaded
    suite data yet. Deliberately a ServiceUnavailableError (503), not a
    DatasetError — from the caller's point of view this is the same shape
    of problem as an unreachable LLM provider: a required dependency
    (here, downloaded data on disk rather than a network service) is
    missing. Raised by discovery (Part 4) before it ever attempts to list
    or open case directories.
    """

    def __init__(
        self,
        message: str = (
            "RCAEval data directory is missing or empty. Run "
            "scripts/download_rcaeval.sh (see README) before using dataset endpoints."
        ),
    ) -> None:
        super().__init__(message)


class DatasetCaseNotFoundError(DatasetError):
    """
    Raised when a requested opaque case_id does not exist in the case ID
    registry (app.services.datasets.rcaeval.case_id_registry, Part 3) or
    is not present under the configured suite/system filters. Distinct
    from DatasetNotConfiguredError: the dataset itself is fine, this one
    case just doesn't exist. Used by discovery (Part 4), the adapter
    (Part 5), the API endpoints (Part 7), and the CLI (Part 8).
    """

    def __init__(self, case_id: str) -> None:
        super().__init__(
            f"No dataset case found for case_id={case_id!r}.",
            status.HTTP_404_NOT_FOUND,
        )
        self.case_id = case_id


class CorruptDatasetFileError(DatasetError):
    """
    Raised when a case is found (its directory/case_id is valid) but one
    of its expected files (metrics.json, inject_time.txt, logs.csv,
    traces.csv) is missing when it should be present for that case's
    suite, or fails to parse (malformed JSON/CSV, unreadable timestamp).
    Raised by the raw loader (Part 3), never silently swallowed into an
    empty DataFrame — a corrupt file is a data-integrity problem the
    caller needs to know about, not something to paper over.
    """

    def __init__(self, case_id: str, file_name: str, reason: str) -> None:
        super().__init__(
            f"Corrupt or unreadable dataset file for case_id={case_id!r}: "
            f"{file_name} ({reason}).",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        self.case_id = case_id
        self.file_name = file_name


class GroundTruthAccessError(RootCauseError):
    """
    Raised when ground-truth data (app.schemas.ground_truth.GroundTruth) is
    requested from a code path that is not the designated evaluation-only
    accessor. This is a safety-invariant guard, not an ordinary runtime
    condition — it should only ever fire if a future phase's inference
    code accidentally imports/calls the ground-truth accessor directly
    instead of going through the evaluation harness. Deliberately NOT a
    DatasetError subclass: this is a "we almost leaked the answer" bug,
    not a dataset-availability or data-quality problem, and it's worth
    keeping that distinction visible in the exception hierarchy and in
    logs/alerts if this is ever wired up to one.
    """

    def __init__(
        self,
        message: str = (
            "Ground truth was requested from a non-evaluation code path. "
            "This is a safety-invariant violation, not a normal runtime error."
        ),
    ) -> None:
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)


async def rootcause_exception_handler(request: Request, exc: RootCauseError) -> JSONResponse:
    logger.error("Application error on %s: %s", request.url.path, exc.message, exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "path": str(request.url.path),
        },
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Request validation failed.",
            "details": exc.errors(),
            "path": str(request.url.path),
        },
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    logger.warning("HTTP exception on %s: %s", request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "message": exc.detail,
            "path": str(request.url.path),
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.critical("Unhandled exception on %s", request.url.path, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred.",
            "path": str(request.url.path),
        },
    )