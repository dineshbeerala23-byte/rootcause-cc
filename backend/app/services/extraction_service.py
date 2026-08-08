"""
Entity/relationship extraction service (Phase 2).

Sits between the API layer and the LLM provider:

    API layer  ->  ExtractionService  ->  LLMProvider  ->  Groq

Owns the extraction prompt and all parsing/validation of the raw LLM
response. The API layer never sees unvalidated LLM output — this module's
job is to turn "whatever text the model returned" into a validated
ExtractionResult, or raise a RootCauseError subclass that the existing
global exception handlers turn into a clean HTTP response.
"""

import json
import logging
import re
from typing import Any, List, Set

from pydantic import ValidationError

from app.core.exceptions import ExtractionError
from app.schemas.extraction import Entity, ExtractionResult, Relationship
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an information extraction engine for RootCause-CC, an incident \
intelligence platform. Given raw cloud/system/incident text, extract the entities and \
relationships it describes.

Respond with ONLY a single JSON object — no markdown, no code fences, no commentary — \
matching exactly this shape:

{
  "entities": [
    {"id": "e1", "name": "...", "type": "Service", "description": "...", "attributes": {}}
  ],
  "relationships": [
    {"source_id": "e1", "target_id": "e2", "type": "DEPENDS_ON", "description": "..."}
  ]
}

Allowed entity "type" values: Service, Application, Host, Container, Pod, Database, \
Deployment, Error, Incident, Alert, Dependency, Configuration, CloudResource.

Allowed relationship "type" values: DEPENDS_ON, CALLS, RUNS_ON, CONNECTED_TO, CAUSED_BY, \
AFFECTS, TRIGGERED, FAILED_DUE_TO, PART_OF.

Rules:
- Assign short sequential ids to entities: "e1", "e2", "e3", ...
- Every relationship's source_id and target_id MUST reference the id of an entity present \
in the same "entities" list.
- "attributes" and "description" are optional — omit them or use empty values if unknown.
- Do not invent facts that are not supported by the text.
- If the text contains no relevant entities or relationships, return \
{"entities": [], "relationships": []}.
"""

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fences(raw: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrapping some models add despite instructions."""
    return _CODE_FENCE_RE.sub("", raw).strip()


class ExtractionService:
    """Coordinates prompting the LLM provider and validating its response."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    @property
    def provider_name(self) -> str:
        return self._llm_provider.name

    @property
    def model_name(self) -> str:
        return self._llm_provider.model

    async def extract(self, text: str) -> ExtractionResult:
        user_prompt = f"Text to analyze:\n\n{text}"
        raw_output = await self._llm_provider.complete(SYSTEM_PROMPT, user_prompt)
        return self._parse_output(raw_output)

    def _parse_output(self, raw_output: str) -> ExtractionResult:
        if not raw_output or not raw_output.strip():
            raise ExtractionError("The LLM provider returned an empty response.")

        cleaned = _strip_code_fences(raw_output)

        try:
            payload: Any = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse LLM output as JSON: %s", exc)
            raise ExtractionError(
                "The LLM provider returned a response that could not be parsed as JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise ExtractionError("The LLM provider returned an unexpected response shape.")

        warnings: List[str] = []

        entities = self._parse_entities(payload.get("entities"), warnings)
        seen_ids: Set[str] = {entity.id for entity in entities}
        relationships = self._parse_relationships(payload.get("relationships"), seen_ids, warnings)

        return ExtractionResult(entities=entities, relationships=relationships, warnings=warnings)

    @staticmethod
    def _parse_entities(raw_entities: Any, warnings: List[str]) -> List[Entity]:
        entities: List[Entity] = []
        seen_ids: Set[str] = set()

        if raw_entities is None:
            return entities
        if not isinstance(raw_entities, list):
            warnings.append("Ignored 'entities': expected a list.")
            return entities

        for index, raw_entity in enumerate(raw_entities):
            if not isinstance(raw_entity, dict):
                warnings.append(f"Skipped entity at index {index}: not an object.")
                continue
            try:
                entity = Entity(**raw_entity)
            except ValidationError as exc:
                warnings.append(
                    f"Skipped invalid entity at index {index} "
                    f"({raw_entity.get('id', '?')}): {_summarize_validation_error(exc)}"
                )
                continue

            if entity.id in seen_ids:
                warnings.append(f"Skipped duplicate entity id '{entity.id}'.")
                continue

            seen_ids.add(entity.id)
            entities.append(entity)

        return entities

    @staticmethod
    def _parse_relationships(
        raw_relationships: Any,
        valid_entity_ids: Set[str],
        warnings: List[str],
    ) -> List[Relationship]:
        relationships: List[Relationship] = []

        if raw_relationships is None:
            return relationships
        if not isinstance(raw_relationships, list):
            warnings.append("Ignored 'relationships': expected a list.")
            return relationships

        for index, raw_rel in enumerate(raw_relationships):
            if not isinstance(raw_rel, dict):
                warnings.append(f"Skipped relationship at index {index}: not an object.")
                continue
            try:
                relationship = Relationship(**raw_rel)
            except ValidationError as exc:
                warnings.append(
                    f"Skipped invalid relationship at index {index}: {_summarize_validation_error(exc)}"
                )
                continue

            if relationship.source_id not in valid_entity_ids or relationship.target_id not in valid_entity_ids:
                warnings.append(
                    f"Skipped relationship '{relationship.source_id}' -> '{relationship.target_id}': "
                    "references an entity id that was not extracted."
                )
                continue

            relationships.append(relationship)

        return relationships


def _summarize_validation_error(exc: ValidationError) -> str:
    """Short, log/warning-friendly summary of a pydantic ValidationError."""
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first.get("loc", ()))
    return f"{field or 'value'}: {first.get('msg', 'invalid')}"