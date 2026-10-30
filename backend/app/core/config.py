# ============================================================
# app/core/config.py  (MODIFIED — full file, Phase 3 additions marked)
# ============================================================
"""
Application configuration.

All runtime configuration is loaded from environment variables (via a local
.env file in development, or real environment variables in deployment).
Nothing in this file should ever contain a real secret — see .env.example
for the documented list of variables this application understands.

Settings for services introduced in later phases (Neo4j, Supabase/pgvector)
are declared here now as optional fields so the configuration surface
doesn't need to be redesigned later — they are not read or used anywhere
yet. Groq/LLM settings (Phase 2) ARE now read and used by
app/services/llm/groq_provider.py. RCAEval settings (Phase 3) ARE now read
and used by app/services/datasets/rcaeval/*.
"""

from functools import lru_cache
from pathlib import Path  # Phase 3
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Application ---
    APP_NAME: str = "RootCause-CC"
    APP_ENV: str = "development"  # development | staging | production
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # --- API ---
    API_V1_PREFIX: str = "/api/v1"

    # --- CORS ---
    # Stored as a raw comma-separated string (pydantic-settings would try to
    # JSON-decode a List[str] field straight from the env value, which
    # breaks on a plain comma-separated string). Parsed on access below.
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- LLM provider (Phase 2 — Groq) ---
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TIMEOUT: float = 30.0
    LLM_MAX_RETRIES: int = 2

    # --- RCAEval dataset foundation (Phase 3) ---
    # Directory that RCAEval suite folders (RE1/, RE2/, RE3/) are downloaded
    # into. Never committed to git — see .gitignore and scripts/download_rcaeval.sh.
    RCAEVAL_DATA_DIR: str = "../data/rcaeval-data"

    # Optional hard cap on how many cases discovery/listing will ever return,
    # regardless of how many exist on disk. None = no cap (full suite).
    # Intended for local development so iteration doesn't require touching
    # the entire downloaded dataset. Can be overridden per-call (CLI/API),
    # but this is the default when no explicit limit is passed.
    RCAEVAL_DEV_MODE_MAX_CASES: Optional[int] = None

    # --- Reserved for later phases (not read/used yet) ---
    NEO4J_URI: Optional[str] = None
    NEO4J_USER: Optional[str] = None
    NEO4J_PASSWORD: Optional[str] = None
    SUPABASE_DB_URL: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def groq_configured(self) -> bool:
        return bool(self.GROQ_API_KEY)

    @property
    def rcaeval_data_dir_path(self) -> Path:
        """
        Resolved Path form of RCAEVAL_DATA_DIR. Every Phase 3 consumer
        (discovery, CLI, tests) should use this property rather than
        re-wrapping the raw string field themselves — mirrors the existing
        cors_origins_list / groq_configured pattern of exposing one derived
        accessor instead of letting every call site re-parse the raw value.
        Does not create the directory or validate it exists; callers that
        need it to exist (e.g. discovery) are responsible for that check
        and for raising a clear error if it's missing.
        """
        return Path(self.RCAEVAL_DATA_DIR)


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor. FastAPI dependencies should use this via
    Depends(get_settings) rather than importing a module-level instance,
    so settings can be overridden cleanly in tests.
    """
    return Settings()