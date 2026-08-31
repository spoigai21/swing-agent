"""Configuration. Fails loudly at import rather than at 3am inside a poller."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from swing.paths import CONFIG, ENV_FILE, ROOT

# Re-exported for callers that predate swing.paths.
REPO_ROOT = ROOT
CONFIG_DIR = CONFIG


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    # --- Phase -1: the collector needs only these two ---
    sec_user_agent: str
    database_url: str

    # --- Phase 0 onward ---
    finnhub_api_key: str = ""
    gemini_api_key: str = ""
    marketaux_api_key: str = ""
    tiingo_api_key: str = ""
    attribution_model: str = "google_genai:gemini-flash-latest"

    poll_jitter_pct: float = Field(default=0.1, ge=0.0, le=0.5)

    @field_validator("sec_user_agent")
    @classmethod
    def _sec_ua_has_contact(cls, v: str) -> str:
        # SEC returns 403 without a descriptive UA, and may block rather than
        # contact you if it lacks an email. data-sources.md D.2 rule 1.
        if "@" not in v:
            raise ValueError(
                "SEC_USER_AGENT must contain a contact email address, "
                f"got {v!r}. SEC will return 403 otherwise."
            )
        return v

    def require(self, *names: str) -> None:
        """Assert optional keys are present before a stage that needs them."""
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            raise RuntimeError(f"missing required settings: {', '.join(missing)}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
