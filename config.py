"""
config.py — Central configuration using Pydantic Settings.

All values are loaded from the .env file (or environment variables).
Never hardcode secrets — always add them to .env (which is in .gitignore).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., description="Groq API key")
    llm_model_name: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model identifier",
    )

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace sentence-transformers model name",
    )

    # ── Vector Store ───────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(
        default="./chroma_data",
        description="Directory for ChromaDB persistent storage",
    )

    # ── Audit Trail ────────────────────────────────────────────────────────────
    audit_db_path: str = Field(
        default="./audit_trail.db",
        description="SQLite database path for audit trail",
    )

    # ── Pipeline Thresholds ────────────────────────────────────────────────────
    confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to pass guardrail",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Max LLM retries before escalating to human review",
    )

    # ── Data Paths ─────────────────────────────────────────────────────────────
    raw_regulatory_dir: str = Field(
        default="./data/raw_regulatory",
        description="Directory for regulatory PDFs",
    )
    raw_internal_policy_dir: str = Field(
        default="./data/raw_internal_policy",
        description="Directory for internal policy PDFs",
    )

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for path_str in [
            self.chroma_persist_dir,
            self.raw_regulatory_dir,
            self.raw_internal_policy_dir,
        ]:
            Path(path_str).mkdir(parents=True, exist_ok=True)


# Singleton instance — import this everywhere
settings = Settings()


if __name__ == "__main__":
    print(settings.model_dump(exclude={"groq_api_key"}))
    print("✓ Config loaded successfully. GROQ_API_KEY is set:", bool(settings.groq_api_key))
