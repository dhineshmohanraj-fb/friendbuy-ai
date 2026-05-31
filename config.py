"""Central configuration loaded from environment / .env file."""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Anthropic — None means "not configured"; empty/whitespace is coerced to None
    anthropic_api_key: str | None = None

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    local_model: str = "qwen2.5:3b"
    embedding_model: str = "nomic-embed-text"

    # ChromaDB
    chroma_persist_dir: str = "./friendbuy-knowledge-base"
    chroma_collection_name: str = "friendbuy_codebase"

    # Repos
    repos_dir: str = "./repos"

    # Retrieval
    top_k_results: int = 5
    min_relevance_score: float = 0.30   # chunks below this score are dropped

    # Claude
    claude_model: str = "claude-sonnet-4-5"
    claude_max_tokens: int = 4096

    # Local / Qwen model
    qwen_temperature: float = 0.0

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embedding batching (keep low on 8 GB M1)
    embed_batch_size: int = 100

    # Cache / delta tracking
    cache_dir: str = "./cache"

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ---------------------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------------------

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def _strip_api_key(cls, v: str | None) -> str | None:
        """Coerce empty / whitespace-only string to None so checks are unambiguous."""
        if isinstance(v, str):
            stripped = v.strip()
            return stripped if stripped else None
        return v

    # ---------------------------------------------------------------------------
    # Computed paths
    # ---------------------------------------------------------------------------

    @property
    def repos_path(self) -> Path:
        return Path(self.repos_dir)

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application settings singleton.

    Wrapped in ``lru_cache`` so tests can call
    ``get_settings.cache_clear()`` to reset the singleton between runs.
    """
    return Settings()


# Module-level alias — keeps ``from config import settings`` working everywhere.
settings = get_settings()

# ---------------------------------------------------------------------------
# File-discovery constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rb", ".go",
    ".md", ".mdx", ".txt", ".rst",
    ".yml", ".yaml", ".json", ".toml",
    ".html", ".css", ".scss",
    ".sql", ".sh",
}

SUPPORTED_FILENAMES: set[str] = {"Dockerfile", ".env.example"}

SKIP_DIRS: set[str] = {
    "node_modules", ".git", "__pycache__", "dist", "build",
    ".next", "venv", ".venv", ".mypy_cache", ".pytest_cache",
    "coverage", ".tox", "eggs", ".eggs",
}

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "js",
    ".ts": "js",
    ".jsx": "js",
    ".tsx": "js",
    ".go": "go",
    ".rb": "ruby",
}
