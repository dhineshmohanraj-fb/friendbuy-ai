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

    # Graph DB (Kuzu) — CP1
    graph_db_dir: str = "./friendbuy-graph-db"
    use_graph: bool = True              # set USE_GRAPH=false to skip graph entirely

    # File loading — skip huge files to protect M1 RAM
    file_size_cap_bytes: int = 512_000  # 500 KB

    # CP3 — Hybrid retrieval
    use_bm25:       bool = True   # BM25 sparse search alongside vector
    hybrid_rrf_k:   int  = 60    # Reciprocal Rank Fusion constant
    graph_max_hops: int  = 2     # max traversal depth in Kuzu
    bm25_top_k:     int  = 20    # candidates from BM25 before RRF
    vector_top_k:   int  = 20    # candidates from dense search before RRF

    # CP4 — Semantic cache
    use_semantic_cache:        bool  = True   # enable semantic query cache
    semantic_cache_threshold:  float = 0.93   # cosine similarity hit threshold
    semantic_cache_max_size:   int   = 1000   # max entries (LRU eviction)

    # CP4 — Reranker
    use_reranker: bool = True   # cross-encoder reranking via flashrank

    # CP4 — Cross-repo linking
    use_cross_repo_linking: bool = True   # detect HTTP/Kafka cross-repo edges

    # CP5 — Observability
    log_level: str  = "INFO"            # DEBUG | INFO | WARNING | ERROR
    log_file:  str  = "./cache/app.log" # set to "" to disable file logging

    # CP5 — API auth
    api_key: str | None = None          # Bearer token; if None, auth is disabled

    # CP5 — Embedding drift detection
    drift_similarity_threshold: float = 0.999   # cosine threshold for "same model"

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

    @property
    def graph_db_path(self) -> Path:
        return Path(self.graph_db_dir)


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
