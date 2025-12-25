# config.py
"""Centralized configuration management for Agent RAG MCP."""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# Load environment variables from .env file (once at module load)
load_dotenv(override=True)


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables.

    All environment variables are loaded once and cached for efficiency.
    """

    # Gemini API
    gemini_api_key: str | None

    # Document Store - Git Repository
    rag_repo_url: str | None
    rag_docs_path: str
    rag_branch: str

    # Document Store - Local Path
    rag_local_docs_path: str | None

    # Document Store - Options
    rag_store_name: str | None
    rag_force_reindex: bool

    # Authentication
    auth_token: str | None

    @property
    def is_auth_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return bool(self.auth_token)

    @property
    def has_document_source(self) -> bool:
        """Check if a document source is configured."""
        return bool(self.rag_repo_url or self.rag_local_docs_path)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get the application configuration (cached).

    Returns:
        Config object with all settings loaded from environment variables.
    """
    return Config(
        # Gemini API
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        # Document Store - Git Repository
        rag_repo_url=os.getenv("RAG_REPO_URL"),
        rag_docs_path=os.getenv("RAG_DOCS_PATH", "Docs"),
        rag_branch=os.getenv("RAG_BRANCH", "main"),
        # Document Store - Local Path
        rag_local_docs_path=os.getenv("RAG_LOCAL_DOCS_PATH"),
        # Document Store - Options
        rag_store_name=os.getenv("RAG_STORE_NAME"),
        rag_force_reindex=os.getenv("RAG_FORCE_REINDEX", "").lower()
        in ("true", "1", "yes"),
        # Authentication
        auth_token=os.getenv("AUTH_TOKEN"),
    )


def reload_config() -> Config:
    """Force reload configuration (clears cache).

    Useful for testing or when environment variables change at runtime.

    Returns:
        Fresh Config object with reloaded settings.
    """
    get_config.cache_clear()
    return get_config()
