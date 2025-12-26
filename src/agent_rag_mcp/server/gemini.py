# gemini.py
"""Unified Gemini Client for Agent RAG MCP.

Handles both standard content generation and File Search RAG operations.
Uses thread-based async wrapping for stability with FastMCP.
"""

import asyncio
from pathlib import Path

from google import genai
from google.genai import types

from agent_rag_mcp.core.config import get_config


class GeminiClient:
    """Unified client for Gemini operations using thread-safe async pattern."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the Gemini client.

        Args:
            api_key: Optional API key. If not provided, uses config.
        """
        config = get_config()
        self.api_key = api_key or config.gemini_api_key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        self.client = genai.Client(api_key=self.api_key)
        self.file_search_store_name: str | None = None

    # ==============================================================================
    # Standard Generation
    # ==============================================================================
    async def generate_content(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.7,
    ) -> str:
        """Generate content using Gemini.

        Args:
            prompt: The input prompt
            model: Model to use
            temperature: Generation temperature

        Returns:
            Generated text content
        """
        def _sync_generate() -> str:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            return response.text or ""
        
        return await asyncio.to_thread(_sync_generate)

    # ==============================================================================
    # File Search / Document RAG
    # ==============================================================================
    async def check_store_exists(self, store_display_name: str) -> tuple[str | None, bool]:
        """Check if a store with the given name already exists."""
        def _sync_check() -> tuple[str | None, bool]:
            for store in self.client.file_search_stores.list():
                if store.display_name == store_display_name:
                    return store.name, True
            return None, False
        
        return await asyncio.to_thread(_sync_check)

    async def get_or_create_store(self, store_display_name: str) -> str:
        """Get existing store or create new one."""
        existing_name, exists = await self.check_store_exists(store_display_name)
        if exists and existing_name:
            self.file_search_store_name = existing_name
            return existing_name

        def _sync_create() -> str:
            store = self.client.file_search_stores.create(
                config={"display_name": store_display_name}
            )
            return store.name
        
        store_name = await asyncio.to_thread(_sync_create)
        self.file_search_store_name = store_name
        return store_name

    async def upload_single_file(self, file_path: Path, target_store: str) -> str | None:
        """Upload a single file to the file search store."""
        if not file_path.exists():
            return None

        def _sync_upload() -> str:
            import time
            operation = self.client.file_search_stores.upload_to_file_search_store(
                file=str(file_path),
                file_search_store_name=target_store,
                config={"display_name": file_path.name},
            )
            while not operation.done:
                time.sleep(2)
                operation = self.client.operations.get(operation)
            return file_path.name
        
        return await asyncio.to_thread(_sync_upload)

    async def upload_documents(
        self,
        files: list[Path],
        store_name: str | None = None,
        progress_callback: object | None = None,
    ) -> list[str]:
        """Upload documents to the file search store."""
        target_store = store_name or self.file_search_store_name
        if not target_store:
            raise ValueError("No file search store configured")

        uploaded_files: list[str] = []
        total = len(files)

        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(i + 1, total, file_path.name)

            result = await self.upload_single_file(file_path, target_store)
            if result:
                uploaded_files.append(result)

        return uploaded_files

    async def query_docs(
        self,
        question: str,
        store_name: str | None = None,
        model: str = "gemini-2.5-flash-lite",
    ) -> str:
        """Query the document store with a question.

        Args:
            question: The question to ask
            store_name: Optional store name. Uses instance store if not provided.
            model: Model to use for generation (must support File Search)

        Returns:
            Generated answer based on document context
        """
        target_store = store_name or self.file_search_store_name
        if not target_store:
            raise ValueError("No file search store configured")

        def _sync_query() -> str:
            response = self.client.models.generate_content(
                model=model,
                contents=question,
                config=types.GenerateContentConfig(
                    tools=[
                        types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[target_store]
                            )
                        )
                    ]
                ),
            )
            return response.text or "No answer found in the documents."
        
        return await asyncio.to_thread(_sync_query)

    async def delete_store(
        self, store_name: str | None = None, force: bool = False
    ) -> bool:
        """Delete a file search store."""
        target_store = store_name or self.file_search_store_name
        if not target_store:
            raise ValueError("No file search store to delete")

        def _sync_delete() -> bool:
            self.client.file_search_stores.delete(
                name=target_store,
                config={"force": force},
            )
            return True
        
        result = await asyncio.to_thread(_sync_delete)

        if target_store == self.file_search_store_name:
            self.file_search_store_name = None

        return result
