# gemini.py
"""Unified Gemini Client for Agent RAG MCP.

Handles both standard content generation and File Search RAG operations.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from google import genai
from google.genai import types

from agent_rag_mcp.core.config import get_config


class GeminiClient:
    """Unified client for Gemini operations."""

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
        self._executor = ThreadPoolExecutor(max_workers=4)

    # ==============================================================================
    # Standard Generation
    # ==============================================================================
    async def generate_content(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash-lite-preview-02-05",  # Using latest efficient model
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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                self._sync_generate,
                prompt=prompt,
                model=model,
                temperature=temperature,
            ),
        )

    def _sync_generate(
        self, prompt: str, model: str, temperature: float
    ) -> str:
        """Synchronous helper for content generation."""
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return response.text or ""

    # ==============================================================================
    # File Search / Document RAG
    # ==============================================================================
    def _sync_find_store(self, store_display_name: str) -> tuple[str | None, bool]:
        """Synchronous helper to find existing store."""
        for store in self.client.file_search_stores.list():
            if store.display_name == store_display_name:
                return store.name, True
        return None, False

    def _sync_create_store(self, store_display_name: str) -> str:
        """Synchronous helper to create a new store."""
        store = self.client.file_search_stores.create(
            config={"display_name": store_display_name}
        )
        return store.name

    async def check_store_exists(self, store_display_name: str) -> tuple[str | None, bool]:
        """Check if a store with the given name already exists."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(self._sync_find_store, store_display_name),
        )

    async def get_or_create_store(self, store_display_name: str) -> str:
        """Get existing store or create new one."""
        existing_name, exists = await self.check_store_exists(store_display_name)
        if exists and existing_name:
            self.file_search_store_name = existing_name
            return existing_name

        loop = asyncio.get_event_loop()
        store_name = await loop.run_in_executor(
            self._executor,
            partial(self._sync_create_store, store_display_name),
        )
        self.file_search_store_name = store_name
        return store_name

    def _sync_upload_file(self, file_path: Path, target_store: str) -> str | None:
        """Synchronous helper to upload a single file."""
        if not file_path.exists():
            return None

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

        loop = asyncio.get_event_loop()
        uploaded_files: list[str] = []
        total = len(files)

        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(i + 1, total, file_path.name)

            result = await loop.run_in_executor(
                self._executor,
                partial(self._sync_upload_file, file_path, target_store),
            )
            if result:
                uploaded_files.append(result)

        return uploaded_files

    def _sync_query_docs(
        self,
        question: str,
        target_store: str,
        model: str,
    ) -> str:
        """Synchronous helper for document query."""
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

    async def query_docs(
        self,
        question: str,
        store_name: str | None = None,
        model: str = "gemini-2.5-flash",
    ) -> str:
        """Query the document store with a question.

        Args:
            question: The question to ask
            store_name: Optional store name. Uses instance store if not provided.
            model: Model to use for generation (default: gemini-2.5-flash for RAG)

        Returns:
            Generated answer based on document context
        """
        target_store = store_name or self.file_search_store_name
        if not target_store:
            raise ValueError("No file search store configured")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(self._sync_query_docs, question, target_store, model),
        )

    def _sync_delete_store(self, target_store: str, force: bool) -> bool:
        """Synchronous helper for delete."""
        self.client.file_search_stores.delete(
            name=target_store,
            config={"force": force},
        )
        return True

    async def delete_store(
        self, store_name: str | None = None, force: bool = False
    ) -> bool:
        """Delete a file search store."""
        target_store = store_name or self.file_search_store_name
        if not target_store:
            raise ValueError("No file search store to delete")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            partial(self._sync_delete_store, target_store, force),
        )

        if target_store == self.file_search_store_name:
            self.file_search_store_name = None

        return result
