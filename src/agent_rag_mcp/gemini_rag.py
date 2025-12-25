# gemini_rag.py
"""Gemini File Search RAG client for document queries."""

import os
import time
from pathlib import Path

from google import genai
from google.genai import types


class GeminiRAGClient:
    """Client for Gemini File Search RAG operations."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the Gemini client.

        Args:
            api_key: Optional API key. If not provided, uses GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        self.client = genai.Client(api_key=self.api_key)
        self.file_search_store_name: str | None = None

    async def get_or_create_store(self, store_display_name: str) -> str:
        """Get existing store or create new one.

        Args:
            store_display_name: Display name for the store

        Returns:
            The file search store name
        """
        # Check if store already exists
        for store in self.client.file_search_stores.list():
            if store.display_name == store_display_name:
                self.file_search_store_name = store.name
                return store.name

        # Create new store
        store = self.client.file_search_stores.create(
            config={"display_name": store_display_name}
        )
        self.file_search_store_name = store.name
        return store.name

    async def upload_documents(
        self,
        files: list[Path],
        store_name: str | None = None,
    ) -> list[str]:
        """Upload documents to the file search store.

        Args:
            files: List of file paths to upload
            store_name: Optional store name. Uses instance store if not provided.

        Returns:
            List of uploaded file names
        """
        target_store = store_name or self.file_search_store_name
        if not target_store:
            raise ValueError("No file search store configured")

        uploaded_files: list[str] = []
        for file_path in files:
            if not file_path.exists():
                continue

            operation = self.client.file_search_stores.upload_to_file_search_store(
                file=str(file_path),
                file_search_store_name=target_store,
                config={"display_name": file_path.name},
            )

            # Wait for upload to complete
            while not operation.done:
                time.sleep(2)
                operation = self.client.operations.get(operation)

            uploaded_files.append(file_path.name)

        return uploaded_files

    async def query(
        self,
        question: str,
        store_name: str | None = None,
        model: str = "gemini-2.5-flash",
    ) -> str:
        """Query the document store with a question.

        Args:
            question: The question to ask
            store_name: Optional store name. Uses instance store if not provided.
            model: Model to use for generation

        Returns:
            Generated answer based on document context
        """
        target_store = store_name or self.file_search_store_name
        if not target_store:
            raise ValueError("No file search store configured")

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

    async def list_store_files(self, store_name: str | None = None) -> list[str]:
        """List files in the store.

        Args:
            store_name: Optional store name. Uses instance store if not provided.

        Returns:
            List of file display names in the store
        """
        # Note: The API may not have a direct method to list files in a store
        # This is a placeholder for future implementation
        return []

    async def delete_store(self, store_name: str | None = None, force: bool = False) -> bool:
        """Delete a file search store.

        Args:
            store_name: Optional store name. Uses instance store if not provided.
            force: If True, delete even if store contains files

        Returns:
            True if deletion was successful
        """
        target_store = store_name or self.file_search_store_name
        if not target_store:
            raise ValueError("No file search store to delete")

        self.client.file_search_stores.delete(
            name=target_store,
            config={"force": force},
        )

        if target_store == self.file_search_store_name:
            self.file_search_store_name = None

        return True
