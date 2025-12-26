# weaviate_store.py
"""Weaviate Store for Dynamic Learning RAG."""

import json
from typing import Any, Dict, List

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery

from agent_rag_mcp.core.config import get_config
from agent_rag_mcp.server.embeddings import OllamaClient


class ExperienceStore:
    """Manages 'Experience' data in Weaviate."""

    CLASS_NAME = "Experience"

    def __init__(self) -> None:
        """Initialize Weaviate connection."""
        config = get_config()
        self.client = weaviate.connect_to_local(
            host=config.weaviate_url.split(":")[1].replace("//", ""),
            port=int(config.weaviate_url.split(":")[-1]),
        )
        self.ollama_client = OllamaClient()
        
        try:
            self._ensure_schema()
        except Exception:
            self.client.close()
            raise

    def _ensure_schema(self) -> None:
        """Ensure the Experience schema exists."""
        if not self.client.collections.exists(self.CLASS_NAME):
            self.client.collections.create(
                name=self.CLASS_NAME,
                properties=[
                    Property(name="language", data_type=DataType.TEXT),
                    Property(name="framework", data_type=DataType.TEXT),
                    Property(name="pattern", data_type=DataType.TEXT),
                    Property(name="input_sample", data_type=DataType.TEXT),
                    Property(name="code_result", data_type=DataType.TEXT),
                    Property(name="success", data_type=DataType.BOOL),
                    Property(name="execution_time", data_type=DataType.NUMBER),
                    Property(name="full_json", data_type=DataType.TEXT),  # Store full request
                ],
                # We manage vectors manually via Ollama
                vectorizer_config=Configure.Vectorizer.none(),
            )

    def add_experience(self, request_data: Dict[str, Any]) -> str:
        """Add a new experience to the store.

        Args:
            request_data: Dictionary matching request_schema.toon

        Returns:
            UUID of the created object.
        """
        # Create text to embed - combining important fields
        # This determines what part of the experience allows it to be found again
        embed_text = (
            f"Language: {request_data.get('request', {}).get('language', '')} "
            f"Framework: {request_data.get('request', {}).get('framework', '')} "
            f"Pattern: {request_data.get('request', {}).get('design_context', {}).get('pattern', '')} "
            f"Feature: {request_data.get('request', {}).get('content', {}).get('feature_details', '')}"
        )

        vector = self.ollama_client.get_embedding(embed_text)

        collection = self.client.collections.get(self.CLASS_NAME)
        
        # Flatten structure for querying properties if needed, 
        # but storing full_json is good for retrieval context
        properties = {
            "language": request_data.get("request", {}).get("language", ""),
            "framework": request_data.get("request", {}).get("framework", ""),
            "pattern": request_data.get("request", {}).get("design_context", {}).get("pattern", ""),
            "input_sample": str(request_data.get("request", {}).get("reproduction", {}).get("input_sample", "")),
            "code_result": json.dumps(request_data.get("request", {}).get("content", {}).get("code", {})),
            "success": request_data.get("request", {}).get("content", {}).get("result") == "SUCCESS",
            "execution_time": request_data.get("request", {}).get("metrics", {}).get("execution_time_ms", 0),
            "full_json": json.dumps(request_data),
        }

        uuid_val = collection.data.insert(
            properties=properties,
            vector=vector,
        )
        return str(uuid_val)

    def search_experience(self, query_text: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant experiences.

        Args:
            query_text: Text description of what we are looking for.
            limit: Number of results.

        Returns:
            List of similar experience objects.
        """
        vector = self.ollama_client.get_embedding(query_text)
        collection = self.client.collections.get(self.CLASS_NAME)
        
        response = collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            results.append({
                "properties": obj.properties,
                "distance": obj.metadata.distance,
                # Parse full_json back if needed, or just use properties
                "data": json.loads(obj.properties["full_json"]) if obj.properties.get("full_json") else {}
            })
        
        return results

    def close(self) -> None:
        """Close the client connection."""
        self.client.close()
