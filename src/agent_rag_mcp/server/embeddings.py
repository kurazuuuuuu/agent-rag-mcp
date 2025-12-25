# embeddings.py
"""Ollama Client Wrapper for Embeddings."""

import ollama
from agent_rag_mcp.core.config import get_config


class OllamaClient:
    """Wrapper for Ollama operations (specifically embeddings)."""

    def __init__(self) -> None:
        """Initialize the Ollama client."""
        config = get_config()
        self.host = config.ollama_host
        self.model = config.ollama_model
        
        # Configure the ollama client host
        # The ollama python client uses OLLAMA_HOST env var by default, 
        # but we can also set the client explicitly if needed.
        # However, the python library acts as a global client mostly.
        # We will assume environment variable OLLAMA_HOST is handled or we set it?
        # The ollama library reads OLLAMA_HOST env var. 
        # Since we use python-dotenv, it should be set if in .env, 
        # but we also read from config.
        
        # We'll use the Client instance to be safe and explicit
        self.client = ollama.Client(host=self.host)

    def get_embedding(self, text: str) -> list[float]:
        """Get vector embedding for a text string.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the vector.
        """
        response = self.client.embeddings(model=self.model, prompt=text)
        # Verify valid response
        if "embedding" not in response:
            raise ValueError(f"Failed to get embedding from Ollama: {response}")
        
        return response["embedding"]
