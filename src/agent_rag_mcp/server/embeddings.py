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
        
        # Ensure model is available
        self._ensure_model()

    def _ensure_model(self) -> None:
        """Ensure the configured model exists in Ollama, pulling if necessary."""
        try:
            # Check if model exists
            models_response = self.client.list()
            # Handle different response structures (list of objects vs dict)
            # models.models keys: 'name', 'model', 'modified_at', etc.
            existing_models = [m.model for m in models_response.models]
            
            # Simple check: exact match or tag match
            # self.model might be "qwen3-embedding:0.6b"
            # existing might be "qwen3-embedding:0.6b" or "qwen3-embedding"
            
            if self.model not in existing_models:
                print(f"📦 Model '{self.model}' not found in Ollama. Pulling now... (This may take a while)")
                # stream=True allows us to see progress if we iterated, but for simplicity we block
                self.client.pull(self.model)
                print(f"✅ Model '{self.model}' pulled successfully.")
            else:
                print(f"✅ Model '{self.model}' is ready.")
                
        except Exception as e:
            print(f"⚠️ Failed to ensure model '{self.model}': {e}")
            print("   Embeddings might fail if model is missing.")

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
