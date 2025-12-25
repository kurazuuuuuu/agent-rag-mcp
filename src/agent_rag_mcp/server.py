# server.py
"""FastMCP server for Agent RAG MCP."""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastmcp import FastMCP

from .gemini_rag import GeminiRAGClient

# Load environment variables from .env file (override existing)
load_dotenv(override=True)

mcp = FastMCP(
    name="AgentRAG-MCP",
    instructions="""
        This server provides "Retrieval-Augmented Generation" tools for AI agents.
        It enables you to:
        - Ask questions about project documentation
        - Learn from code patterns (success/error)
        - Query best practices for implementation

        The server supports continuous learning. Share your experiences!
    """,
)

# Initialize Gemini RAG client (lazy loading)
_rag_client: GeminiRAGClient | None = None

# Store name -> store ID mapping
_store_cache: dict[str, str] = {}

# Supported file extensions for documentation
SUPPORTED_EXTENSIONS = ["*.md", "*.txt", "*.rst", "*.json", "*.yaml", "*.yml"]


def get_rag_client() -> GeminiRAGClient:
    """Get or initialize the RAG client."""
    global _rag_client
    if _rag_client is None:
        _rag_client = GeminiRAGClient()
    return _rag_client


def generate_store_name_from_url(repo_url: str) -> str:
    """Generate a store name from a repository URL.

    Examples:
        https://github.com/Krz-Tech/minecraft-project -> krz-tech-minecraft-project
        git@github.com:user/repo.git -> user-repo
    """
    # Remove .git suffix
    url = repo_url.rstrip("/").removesuffix(".git")

    # Handle SSH URLs
    if url.startswith("git@"):
        # git@github.com:user/repo -> user/repo
        url = url.split(":")[-1]
    else:
        # Parse HTTP URLs
        parsed = urlparse(url)
        url = parsed.path.lstrip("/")

    # Convert to lowercase and replace slashes with dashes
    store_name = url.lower().replace("/", "-")

    # Remove any invalid characters (keep only alphanumeric and dashes)
    store_name = re.sub(r"[^a-z0-9-]", "", store_name)

    # Ensure it doesn't start or end with dashes
    store_name = store_name.strip("-")

    return store_name or "unknown-repo"


def generate_store_name_from_path(local_path: str) -> str:
    """Generate a store name from a local path.

    Examples:
        /path/to/minecraft-project/Docs -> minecraft-project
        ./my_project/docs -> my-project
    """
    path = Path(local_path).resolve()

    # Use parent directory name if path ends with common doc folder names
    doc_folders = {"docs", "doc", "documentation", "wiki"}
    if path.name.lower() in doc_folders and path.parent.name:
        name = path.parent.name
    else:
        name = path.name

    # Convert to lowercase and replace underscores with dashes
    store_name = name.lower().replace("_", "-")

    # Remove any invalid characters
    store_name = re.sub(r"[^a-z0-9-]", "", store_name)

    return store_name or "local-docs"


@mcp.tool
async def ask_project_document(question: str, store_name: str | None = None) -> str:
    """Ask questions about the project documentation.

    Use this tool to get answers based on the project's documentation.
    The server uses semantic search to find relevant information and
    generates accurate answers grounded in the actual documents.

    Args:
        question: Your question about the project documentation.
                  Be specific for better results.
        store_name: Optional store name to query. If not provided,
                    uses the most recently created store.

    Returns:
        Answer generated from the project documentation with citations.
    """
    client = get_rag_client()

    # Use provided store name or get from cache
    if store_name and store_name in _store_cache:
        target_store = _store_cache[store_name]
    elif _store_cache:
        # Use the most recently added store
        target_store = list(_store_cache.values())[-1]
    else:
        return "Error: No document store has been set up. Use setup_document_store_from_repo first."

    # Query the documents
    answer = await client.query(question, store_name=target_store)
    return answer


@mcp.tool
async def list_document_stores() -> str:
    """List all available document stores.

    Returns:
        List of store names that have been set up in this session.
    """
    if not _store_cache:
        return "No document stores have been set up yet."

    stores = "\n".join(f"- {name}" for name in _store_cache)
    return f"Available document stores:\n{stores}"


@mcp.tool
async def setup_document_store_from_repo(
    repo_url: str,
    docs_path: str = "Docs",
    branch: str = "main",
    store_name: str | None = None,
) -> str:
    """Clone a repository and upload documentation to the RAG store.

    This tool clones a Git repository to a temporary directory,
    uploads documentation files, and then cleans up the temporary files.
    The store name is automatically generated from the repository URL.

    Args:
        repo_url: Git repository URL (e.g., "https://github.com/user/repo")
        docs_path: Path to docs directory within the repo (default: "Docs")
        branch: Git branch to clone (default: "main")
        store_name: Optional custom store name. If not provided,
                    generated from repo URL.

    Returns:
        Status message with list of uploaded files
    """
    client = get_rag_client()

    # Generate or use provided store name
    display_name = store_name or generate_store_name_from_url(repo_url)

    # Get or create the store
    store_id = await client.get_or_create_store(display_name)
    _store_cache[display_name] = store_id

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="agent-rag-")

    try:
        # Clone repository
        clone_result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, repo_url, temp_dir],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if clone_result.returncode != 0:
            return f"Error cloning repository: {clone_result.stderr}"

        # Find docs directory
        docs_full_path = Path(temp_dir) / docs_path
        if not docs_full_path.exists():
            return f"Error: Docs path not found: {docs_path}"

        # Collect all documentation files
        files_to_upload: list[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            files_to_upload.extend(docs_full_path.rglob(ext))

        if not files_to_upload:
            return f"No documentation files found in {docs_path}"

        # Upload files
        uploaded = await client.upload_documents(files_to_upload, store_name=store_id)

        return (
            f"Successfully created store '{display_name}' and uploaded {len(uploaded)} files:\n"
            + "\n".join(f"- {f}" for f in uploaded)
        )

    except subprocess.TimeoutExpired:
        return "Error: Git clone timed out after 120 seconds"
    except Exception as e:
        return f"Error: {e!s}"
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@mcp.tool
async def setup_document_store(
    repo_docs_path: str,
    store_name: str | None = None,
) -> str:
    """Initialize or update the document store with local documentation.

    This tool sets up the RAG store by uploading documentation files
    from a local directory. The store name is automatically generated
    from the directory path.

    Args:
        repo_docs_path: Path to the docs directory (e.g., "/path/to/repo/Docs")
        store_name: Optional custom store name. If not provided,
                    generated from directory path.

    Returns:
        Status message with list of uploaded files
    """
    client = get_rag_client()

    # Generate or use provided store name
    display_name = store_name or generate_store_name_from_path(repo_docs_path)

    # Get or create the store
    store_id = await client.get_or_create_store(display_name)
    _store_cache[display_name] = store_id

    # Find all markdown files in the docs directory
    docs_path = Path(repo_docs_path)
    if not docs_path.exists():
        return f"Error: Directory not found: {repo_docs_path}"

    # Collect all documentation files
    files_to_upload: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files_to_upload.extend(docs_path.rglob(ext))

    if not files_to_upload:
        return f"No documentation files found in {repo_docs_path}"

    # Upload files
    uploaded = await client.upload_documents(files_to_upload, store_name=store_id)

    return (
        f"Successfully created store '{display_name}' and uploaded {len(uploaded)} files:\n"
        + "\n".join(f"- {f}" for f in uploaded)
    )