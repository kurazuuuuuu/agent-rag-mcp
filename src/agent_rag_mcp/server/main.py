# server.py
"""FastMCP server for Agent RAG MCP."""

import asyncio
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlparse

from fastmcp import FastMCP
from fastmcp.server.auth.providers.debug import DebugTokenVerifier

from agent_rag_mcp.core.config import get_config
from agent_rag_mcp.server.gemini import GeminiClient
from agent_rag_mcp.server.weaviate_store import ExperienceStore
import json

# Supported file extensions for documentation
SUPPORTED_EXTENSIONS = ["*.md", "*.txt", "*.rst", "*.json", "*.yaml", "*.yml"]


# ==============================================================================
# Authentication Provider
# ==============================================================================
def get_auth_provider() -> DebugTokenVerifier | None:
    """Get authentication provider if AUTH_TOKEN is configured.

    Returns:
        DebugTokenVerifier if AUTH_TOKEN is set, None otherwise.
    """
    config = get_config()
    if not config.auth_token:
        return None

    # Create a simple token validator that checks against the configured token
    def validate_token(token: str) -> bool:
        return token == config.auth_token

    return DebugTokenVerifier(
        validate=validate_token,
        client_id="agent-rag-client",
    )


# ==============================================================================
# Helper Functions
# ==============================================================================
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


async def init_store_from_repo(
    client: GeminiClient,
    repo_url: str,
    docs_path: str,
    branch: str,
    store_name: str | None,
) -> tuple[str, str, list[str]]:
    """Clone a repository and upload documentation to the RAG store.

    Returns:
        Tuple of (display_name, store_id, uploaded_files)
    """
    display_name = store_name or generate_store_name_from_url(repo_url)
    store_id = await client.get_or_create_store(display_name)

    temp_dir = tempfile.mkdtemp(prefix="agent-rag-")

    try:
        # Clone repository using async subprocess
        process = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", "--branch", branch, repo_url, temp_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
        except asyncio.TimeoutError:
            process.kill()
            raise RuntimeError("Git clone timed out after 120 seconds")

        if process.returncode != 0:
            raise RuntimeError(f"Git clone failed: {stderr.decode()}")

        # Find docs directory
        docs_full_path = Path(temp_dir) / docs_path
        if not docs_full_path.exists():
            raise FileNotFoundError(f"Docs path not found: {docs_path}")

        # Collect all documentation files
        files_to_upload: list[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            files_to_upload.extend(docs_full_path.rglob(ext))

        if not files_to_upload:
            raise FileNotFoundError(f"No documentation files found in {docs_path}")

        # Progress callback
        def progress(current: int, total: int, filename: str) -> None:
            print(f"   📄 [{current}/{total}] {filename}")

        # Upload files
        print(f"   Uploading {len(files_to_upload)} files...")
        uploaded = await client.upload_documents(
            files_to_upload, store_name=store_id, progress_callback=progress
        )

        return display_name, store_id, uploaded

    finally:
        # Clean up temporary directory
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


async def init_store_from_local(
    client: GeminiClient,
    local_docs_path: str,
    store_name: str | None,
) -> tuple[str, str, list[str]]:
    """Initialize store from local documentation directory.

    Returns:
        Tuple of (display_name, store_id, uploaded_files)
    """
    display_name = store_name or generate_store_name_from_path(local_docs_path)
    store_id = await client.get_or_create_store(display_name)

    docs_path = Path(local_docs_path)
    if not docs_path.exists():
        raise FileNotFoundError(f"Directory not found: {local_docs_path}")

    # Collect all documentation files
    files_to_upload: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files_to_upload.extend(docs_path.rglob(ext))

    if not files_to_upload:
        raise FileNotFoundError(f"No documentation files found in {local_docs_path}")

    # Progress callback
    def progress(current: int, total: int, filename: str) -> None:
        print(f"   📄 [{current}/{total}] {filename}")

    # Upload files
    print(f"   Uploading {len(files_to_upload)} files...")
    uploaded = await client.upload_documents(
        files_to_upload, store_name=store_id, progress_callback=progress
    )

    return display_name, store_id, uploaded


# ==============================================================================
# Server State (populated during startup)
# ==============================================================================
class ServerState:
    """Global server state, initialized during lifespan startup."""

    rag_client: GeminiClient | None = None
    experience_store: ExperienceStore | None = None
    store_name: str | None = None
    store_id: str | None = None


_state = ServerState()


# ==============================================================================
# Lifespan: Initialize Document Store on Startup
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastMCP) -> AsyncIterator[None]:
    """Initialize the document store when the server starts."""
    config = get_config()

    # Show auth status
    if config.is_auth_enabled:
        print("🔐 Authentication enabled (AUTH_TOKEN is set)")
    else:
        print("⚠️  Authentication disabled (set AUTH_TOKEN to enable)")

    # Initialize Experience Store (Weaviate)
    try:
        print("🧠 Initializing Experience Store (Weaviate + Ollama)...")
        _state.experience_store = ExperienceStore()
        print(f"   Connected to Weaviate at {config.weaviate_url}")
    except Exception as e:
        print(f"❌ Failed to connect to Experience Store: {e}")
        print("   Dynamic Learning features will be unavailable.")

    # Check if we have required configuration for Doc RAG
    if not config.has_document_source:
        print("⚠️  No document source configured.")
        print("   Set RAG_REPO_URL or RAG_LOCAL_DOCS_PATH environment variable.")
        print("   Server will start without document store.")
        yield
        return

    # Initialize RAG client
    print("🔧 Initializing Gemini RAG client...")
    _state.rag_client = GeminiClient()

    # Determine store name
    if config.rag_store_name:
        display_name = config.rag_store_name
    elif config.rag_repo_url:
        display_name = generate_store_name_from_url(config.rag_repo_url)
    else:
        display_name = generate_store_name_from_path(config.rag_local_docs_path)

    try:
        # Check if store already exists (to avoid re-indexing costs)
        existing_store, exists = await _state.rag_client.check_store_exists(display_name)

        if exists and existing_store and not config.rag_force_reindex:
            # Use existing store - no upload needed!
            print(f"✅ Found existing document store '{display_name}'")
            print("   Skipping upload (set RAG_FORCE_REINDEX=true to re-index)")
            _state.store_name = display_name
            _state.store_id = existing_store
        else:
            # Need to upload documents
            if config.rag_force_reindex and exists:
                print(f"🔄 Force re-indexing '{display_name}' (RAG_FORCE_REINDEX=true)")

            if config.rag_repo_url:
                # Initialize from git repository
                print(f"📦 Cloning repository: {config.rag_repo_url}")
                print(f"   Branch: {config.rag_branch}, Docs path: {config.rag_docs_path}")

                display_name, store_id, uploaded = await init_store_from_repo(
                    _state.rag_client,
                    config.rag_repo_url,
                    config.rag_docs_path,
                    config.rag_branch,
                    config.rag_store_name,
                )
            else:
                # Initialize from local path
                print(f"📂 Loading local docs: {config.rag_local_docs_path}")

                display_name, store_id, uploaded = await init_store_from_local(
                    _state.rag_client,
                    config.rag_local_docs_path,
                    config.rag_store_name,
                )

            _state.store_name = display_name
            _state.store_id = store_id

            print(f"✅ Document store '{display_name}' ready!")
            print(f"   Indexed {len(uploaded)} files")

    except Exception as e:
        print(f"❌ Failed to initialize document store: {e}")
        print("   Server will start without document store.")
        _state.rag_client = None

    yield

    # Cleanup on shutdown
    print("👋 Server shutting down...")
    if _state.experience_store:
        _state.experience_store.close()


# ==============================================================================
# FastMCP Server
# ==============================================================================
# Get auth provider (None if AUTH_TOKEN not set)
_auth_provider = get_auth_provider()

mcp = FastMCP(
    name="AgentRAG-MCP",
    instructions="""
        This server provides "Retrieval-Augmented Generation" tools for AI agents.
        It enables you to:
        - Ask questions about project documentation

        Use the ask_project_document tool to get answers based on the indexed documentation.
        The server uses semantic search to find relevant information and generates
        accurate answers grounded in the actual documents.

        Use the ask_code_pattern tool to learn from previous coding experiences and 
        retrieve dynamic patterns.
    """,
    lifespan=lifespan,
    auth=_auth_provider,
)


# ==============================================================================
# MCP Tools
# ==============================================================================
@mcp.tool
async def ask_project_document(question: str) -> str:
    """Ask questions about the project documentation.

    Use this tool to get answers based on the project's documentation.
    The server uses semantic search to find relevant information and
    generates accurate answers grounded in the actual documents.

    Args:
        question: Your question about the project documentation.
                  Be specific for better results.

    Returns:
        Answer generated from the project documentation with citations.
    """
    if _state.rag_client is None or _state.store_id is None:
        return (
            "Error: Document store is not initialized. "
            "Please configure RAG_REPO_URL or RAG_LOCAL_DOCS_PATH environment variable "
            "and restart the server."
        )

    # Query the documents
    try:
        answer = await asyncio.wait_for(
            _state.rag_client.query_docs(question, store_name=_state.store_id),
            timeout=60.0,
        )
        return answer
    except asyncio.TimeoutError:
        return "Error: Processing timed out (60s limit). Please try a more specific query."


@mcp.tool
async def get_store_info() -> str:
    """Get information about the current document store.

    Returns:
        Information about the initialized document store.
    """
    if _state.store_name is None:
        return "No document store is currently initialized."

    return f"Document Store: {_state.store_name}\nStore ID: {_state.store_id}"


@mcp.tool
async def get_request_schema_template() -> str:
    """Get the schema template for code pattern requests.

    Returns:
        The content of schema/request_schema.toon template.
    """
    schema_path = Path("schema/request_schema.toon")
    if not schema_path.exists():
        # Try finding it relative to module if cwd is different
        # Assuming src layout: src/agent_rag_mcp/server.py -> ../../../schema
        alt_path = Path(__file__).parent.parent.parent / "schema" / "request_schema.toon"
        if alt_path.exists():
            schema_path = alt_path
        else:
            return "Error: Schema template file not found."

    try:
        return schema_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading schema file: {e}"


@mcp.tool
async def ask_code_pattern(request_data: str) -> str:
    """Ask for code patterns or share an experience to learn from.

    This tool enables Dynamic Learning RAG.
    - If you provide a result/code in your request, it will be learned (stored).
    - It always searches for similar past experiences to provide context.
    - It uses Gemini to analyze the request and retrieved patterns to give advice.

    Args:
        request_data: JSON string matching the request_schema.toon structure.
                      Must include 'request' key.

    Returns:
        Analysis and relevant patterns from the dynamic knowledge base.
    """
    if _state.experience_store is None:
        return "Error: Experience Store is not available (Weaviate connection failed)."
    
    if _state.rag_client is None:
        return "Error: Gemini Client is not available."

    try:
        data = json.loads(request_data)
    except json.JSONDecodeError:
        return "Error: Invalid JSON format. Please ensure request_data is a valid JSON string."

    # extraction for search
    req_body = data.get("request", {})
    query_text = (
        f"Language: {req_body.get('language', '')} "
        f"Framework: {req_body.get('framework', '')} "
        f"Pattern: {req_body.get('design_context', {}).get('pattern', '')} "
        f"Feature: {req_body.get('content', {}).get('feature_details', '')} "
        f"Input: {req_body.get('reproduction', {}).get('input_sample', '')}"
    )

    # 1. Search for existing experiences
    # We always search to provide context
    similar_exps = _state.experience_store.search_experience(query_text, limit=3)
    
    context_str = "Found similar past experiences:\n"
    for i, exp in enumerate(similar_exps):
        props = exp.get("properties", {})
        context_str += (
            f"\n--- Experience {i+1} (Distance: {exp.get('distance', 'N/A')}) ---\n"
            f"Language: {props.get('language')}\n"
            f"Pattern: {props.get('pattern')}\n"
            f"Code Result: {props.get('code_result')}\n"
            f"Success: {props.get('success')}\n"
        )

    # 2. Reasoning with Gemini
    prompt = (
        f"You are an expert software engineer assistant.\n"
        f"Analyze the following request and the provided past experiences.\n"
        f"If the user is reporting a success/failure, verify it and summarize the learning.\n"
        f"If the user is asking for help, use the past experiences to provide the best code pattern.\n\n"
        f"CURRENT REQUEST:\n{json.dumps(req_body, indent=2)}\n\n"
        f"PAST EXPERIENCES:\n{context_str}\n\n"
        f"Provide a helpful response, including code examples if applicable."
    )

    try:
        response = await asyncio.wait_for(
            _state.rag_client.generate_content(prompt),
            timeout=60.0
        )
    except asyncio.TimeoutError:
        return "Error: Processing timed out (60s limit). Please simplify your request."

    # 3. Learning (Store) if this is a report with a result
    learning_msg = ""
    result_val = req_body.get("content", {}).get("result")
    code_val = req_body.get("content", {}).get("code")
    
    # Simple heuristic: if we have a result status (SUCCESS/FAILED) or code content, we treat it as an experience to learn
    if result_val or code_val:
        try:
            uuid_id = _state.experience_store.add_experience(data)
            learning_msg = f"\n\n[System] New experience learned/recorded! (ID: {uuid_id})"
        except Exception as e:
            learning_msg = f"\n\n[System] Failed to record experience: {e}"

    return response + learning_msg