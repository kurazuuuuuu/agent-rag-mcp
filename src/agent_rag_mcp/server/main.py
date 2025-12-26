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
import yaml

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
    try:
        print("🔧 Initializing Gemini RAG client...")
        _state.rag_client = GeminiClient()
        print("   Gemini RAG client initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini RAG client: {e}")
        print("   Document RAG features will be unavailable.")
        _state.rag_client = None
        yield
        return

    # Determine store name
    if config.rag_store_name:
        display_name = config.rag_store_name
    elif config.rag_repo_url:
        display_name = generate_store_name_from_url(config.rag_repo_url)
    else:
        display_name = generate_store_name_from_path(config.rag_local_docs_path)

    try:
        # Check if store already exists (to avoid re-indexing costs)
        print(f"🔍 Checking for existing document store '{display_name}'...")
        existing_store, exists = await _state.rag_client.check_store_exists(display_name)

        if exists and existing_store and not config.rag_force_reindex:
            # Use existing store - no upload needed!
            print(f"✅ Found existing document store '{display_name}' ({existing_store})")
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
        import traceback
        traceback.print_exc()
        print("   Server will start without document store.")

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
        このサーバーは、AIエージェントのための「RAG（検索拡張生成）」ツールを提供します。
        以下の2つの主要な機能を使い分けてください：

        1. **ask_project_document**: プロジェクトのドキュメント検索
           - 仕様書、設計書、READMEなどの「静的なドキュメント」について質問する際に使用します。
           - 例: 「認証機能の仕様は？」「データベースのスキーマ構造は？」

        2. **ask_code_pattern**: コーディング経験の検索と学習（Dynamic Learning）
           - 過去の実装パターン、成功例、失敗談などの「経験則」を知りたい場合に使用します。
           - また、あなたの実装結果を送信することで、システムに新しい知識を学習させることができます。
           - 入力は必ず指定されたJSONスキーマに従ってください。
    """,
    lifespan=lifespan,
    auth=_auth_provider,
)


@mcp.tool
async def ask_project_document(question: str) -> str:
    """プロジェクトのドキュメント（仕様書・設計書など）について質問します。

    プロジェクトの「静的な仕様」や「設計の背景」を知りたい場合に使用してください。
    サーバーは、インデックス化されたドキュメントから関連情報を検索し、
    事実に基づいた回答を生成します。

    Args:
        question: ドキュメントに対する質問内容。
                  より具体的な結果を得るために、詳細に記述してください。

    Returns:
        ドキュメントの内容に基づいた回答（引用付き）。
    """
    if _state.rag_client is None or _state.store_id is None:
        return (
            "Error: Document store is not initialized. "
            "Please configure RAG_REPO_URL or RAG_LOCAL_DOCS_PATH environment variable "
            "and restart the server."
        )

    target_store = _state.store_id
    model = "gemini-2.5-flash"
    
    # Use direct query_docs (via asyncio.to_thread)
    # The SSE transport handles long-running calls better than streamable-http
    try:
        answer = await _state.rag_client.query_docs(question, store_name=target_store, model=model)
        return answer
    except Exception as e:
        return f"Error: Failed to execute RAG query: {str(e)}"


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
async def test_large_response() -> str:
    """Test tool that returns a large response string.
    
    Used for debugging response size issues.
    """
    # Generate a large string (about 3000 chars)
    large_text = "This is a test response. " * 150
    return f"Large response test ({len(large_text)} chars):\n{large_text}"


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


def _parse_code_request(request_data: str) -> dict | None:
    """Helper to parse TOON/YAML or JSON request data."""
    data = None
    # 1. Try parsing as TOON (YAML-compatible)
    try:
        parsed = yaml.safe_load(request_data)
        if isinstance(parsed, dict):
            data = parsed
    except Exception:
        pass

    # 2. Try parsing as JSON
    if data is None:
        if isinstance(request_data, dict):
            data = request_data
        else:
            try:
                data = json.loads(request_data)
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        pass
            except (json.JSONDecodeError, TypeError):
                pass
    return data


@mcp.tool
async def ask_code_pattern(request_data: str) -> str:
    """コーディングの「経験則（パターン・成功/失敗例）」を検索し、最適な実装を提案します。

    Dynamic Learning RAG を使用して、過去の成功例や失敗談からアドバイスを生成します。
    - 新しい機能を実装する前に、最適なパターンを知りたい場合に使用してください。
    - 特定のライブラリやフレームワークの「ハマりポイント」を確認するのに役立ちます。

    Args:
        request_data: `request_schema.toon` の構造（requestキー必須）に従ったデータ文字列。
                      TOONフォーマットの使用を推奨します。

    Returns:
        過去の類似事例に基づいた分析と実装アドバイス。
    """
    if _state.experience_store is None or _state.rag_client is None:
        return "Error: Experience Store or Gemini Client is not available."

    data = _parse_code_request(request_data)
    if not isinstance(data, dict):
        return "Error: Invalid data format. Please provide valid TOON or JSON string."

    req_body = data.get("request", {})
    query_text = (
        f"Language: {req_body.get('language', '')} "
        f"Framework: {req_body.get('framework', '')} "
        f"Pattern: {req_body.get('design_context', {}).get('pattern', '')} "
        f"Feature: {req_body.get('content', {}).get('feature_details', '')}"
    )

    # 1. Search for existing experiences
    similar_exps = _state.experience_store.search_experience(query_text, limit=3)
    
    context_str = "Found similar past experiences:\n"
    for i, exp in enumerate(similar_exps):
        props = exp.get("properties", {})
        context_str += (
            f"\n--- Experience {i+1} ---\n"
            f"Language: {props.get('language')}\n"
            f"Pattern: {props.get('pattern')}\n"
            f"Success: {props.get('success')}\n"
            f"Code/Result: {props.get('code_result')}\n"
        )

    # 2. Reasoning with Gemini
    prompt = (
        f"あなたは世界最高のソフトウェアエンジニアです。\n"
        f"ユーザーが実装に関するコードパターンやアドバイスを求めています。\n"
        f"以下の過去の経験（成功例・失敗例）を参考に、最適な回答を【日本語】で提供してください。\n\n"
        f"リクエスト内容:\n{json.dumps(req_body, indent=2, ensure_ascii=False)}\n\n"
        f"過去の類似経験:\n{context_str}\n\n"
        f"回答は具体的かつ丁寧に行い、必要に応じてコード例やベストプラクティスを含めてください。"
    )

    return await _state.rag_client.generate_content(prompt)


@mcp.tool
async def tell_code_pattern(request_data: str) -> str:
    """実装したコードの結果（成功・失敗・エラー）を報告し、システムに学習させます。

    あなたが経験した「成功した実装」や「発生したエラー」を記録することで、
    次回以降の `ask_code_pattern` で自分自身や他のエージェントが同じ轍を踏まないようにします。

    Args:
        request_data: `request_schema.toon` の構造に従ったデータ文字列。
                      'content' 内に 'result' (SUCCESS/FAILED) を含める必要があります。
                      エラーが発生した場合は、エラーログや再現手順を含めてください。

    Returns:
        学習完了のメッセージ。エラー報告の場合は、過去の知見に基づいた改善案も提示されます。
    """
    if _state.experience_store is None or _state.rag_client is None:
        return "Error: Experience Store or Gemini Client is not available."

    data = _parse_code_request(request_data)
    if not isinstance(data, dict):
        return "Error: Invalid data format. Please provide valid TOON or JSON string."

    # 1. Record the experience
    try:
        uuid_id = _state.experience_store.add_experience(data)
        learning_msg = f"[System] 経験を学習しました。 (ID: {uuid_id})\n\n"
    except Exception as e:
        return f"Error recording experience: {e}"

    req_body = data.get("request", {})
    result_val = req_body.get("content", {}).get("result")

    # 2. If it's a failure, provide immediate advice based on search
    if result_val == "FAILED":
        query_text = (
            f"Error in {req_body.get('language')} {req_body.get('framework')}: "
            f"{req_body.get('content', {}).get('feature_details', '')}"
        )
        similar_exps = _state.experience_store.search_experience(query_text, limit=3)
        
        context_str = ""
        for i, exp in enumerate(similar_exps):
            if exp.get("properties", {}).get("success"):
                props = exp.get("properties", {})
                context_str += f"\n- Successful Pattern {i+1}: {props.get('pattern')}\nCode: {props.get('code_result')}\n"

        if context_str:
            prompt = (
                f"ユーザーの実装が失敗しました。以下の成功事例を参考に、修正案を【日本語】で提案してください。\n"
                f"失敗したリクエスト:\n{json.dumps(req_body, indent=2, ensure_ascii=False)}\n\n"
                f"参考にすべき成功例:\n{context_str}"
            )
            advice = await _state.rag_client.generate_content(prompt)
            return learning_msg + "### 過去の成功事例に基づく改善案:\n" + advice
        else:
            return learning_msg + "過去に類似の成功事例は見つかりませんでした。この失敗は将来の参照のために記録されました。"

    return learning_msg + "素晴らしい！この成功体験は将来の実装アドバイスに反映されます。"