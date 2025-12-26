# agent_rag_mcp/client/main_http.py
"""MCP Proxy Client (HTTP version) - archived."""

import argparse
import os

from fastmcp import Client, FastMCP
from fastmcp.client import StreamableHttpTransport


def main() -> None:
    """Entry point for the archived HTTP client."""
    parser = argparse.ArgumentParser(
        description="MCP Proxy Client (HTTP) - Archived",
        prog="agent-rag-client-http",
    )
    parser.add_argument(
        "--server-url",
        "-s",
        default=os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp"),
        help="Remote MCP server URL (default: http://127.0.0.1:8000/mcp)",
    )
    parser.add_argument(
        "--token",
        "-t",
        default=os.getenv("AGENT_RAG_TOKEN"),
        help="Authentication token (optional)",
    )

    args = parser.parse_args()

    # Create client for remote server
    transport = StreamableHttpTransport(args.server_url)
    client = Client(transport, auth=args.token)

    # Create proxy that exposes remote server via stdio
    proxy = FastMCP.as_proxy(client)

    # Run proxy with stdio transport
    proxy.run(transport="stdio")


if __name__ == "__main__":
    main()
