# agent_rag_mcp/client/main.py
"""MCP Proxy Client (Production SSE version) - connects to remote MCP server and exposes via stdio."""

import argparse
import os

from fastmcp import Client, FastMCP
from fastmcp.client.transports import SSETransport


def main() -> None:
    """Entry point for the production SSE client."""
    parser = argparse.ArgumentParser(
        description="MCP Proxy Client (SSE) - Connect to remote Agent RAG MCP server",
        prog="agent-rag-client",
    )
    parser.add_argument(
        "--server-url",
        "-s",
        # Default to /sse endpoint for the new transport
        default=os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse"),
        help="Remote MCP server SSE URL (default: http://127.0.0.1:8000/sse)",
    )
    parser.add_argument(
        "--token",
        "-t",
        default=os.getenv("AGENT_RAG_TOKEN"),
        help="Authentication token (optional, can also use AGENT_RAG_TOKEN env var)",
    )

    args = parser.parse_args()

    # Create SSE client for remote server
    # SSETransport handles long-running tool calls much better than StreamableHttpTransport
    transport = SSETransport(args.server_url)
    client = Client(transport, auth=args.token)

    # Create proxy that exposes remote server via stdio (standard MCP connection for AI apps)
    proxy = FastMCP.as_proxy(client)

    # Run proxy with stdio transport
    proxy.run(transport="stdio")


if __name__ == "__main__":
    main()
