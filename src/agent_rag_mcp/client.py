# agent_rag_mcp/client.py
"""MCP Proxy Client - connects to remote MCP server and exposes via stdio."""

import argparse
import os

from fastmcp import Client, FastMCP


def main() -> None:
    """Entry point for the agent-rag-client command."""
    parser = argparse.ArgumentParser(
        description="MCP Proxy Client - Connect to remote Agent RAG MCP server",
        prog="agent-rag-client",
    )
    parser.add_argument(
        "--server-url",
        "-s",
        default=os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse"),
        help="Remote MCP server URL (default: http://127.0.0.1:8000/sse)",
    )
    parser.add_argument(
        "--token",
        "-t",
        default=os.getenv("AGENT_RAG_TOKEN"),
        help="Authentication token (optional, can also use AGENT_RAG_TOKEN env var)",
    )

    args = parser.parse_args()

    # Create client for remote server
    client = Client(args.server_url)

    # Create proxy that exposes remote server via stdio
    proxy = FastMCP.as_proxy(client)

    # Run proxy with stdio transport (this handles its own event loop)
    proxy.run(transport="stdio")


if __name__ == "__main__":
    main()
