# agent_rag_mcp/__init__.py
"""Agent RAG MCP - Dynamic RAG server for AI agents."""

import argparse

from agent_rag_mcp.server import mcp


def main() -> None:
    """Entry point for the agent-rag-mcp command."""
    parser = argparse.ArgumentParser(
        description="Agent RAG MCP Server",
        prog="agent-rag-mcp",
    )
    parser.add_argument(
        "--transport",
        "-t",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )

    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
        )


__all__ = ["main", "mcp"]
