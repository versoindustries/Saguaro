"""
SAGUARO MCP (Model Context Protocol) Package

Provides:
- MCPServer: JSON-RPC 2.0 MCP server implementation
- SaguaroToolInterceptor: Tool interception for AI adoption enforcement
- AdoptionTracker: Metrics tracking for Saguaro tool adoption
"""

from saguaro.mcp.server import MCPServer
from saguaro.mcp.tool_interceptor import SaguaroToolInterceptor, InterceptResult
from saguaro.mcp.adoption_metrics import AdoptionTracker, get_tracker

__all__ = [
    "MCPServer",
    "SaguaroToolInterceptor",
    "InterceptResult",
    "AdoptionTracker",
    "get_tracker",
]
