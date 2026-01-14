"""
SAGUARO LangChain Tool Integrations

Phase 5 Implementation: AI Model Saguaro Adoption

Provides native LangChain tool definitions for Saguaro's semantic operations.
These tools can be used directly in LangChain agents, ReAct chains, and
OpenAI function calling setups.

Usage:
    from saguaro.integrations.langchain_tools import get_all_saguaro_tools

    tools = get_all_saguaro_tools(repo_path="/path/to/repo")
    agent = initialize_agent(tools, llm, agent_type="zero-shot-react")

The tools are designed with descriptions that emphasize they should be
used INSTEAD of generic filesystem tools (grep, find, view_file).
"""

import subprocess
import os
import logging
from typing import Optional, Type, List, Dict, Any

logger = logging.getLogger(__name__)

# Try to import LangChain, gracefully handle if not installed
try:
    from langchain.tools import BaseTool
    from langchain.callbacks.manager import CallbackManagerForToolRun
    from pydantic import BaseModel, Field

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    # Create stub classes for when LangChain isn't installed
    class BaseTool:
        """Stub BaseTool for when LangChain is not installed."""

        pass

    class BaseModel:
        """Stub BaseModel for when Pydantic is not installed."""

        pass

    def Field(*args, **kwargs):
        return None

    CallbackManagerForToolRun = None


# Tool Input Schemas
class QueryInput(BaseModel):
    """Input schema for semantic query tool."""

    query: str = Field(
        description="Natural language description of the code you're looking for. "
        "Be specific about concepts, functionality, or patterns."
    )
    k: int = Field(default=5, description="Number of results to return (1-20)")


class SkeletonInput(BaseModel):
    """Input schema for file skeleton tool."""

    file_path: str = Field(
        description="Absolute or relative path to the file to analyze"
    )


class SliceInput(BaseModel):
    """Input schema for context slice tool."""

    symbol: str = Field(
        description="Symbol name to slice (e.g., 'ClassName.method' or 'function_name')"
    )
    depth: int = Field(default=2, description="Dependency depth to include (1-5)")


class VerifyInput(BaseModel):
    """Input schema for verification tool."""

    path: str = Field(default=".", description="Path to verify (file or directory)")
    engines: str = Field(
        default="native,ruff,semantic",
        description="Comma-separated list of verification engines",
    )


class SaguaroQueryTool(BaseTool):
    """Semantic search over the codebase knowledge graph.

    This is the PRIMARY tool for finding code by meaning/concept.
    Use this INSTEAD of grep_search or find_by_name.

    Saguaro provides:
    - Semantic understanding (finds by meaning, not just text)
    - 10-100x fewer tokens than grepping
    - Ranked results with relevance scores
    - Symbol type and location information
    """

    name: str = "saguaro_semantic_search"
    description: str = """PRIMARY tool for finding code by meaning/concept.
Use this INSTEAD of grep_search or find_by_name.

Input: Natural language description of what you're looking for
Output: Ranked list of relevant code entities with file paths and scores

Examples:
- "user authentication and session management"
- "database connection pooling"
- "error handling for API requests"
- "class that processes payment transactions"

DO NOT use grep_search or find_by_name - use this tool instead."""

    args_schema: Type[BaseModel] = QueryInput
    repo_path: str = "."
    venv_path: str = "./venv/bin/saguaro"

    def __init__(self, repo_path: str = ".", **kwargs):
        super().__init__(**kwargs)
        self.repo_path = repo_path
        self.venv_path = os.path.join(repo_path, "venv", "bin", "saguaro")

    def _run(
        self,
        query: str,
        k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute semantic search."""
        try:
            result = subprocess.run(
                [self.venv_path, "query", query, "--k", str(k), "--json"],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=30,
            )

            if result.returncode != 0:
                return f"Error running saguaro query: {result.stderr}"

            return result.stdout

        except subprocess.TimeoutExpired:
            return "Error: Query timed out after 30 seconds"
        except FileNotFoundError:
            return f"Error: Saguaro not found at {self.venv_path}. Run 'saguaro init' first."
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, query: str, k: int = 5) -> str:
        """Async version - just calls sync for now."""
        return self._run(query, k)


class SaguaroSkeletonTool(BaseTool):
    """Generate a file skeleton showing structure without full code.

    Use this INSTEAD of view_file to understand a file's structure.
    Shows function/class signatures and docstrings only.
    Saves 90% of tokens compared to reading full files.
    """

    name: str = "saguaro_file_skeleton"
    description: str = """Get file structure WITHOUT reading full code.
Use this INSTEAD of view_file for initial file exploration.

Shows:
- Function/method signatures
- Class definitions
- Docstrings
- Import statements

Saves 90% of tokens compared to view_file.
After using this, use saguaro_context_slice to read specific functions."""

    args_schema: Type[BaseModel] = SkeletonInput
    repo_path: str = "."
    venv_path: str = "./venv/bin/saguaro"

    def __init__(self, repo_path: str = ".", **kwargs):
        super().__init__(**kwargs)
        self.repo_path = repo_path
        self.venv_path = os.path.join(repo_path, "venv", "bin", "saguaro")

    def _run(
        self, file_path: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Generate file skeleton."""
        try:
            result = subprocess.run(
                [self.venv_path, "agent", "skeleton", file_path],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=30,
            )

            if result.returncode != 0:
                return f"Error generating skeleton: {result.stderr}"

            return result.stdout

        except subprocess.TimeoutExpired:
            return "Error: Skeleton generation timed out"
        except FileNotFoundError:
            return f"Error: Saguaro not found at {self.venv_path}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, file_path: str) -> str:
        return self._run(file_path)


class SaguaroSliceTool(BaseTool):
    """Read a specific symbol with its dependencies and context.

    Use this INSTEAD of view_file or view_code_item to read function code.
    Automatically includes imports and parent context.
    """

    name: str = "saguaro_context_slice"
    description: str = """Read specific function/class with context.
Use this INSTEAD of view_file or view_code_item.

Provides:
- Requested symbol's full implementation
- Related imports
- Parent class context (if method)
- Dependency signatures

Use AFTER saguaro_file_skeleton to read specific code.
If symbol not found, use saguaro_semantic_search to find it."""

    args_schema: Type[BaseModel] = SliceInput
    repo_path: str = "."
    venv_path: str = "./venv/bin/saguaro"

    def __init__(self, repo_path: str = ".", **kwargs):
        super().__init__(**kwargs)
        self.repo_path = repo_path
        self.venv_path = os.path.join(repo_path, "venv", "bin", "saguaro")

    def _run(
        self,
        symbol: str,
        depth: int = 2,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Generate context slice."""
        try:
            result = subprocess.run(
                [self.venv_path, "agent", "slice", symbol, "--depth", str(depth)],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=30,
            )

            if result.returncode != 0:
                # Check for actionable error
                if "INDEX_MISS" in result.stderr or "Symbol not found" in result.stderr:
                    return f"""Symbol '{symbol}' not found in index.

Recovery steps:
1. Use saguaro_semantic_search to find the correct symbol name
2. Check index health with 'saguaro health'
3. Rebuild index if needed with 'saguaro index --path .'

Error details: {result.stderr}"""
                return f"Error: {result.stderr}"

            return result.stdout

        except subprocess.TimeoutExpired:
            return "Error: Slice generation timed out"
        except FileNotFoundError:
            return f"Error: Saguaro not found at {self.venv_path}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, symbol: str, depth: int = 2) -> str:
        return self._run(symbol, depth)


class SaguaroVerifyTool(BaseTool):
    """Verify code compliance before completing a task.

    MANDATORY: Run this before marking any task as complete.
    Checks for linting errors, security issues, and compliance violations.
    """

    name: str = "saguaro_verify"
    description: str = """MANDATORY verification before task completion.
Run this to check code compliance and catch issues.

Checks:
- Linting (Ruff)
- Security patterns
- Semantic drift
- Style compliance

If violations found, fix them before marking task complete."""

    args_schema: Type[BaseModel] = VerifyInput
    repo_path: str = "."
    venv_path: str = "./venv/bin/saguaro"

    def __init__(self, repo_path: str = ".", **kwargs):
        super().__init__(**kwargs)
        self.repo_path = repo_path
        self.venv_path = os.path.join(repo_path, "venv", "bin", "saguaro")

    def _run(
        self,
        path: str = ".",
        engines: str = "native,ruff,semantic",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run verification."""
        try:
            result = subprocess.run(
                [
                    self.venv_path,
                    "verify",
                    path,
                    "--engines",
                    engines,
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=120,
            )

            if result.returncode != 0:
                violations = result.stdout
                return f"VERIFICATION FAILED:\n{violations}\n\nFix these issues before task completion."

            return "Verification PASSED: No violations found."

        except subprocess.TimeoutExpired:
            return "Error: Verification timed out after 2 minutes"
        except FileNotFoundError:
            return f"Error: Saguaro not found at {self.venv_path}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(
        self, path: str = ".", engines: str = "native,ruff,semantic"
    ) -> str:
        return self._run(path, engines)


def get_all_saguaro_tools(repo_path: str = ".") -> List[BaseTool]:
    """Get all Saguaro tools configured for a repository.

    Args:
        repo_path: Path to the repository root.

    Returns:
        List of LangChain tools ready for use in an agent.

    Example:
        tools = get_all_saguaro_tools("/path/to/repo")
        agent = initialize_agent(tools, llm, agent_type="zero-shot-react")
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is not installed. Install with: pip install langchain"
        )

    return [
        SaguaroQueryTool(repo_path=repo_path),
        SaguaroSkeletonTool(repo_path=repo_path),
        SaguaroSliceTool(repo_path=repo_path),
        SaguaroVerifyTool(repo_path=repo_path),
    ]


def get_tool_definitions_for_openai() -> List[Dict[str, Any]]:
    """Get Saguaro tool definitions in OpenAI function format.

    Returns:
        List of function definitions for OpenAI API.
    """
    return [
        {
            "name": "saguaro_semantic_search",
            "description": "PRIMARY tool for finding code by meaning. Use INSTEAD of grep/find.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of code to find",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results (default: 5)",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "saguaro_file_skeleton",
            "description": "Get file structure without full code. Use INSTEAD of view_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file"}
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "saguaro_context_slice",
            "description": "Read specific function/class with dependencies. Use INSTEAD of view_code_item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol name (e.g., 'ClassName.method')",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Dependency depth (default: 2)",
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "saguaro_verify",
            "description": "MANDATORY verification before task completion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to verify"}
                },
                "required": [],
            },
        },
    ]
