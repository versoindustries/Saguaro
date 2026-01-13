"""
SAGUARO MCP Tool Interceptor

Phase 4 Implementation: AI Model Saguaro Adoption

This module provides a mechanism to intercept generic tool calls
(grep_search, find_by_name, view_file) and suggest Saguaro alternatives.
This helps guide AI models toward using Saguaro's semantic tools.

Usage:
    interceptor = SaguaroToolInterceptor()
    result = interceptor.intercept("grep_search", {"query": "auth", "path": "."})
    if result.blocked:
        print(result.message)  # Suggests saguaro query instead
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class InterceptResult:
    """Result of tool interception check.
    
    Attributes:
        allowed: Whether the tool call should proceed.
        blocked: Whether the tool call was completely blocked.
        warned: Whether the tool call triggered a warning but was allowed.
        message: Human-readable message explaining the interception.
        alternative: The Saguaro command that should be used instead.
        context: Additional context for the decision.
    """
    allowed: bool = True
    blocked: bool = False
    warned: bool = False
    message: str = ""
    alternative: str = ""
    context: Optional[Dict] = None


class SaguaroToolInterceptor:
    """Intercepts generic filesystem tools and suggests Saguaro alternatives.
    
    This implements the enforcement layer for AI model Saguaro adoption.
    When mounted as part of an MCP server or agent framework, it can
    block or warn on usage of generic tools when Saguaro alternatives exist.
    
    Attributes:
        BLOCKED_TOOLS: Tools that are completely blocked with Saguaro alternatives.
        WARNED_TOOLS: Tools that trigger warnings but are allowed to proceed.
        mode: Operation mode - 'strict' blocks tools, 'advisory' only warns.
    """
    
    # Tools that should be completely blocked for code discovery
    BLOCKED_TOOLS: Dict[str, str] = {
        'grep_search': 'saguaro query',
        'find_by_name': 'saguaro query',
    }
    
    # Tools that trigger warnings but may be allowed in some contexts
    WARNED_TOOLS: Dict[str, str] = {
        'view_file': 'saguaro agent skeleton + slice',
        'view_code_item': 'saguaro agent slice',
    }
    
    # Exceptions: contexts where fallback tools are acceptable
    ALLOWED_EXCEPTIONS = [
        'binary_file',      # Binary files can't use Saguaro
        'non_code_file',    # Config files, documentation
        'saguaro_error',    # When Saguaro itself failed
        'explicit_path',    # When user explicitly specified a path
    ]
    
    def __init__(self, mode: str = 'advisory'):
        """Initialize the interceptor.
        
        Args:
            mode: Operation mode - 'strict' blocks tools, 'advisory' only warns.
        """
        self.mode = mode
        self._stats = {
            'blocked': 0,
            'warned': 0,
            'allowed': 0,
        }
    
    def intercept(
        self, 
        tool_name: str, 
        args: Dict,
        context: Optional[Dict] = None
    ) -> InterceptResult:
        """Check if a tool call should be intercepted.
        
        Args:
            tool_name: Name of the tool being called.
            args: Arguments passed to the tool.
            context: Optional context about why the tool is being used.
        
        Returns:
            InterceptResult with decision and guidance.
        """
        context = context or {}
        
        # Check for explicit exceptions
        exception_reason = context.get('exception_reason')
        if exception_reason in self.ALLOWED_EXCEPTIONS:
            logger.debug(f"Tool {tool_name} allowed due to exception: {exception_reason}")
            self._stats['allowed'] += 1
            return InterceptResult(
                allowed=True,
                message=f"Fallback tool allowed: {exception_reason}",
                context={'exception': exception_reason}
            )
        
        # Check blocked tools
        if tool_name in self.BLOCKED_TOOLS:
            alternative = self.BLOCKED_TOOLS[tool_name]
            suggested_cmd = self._generate_suggestion(tool_name, args)
            
            message = f"""BLOCKED: {tool_name} is forbidden for code exploration.

Use Saguaro's semantic search instead:
  {suggested_cmd}

Saguaro provides:
  ✓ Semantic understanding (finds by meaning, not just text)
  ✓ 10-100x fewer tokens
  ✓ Ranked results with context

If Saguaro fails, document the error and check index health with:
  saguaro health
"""
            
            if self.mode == 'strict':
                self._stats['blocked'] += 1
                return InterceptResult(
                    blocked=True,
                    allowed=False,
                    message=message,
                    alternative=suggested_cmd,
                    context={'original_tool': tool_name, 'original_args': args}
                )
            else:
                # Advisory mode: warn but allow
                self._stats['warned'] += 1
                return InterceptResult(
                    warned=True,
                    allowed=True,
                    message=f"WARNING: Consider using {alternative} instead of {tool_name}.\n\nSuggested: {suggested_cmd}",
                    alternative=suggested_cmd,
                    context={'original_tool': tool_name, 'original_args': args}
                )
        
        # Check warned tools
        if tool_name in self.WARNED_TOOLS:
            alternative = self.WARNED_TOOLS[tool_name]
            suggested_cmd = self._generate_suggestion(tool_name, args)
            
            message = f"""WARNING: {tool_name} wastes tokens for code exploration.

Better approach with Saguaro:
  {suggested_cmd}

Benefits:
  ✓ Saves 90% of tokens
  ✓ Shows only relevant code
  ✓ Includes dependency context
"""
            
            self._stats['warned'] += 1
            return InterceptResult(
                warned=True,
                allowed=True,
                message=message,
                alternative=suggested_cmd,
                context={'original_tool': tool_name, 'original_args': args}
            )
        
        # Tool is allowed without any interception
        self._stats['allowed'] += 1
        return InterceptResult(allowed=True)
    
    def _generate_suggestion(self, tool_name: str, args: Dict) -> str:
        """Generate a specific Saguaro command suggestion.
        
        Args:
            tool_name: The blocked/warned tool name.
            args: The arguments passed to the original tool.
        
        Returns:
            A specific Saguaro command string.
        """
        if tool_name in ('grep_search', 'find_by_name'):
            query = args.get('query') or args.get('pattern') or args.get('Query') or 'your search term'
            return f'./venv/bin/saguaro query "{query}" --k 5'
        
        elif tool_name == 'view_file':
            file_path = args.get('path') or args.get('AbsolutePath') or 'file.py'
            return f"""./venv/bin/saguaro agent skeleton {file_path}
# Then to read a specific symbol:
./venv/bin/saguaro agent slice SymbolName --depth 2"""
        
        elif tool_name == 'view_code_item':
            symbol = args.get('symbol') or args.get('NodePaths', ['Symbol'])[0] or 'Symbol'
            return f'./venv/bin/saguaro agent slice {symbol} --depth 2'
        
        return './venv/bin/saguaro query "description" --k 5'
    
    def get_stats(self) -> Dict[str, int]:
        """Get interception statistics.
        
        Returns:
            Dictionary with blocked, warned, and allowed counts.
        """
        return self._stats.copy()
    
    def get_adoption_score(self) -> float:
        """Calculate Saguaro adoption score.
        
        Score = saguaro_calls / (saguaro_calls + fallback_calls)
        
        Returns:
            Adoption score between 0.0 and 1.0.
        """
        total = sum(self._stats.values())
        if total == 0:
            return 1.0  # Perfect score if no calls made
        
        # In this context, 'blocked' and 'warned' represent fallback attempts
        fallback_attempts = self._stats['blocked'] + self._stats['warned']
        saguaro_proper = self._stats['allowed']
        
        if saguaro_proper + fallback_attempts == 0:
            return 1.0
        
        return saguaro_proper / (saguaro_proper + fallback_attempts)


def create_langchain_tool_wrapper(interceptor: SaguaroToolInterceptor):
    """Create a LangChain tool wrapper that enforces Saguaro usage.
    
    This is a factory function that returns a decorator for LangChain tools.
    
    Example:
        wrapper = create_langchain_tool_wrapper(interceptor)
        
        @wrapper
        def grep_search(query: str, path: str):
            # This will be intercepted
            pass
    """
    def wrapper(func):
        def intercepted_func(*args, **kwargs):
            result = interceptor.intercept(func.__name__, kwargs)
            if result.blocked:
                raise RuntimeError(result.message)
            if result.warned:
                logger.warning(result.message)
            return func(*args, **kwargs)
        return intercepted_func
    return wrapper
