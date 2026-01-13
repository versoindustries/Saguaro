"""
SAGUARO Adoption Metrics

Phase 6 Implementation: AI Model Saguaro Adoption

Tracks tool usage patterns to measure AI model adoption of Saguaro tools.
This helps identify when models fall back to generic tools and provides
visibility into adoption improvement over time.

Usage:
    from saguaro.mcp.adoption_metrics import AdoptionTracker
    
    tracker = AdoptionTracker()
    tracker.record_tool_use("saguaro_query")
    tracker.record_fallback_use("grep_search")
    
    print(tracker.get_adoption_score())  # 0.5
"""

import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolUsageEvent:
    """Record of a single tool usage."""
    tool_name: str
    timestamp: float
    is_saguaro: bool
    context: Optional[str] = None


@dataclass
class SessionMetrics:
    """Metrics for a single agent session."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    
    # Saguaro tool usage
    saguaro_query: int = 0
    saguaro_skeleton: int = 0
    saguaro_slice: int = 0
    saguaro_verify: int = 0
    
    # Fallback tool usage
    grep_search: int = 0
    find_by_name: int = 0
    view_file: int = 0
    view_code_item: int = 0
    
    # Computed scores
    events: List[ToolUsageEvent] = field(default_factory=list)
    
    @property
    def total_saguaro(self) -> int:
        return (
            self.saguaro_query + 
            self.saguaro_skeleton + 
            self.saguaro_slice + 
            self.saguaro_verify
        )
    
    @property
    def total_fallback(self) -> int:
        return (
            self.grep_search + 
            self.find_by_name + 
            self.view_file + 
            self.view_code_item
        )
    
    @property
    def adoption_score(self) -> float:
        """Calculate adoption score: saguaro / (saguaro + fallback)."""
        total = self.total_saguaro + self.total_fallback
        if total == 0:
            return 1.0  # Perfect score if no exploration needed
        return self.total_saguaro / total


class AdoptionTracker:
    """Tracks and persists Saguaro adoption metrics.
    
    Stores metrics in .saguaro/metrics.json for persistence across sessions.
    Provides both per-session and aggregate adoption scores.
    """
    
    # Tool name mappings
    SAGUARO_TOOLS = {
        'saguaro_query', 'saguaro_semantic_search', 'query',
        'saguaro_skeleton', 'saguaro_file_skeleton', 'skeleton',
        'saguaro_slice', 'saguaro_context_slice', 'slice',
        'saguaro_verify', 'verify',
    }
    
    FALLBACK_TOOLS = {
        'grep_search', 'grep',
        'find_by_name', 'find',
        'view_file', 'read_file',
        'view_code_item',
    }
    
    def __init__(self, saguaro_dir: Optional[str] = None):
        """Initialize the adoption tracker.
        
        Args:
            saguaro_dir: Path to .saguaro directory. Defaults to cwd/.saguaro
        """
        if saguaro_dir:
            self.saguaro_dir = Path(saguaro_dir)
        else:
            self.saguaro_dir = Path.cwd() / ".saguaro"
        
        self.metrics_file = self.saguaro_dir / "metrics.json"
        self.current_session: Optional[SessionMetrics] = None
        self.all_sessions: List[SessionMetrics] = []
        
        self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from disk."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct sessions (simplified - just load aggregate)
                self._aggregate = data.get('aggregate', {
                    'saguaro_query': 0,
                    'saguaro_skeleton': 0,
                    'saguaro_slice': 0,
                    'saguaro_verify': 0,
                    'grep_search': 0,
                    'find_by_name': 0,
                    'view_file': 0,
                    'view_code_item': 0,
                })
                self._session_count = data.get('session_count', 0)
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")
                self._aggregate = {}
                self._session_count = 0
        else:
            self._aggregate = {
                'saguaro_query': 0,
                'saguaro_skeleton': 0,
                'saguaro_slice': 0,
                'saguaro_verify': 0,
                'grep_search': 0,
                'find_by_name': 0,
                'view_file': 0,
                'view_code_item': 0,
            }
            self._session_count = 0
    
    def _save_metrics(self):
        """Save metrics to disk."""
        try:
            self.saguaro_dir.mkdir(parents=True, exist_ok=True)
            
            data = {
                'aggregate': self._aggregate,
                'session_count': self._session_count,
                'last_updated': datetime.now().isoformat(),
            }
            
            # Add current session if active
            if self.current_session:
                data['current_session'] = {
                    'session_id': self.current_session.session_id,
                    'adoption_score': self.current_session.adoption_score,
                }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new tracking session.
        
        Args:
            session_id: Optional custom session ID.
        
        Returns:
            The session ID.
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        self.current_session = SessionMetrics(
            session_id=session_id,
            start_time=time.time()
        )
        self._session_count += 1
        
        return session_id
    
    def end_session(self):
        """End the current session and save metrics."""
        if self.current_session:
            self.current_session.end_time = time.time()
            self.all_sessions.append(self.current_session)
            self._save_metrics()
            self.current_session = None
    
    def record_tool_use(self, tool_name: str, context: Optional[str] = None):
        """Record usage of a Saguaro tool.
        
        Args:
            tool_name: Name of the tool used.
            context: Optional context about the usage.
        """
        self._record(tool_name, is_saguaro=True, context=context)
    
    def record_fallback_use(self, tool_name: str, context: Optional[str] = None):
        """Record usage of a fallback tool.
        
        Args:
            tool_name: Name of the fallback tool used.
            context: Optional context about the usage.
        """
        self._record(tool_name, is_saguaro=False, context=context)
    
    def _record(self, tool_name: str, is_saguaro: bool, context: Optional[str] = None):
        """Internal method to record tool usage."""
        # Ensure session exists
        if not self.current_session:
            self.start_session()
        
        event = ToolUsageEvent(
            tool_name=tool_name,
            timestamp=time.time(),
            is_saguaro=is_saguaro,
            context=context
        )
        self.current_session.events.append(event)
        
        # Update session counters
        normalized = self._normalize_tool_name(tool_name)
        if hasattr(self.current_session, normalized):
            setattr(
                self.current_session, 
                normalized, 
                getattr(self.current_session, normalized) + 1
            )
        
        # Update aggregate
        if normalized in self._aggregate:
            self._aggregate[normalized] += 1
        
        # Auto-save periodically
        if len(self.current_session.events) % 10 == 0:
            self._save_metrics()
    
    def _normalize_tool_name(self, tool_name: str) -> str:
        """Normalize tool name to standard form."""
        mappings = {
            'saguaro_semantic_search': 'saguaro_query',
            'query': 'saguaro_query',
            'saguaro_file_skeleton': 'saguaro_skeleton',
            'skeleton': 'saguaro_skeleton',
            'saguaro_context_slice': 'saguaro_slice',
            'slice': 'saguaro_slice',
            'verify': 'saguaro_verify',
            'grep': 'grep_search',
            'find': 'find_by_name',
            'read_file': 'view_file',
        }
        return mappings.get(tool_name, tool_name)
    
    def get_adoption_score(self) -> float:
        """Get the current session's adoption score.
        
        Returns:
            Adoption score between 0.0 and 1.0.
        """
        if self.current_session:
            return self.current_session.adoption_score
        return self.get_aggregate_score()
    
    def get_aggregate_score(self) -> float:
        """Get the aggregate adoption score across all sessions.
        
        Returns:
            Aggregate adoption score between 0.0 and 1.0.
        """
        saguaro = (
            self._aggregate.get('saguaro_query', 0) +
            self._aggregate.get('saguaro_skeleton', 0) +
            self._aggregate.get('saguaro_slice', 0) +
            self._aggregate.get('saguaro_verify', 0)
        )
        fallback = (
            self._aggregate.get('grep_search', 0) +
            self._aggregate.get('find_by_name', 0) +
            self._aggregate.get('view_file', 0) +
            self._aggregate.get('view_code_item', 0)
        )
        
        total = saguaro + fallback
        if total == 0:
            return 1.0
        return saguaro / total
    
    def get_report(self) -> Dict:
        """Get a full adoption report.
        
        Returns:
            Dictionary with detailed metrics.
        """
        return {
            'aggregate': self._aggregate.copy(),
            'session_count': self._session_count,
            'adoption_score': self.get_aggregate_score(),
            'current_session': {
                'id': self.current_session.session_id if self.current_session else None,
                'score': self.current_session.adoption_score if self.current_session else None,
                'events': len(self.current_session.events) if self.current_session else 0,
            } if self.current_session else None,
            'recommendation': self._get_recommendation(),
        }
    
    def _get_recommendation(self) -> str:
        """Get a recommendation based on current adoption score."""
        score = self.get_aggregate_score()
        
        if score >= 0.9:
            return "Excellent! Saguaro tools are being used effectively."
        elif score >= 0.7:
            return "Good adoption. Consider replacing remaining view_file calls with skeleton+slice."
        elif score >= 0.5:
            return "Moderate adoption. Train the model to use saguaro query instead of grep_search."
        else:
            return "Low adoption. Review GEMINI.md and ensure Saguaro-First Protocol is enforced."
    
    def print_report(self):
        """Print a formatted adoption report."""
        report = self.get_report()
        
        print("\n=== SAGUARO Adoption Metrics ===\n")
        print(f"Adoption Score: {report['adoption_score']:.2%}")
        print(f"Total Sessions: {report['session_count']}")
        
        print("\n[Saguaro Tools]")
        for key in ['saguaro_query', 'saguaro_skeleton', 'saguaro_slice', 'saguaro_verify']:
            print(f"  {key}: {report['aggregate'].get(key, 0)}")
        
        print("\n[Fallback Tools]")
        for key in ['grep_search', 'find_by_name', 'view_file', 'view_code_item']:
            print(f"  {key}: {report['aggregate'].get(key, 0)}")
        
        if report['current_session']:
            print("\n[Current Session]")
            print(f"  ID: {report['current_session']['id']}")
            print(f"  Score: {report['current_session']['score']:.2%}")
            print(f"  Events: {report['current_session']['events']}")
        
        print("\n[Recommendation]")
        print(f"  {report['recommendation']}")
        print()


# Singleton instance for easy access
_global_tracker: Optional[AdoptionTracker] = None


def get_tracker() -> AdoptionTracker:
    """Get the global adoption tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = AdoptionTracker()
    return _global_tracker
