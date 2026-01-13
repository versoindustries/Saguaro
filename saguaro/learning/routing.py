class IntentRouter:
    """
    Classifies user queries to route them to the optimal retrieval strategy.
    """
    
    def route(self, query: str) -> str:
        """
        Classifies query intent.
        Returns: 'bug_fix', 'refactor', 'security', or 'general_search'.
        """
        q = query.lower()
        if "fix" in q or "bug" in q or "error" in q:
            return "bug_fix"
        if "refactor" in q or "rewrite" in q or "move" in q:
            return "refactor"
        if "security" in q or "vulnerability" in q or "auth" in q:
            return "security"
        return "general_search"
