"""
SAGUARO Auto-Scaler Module
Calculates the ideal Hyperdimensional (HD) vector size based on codebase complexity.
Implements the "Dark Space Buffer" logic to allow for growth without re-indexing.
"""

import os
import logging

logger = logging.getLogger(__name__)

def count_loc(path: str) -> tuple[int, dict[str, int]]:
    """
    Approximation of Lines of Code (LoC) by walking the directory.
    excludes hidden directories and common build artifacts.
    Returns total LoC and a breakdown by extension/language.
    """
    total_lines = 0
    language_breakdown = {}
    
    # Basic exclusion list
    excludes = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', 'build', 'dist', '.idea', '.vscode'}
    
    # Extensions to count
    # Map extension to language name
    ext_map = {
        '.py': 'Python',
        '.js': 'JavaScript', '.jsx': 'JavaScript',
        '.ts': 'TypeScript', '.tsx': 'TypeScript',
        '.cpp': 'C++', '.cc': 'C++', '.c': 'C', '.h': 'C/C++ Header', '.hpp': 'C++ Header',
        '.java': 'Java',
        '.go': 'Go',
        '.rs': 'Rust',
        '.rb': 'Ruby',
        '.php': 'PHP',
        '.cs': 'C#',
        '.sh': 'Shell',
        '.md': 'Markdown',
        '.html': 'HTML',
        '.css': 'CSS'
    }
    
    for root, dirs, files in os.walk(path):
        # Prune excluded dirs
        dirs[:] = [d for d in dirs if d not in excludes]
        
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in ext_map:
                lang = ext_map[ext]
                try:
                    with open(os.path.join(root, file), 'rb') as f:
                        # Count newlines for fast LoC approximation
                        # Using rb and counting bytes avoids encoding issues
                        lines = sum(1 for _ in f)
                        total_lines += lines
                        language_breakdown[lang] = language_breakdown.get(lang, 0) + lines
                except Exception:
                    pass # Ignore unreadable files
                    
    return total_lines, language_breakdown

def calculate_ideal_dim(loc: int, enterprise_mode: bool = False) -> int:
    """
    Calculates the ideal HD dimension based on LoC.
    
    Formula:
    < 10k LoC   -> 4096
    < 500k LoC  -> 8192
    > 500k LoC  -> 12288
    Enterprise  -> 16384 (if specified or huge LoC)
    """
    if enterprise_mode or loc > 1_000_000:
        return 16384
    
    if loc < 10_000:
        return 4096
    elif loc < 500_000:
        return 8192
    else:
        return 12288

def allocate_dark_space(dim: int, buffer_ratio: float = 0.4) -> int:
    """
    Calculates the total dimension size including the Dark Space Buffer.
    
    Args:
        dim: The active semantic dimension (e.g., 8192)
        buffer_ratio: The ratio of dark space to add (default 0.4 or 40%)
        
    Returns:
        The total dimension size, rounded up to next power of 2 for FFT efficiency.
    """
    # Required size
    min_size = int(dim * (1.0 + buffer_ratio))
    
    # Next power of 2
    power = 1
    while power < min_size:
        power *= 2
        
    return power

def get_repo_stats_and_config(path: str):
    """
    Analyzes a repo and returns the recommended configuration.
    """
    logger.info(f"Analyzing repository at {path}...")
    loc, languages = count_loc(path)
    logger.info(f"Estimated LoC: {loc}")
    
    active_dim = calculate_ideal_dim(loc)
    total_dim = allocate_dark_space(active_dim)
    
    logger.info(f"Recommended Active Dim: {active_dim}")
    logger.info(f"Total Dim (with Dark Space): {total_dim}")
    
    return {
        "loc": loc,
        "languages": languages,
        "active_dim": active_dim,
        "total_dim": total_dim,
        "dark_space_ratio": 1.0 - (active_dim / total_dim)
    }
