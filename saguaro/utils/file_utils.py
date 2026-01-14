import os
import fnmatch
from typing import Generator, List


def get_code_files(root_path: str, exclusions: List[str] = None) -> List[str]:
    """
    Scans the directory for code and configuration files, applying exclusions.
    Supports a broad range of extensions relevant to modern full-stack development.
    """
    if exclusions is None:
        exclusions = []

    # Ensure hard defaults are always present
    hard_defaults = [
        ".saguaro",
        ".git",
        "venv",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
        ".idea",
        ".vscode",
    ]
    exclusions.extend([d for d in hard_defaults if d not in exclusions])

    # Supported Extensions
    # We use a set of whitelisted extensions to avoid indexing binaries/media
    EXT_WHITELIST = {
        # Python
        ".py",
        ".pyi",
        # C/C++
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        # Web
        ".js",
        ".mjs",
        ".jsx",
        ".ts",
        ".tsx",
        ".html",
        ".css",
        ".scss",
        ".less",
        # Backend/Systems
        ".java",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".cs",
        ".kt",
        ".swift",
        # Shell
        ".sh",
        ".bash",
        ".zsh",
        # Config / Data
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        "Dockerfile",
        "Makefile",
        ".ini",
        # Docs
        ".md",
        ".rst",
        ".txt",
    }

    all_files = []

    for root, dirs, files in os.walk(root_path):
        # 1. Prune directories (modify dirs in-place)
        dirs_to_remove = []
        for d in dirs:
            # Check exact match
            if d in exclusions:
                dirs_to_remove.append(d)
                continue
            # Check ignored prefix (e.g. .hidden)
            if d.startswith("."):
                dirs_to_remove.append(d)
                continue
            # Check globs
            for pat in exclusions:
                if fnmatch.fnmatch(d, pat):
                    dirs_to_remove.append(d)
                    break

        for d in dirs_to_remove:
            if d in dirs:
                dirs.remove(d)

        # 2. Filter files
        for file in files:
            # Check file exclusions
            skip_file = False
            for pat in exclusions:
                if fnmatch.fnmatch(file, pat):
                    skip_file = True
                    break
            if skip_file:
                continue

            # Extension check
            _, ext = os.path.splitext(file)

            # Special case for "extensionless" files like Dockerfile or Makefile
            if file in EXT_WHITELIST:  # Exact filename match
                full_path = os.path.join(root, file)
                all_files.append(full_path)
                continue

            if ext in EXT_WHITELIST:
                full_path = os.path.join(root, file)
                all_files.append(full_path)

    return all_files


# --- Streaming File Discovery (Memory Optimized) ---

# Extension whitelist for streaming (same as above, but defined at module level)
_EXT_WHITELIST = {
    ".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
    ".js", ".mjs", ".jsx", ".ts", ".tsx", ".html", ".css", ".scss", ".less",
    ".java", ".go", ".rs", ".rb", ".php", ".cs", ".kt", ".swift",
    ".sh", ".bash", ".zsh", ".json", ".yaml", ".yml", ".toml", ".xml",
    ".ini", ".md", ".rst", ".txt",
}

_HARD_EXCLUSIONS = {
    ".saguaro", ".git", "venv", "__pycache__", "node_modules",
    "build", "dist", ".idea", ".vscode",
}


def iter_code_files(
    root_path: str,
    exclusions: list = None,
    batch_size: int = 64
) -> Generator[list, None, None]:
    """
    Generator-based file discovery that yields batches of files.
    
    MEMORY OPTIMIZED: Does not hold all file paths in memory at once.
    Files are discovered and yielded in batches for immediate processing.
    
    Args:
        root_path: Directory to scan
        exclusions: Additional patterns to exclude
        batch_size: Number of files per batch
        
    Yields:
        Lists of file paths, each containing up to batch_size files
    """
    import fnmatch
    
    if exclusions is None:
        exclusions = []
    
    # Combine with hard defaults
    all_exclusions = set(exclusions) | _HARD_EXCLUSIONS
    
    batch = []
    
    for root, dirs, files in os.walk(root_path):
        # Prune directories in-place
        dirs[:] = [
            d for d in dirs 
            if d not in all_exclusions 
            and not d.startswith(".")
            and not any(fnmatch.fnmatch(d, pat) for pat in all_exclusions)
        ]
        
        # Process files
        for file in files:
            # Check exclusions
            if any(fnmatch.fnmatch(file, pat) for pat in all_exclusions):
                continue
            
            # Check extension
            _, ext = os.path.splitext(file)
            if ext in _EXT_WHITELIST or file in {"Dockerfile", "Makefile"}:
                batch.append(os.path.join(root, file))
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    
    # Yield remaining files
    if batch:
        yield batch
