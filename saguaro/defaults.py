
"""
SAGUARO Default Configuration Values
Defines the zero-config defaults that overlap with "correct but conservative" principle.
"""

DEFAULT_CONFIG_YAML = """version: 1

indexing:
  # Automatically scale dimensions based on repo size
  auto_scale: true
  
  # How often to poll for changes in watch mode (seconds)
  watch_interval: 5
  
  # Files and directories to exclude from indexing (glob patterns)
  # SAGUARO automatically respects .gitignore, but these are explicit ignores.
  exclude:
    - ".git"
    - ".venv"
    - "venv"
    - "__pycache__"
    - "node_modules"
    - "dist"
    - "build"
    - ".idea"
    - ".vscode"
    - "*.min.js"
    - "*.map"

sentinel:
  # Default verification level
  level: "standard"
  
  # Engines to run by default on 'saguaro verify .'
  # native: fast regex checks
  # ruff: python linting
  # semantic: drift detection
  default_engines: ["native", "ruff", "semantic"]

chronicle:
  # Auto-snapshot on specific events not yet implemented, but config ready
  auto_snapshot: false
"""

def get_default_yaml() -> str:
    return DEFAULT_CONFIG_YAML.strip()
