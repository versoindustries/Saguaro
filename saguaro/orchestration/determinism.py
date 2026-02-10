"""
Deterministic Parameter Profiles for Ollama/Granite 4.
Ensures consistent, repeatable agentic reasoning.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DeterminismProfile:
    temperature: float
    top_p: float
    top_k: int
    seed: int
    repeat_penalty: float

# SAGUARO Default Determinism
SAGUARO_DETERMINISTIC_AI = DeterminismProfile(
    temperature=1e-14,    # Bypasses temp=0 short-circuits
    top_p=1e-14,          # Forces primary candidate token
    top_k=1,              # Limit selection pool
    seed=720720,          # Highly Composite Number for stack stability
    repeat_penalty=1.0    # Neutral distribution
)

def get_ollama_options(profile: DeterminismProfile = SAGUARO_DETERMINISTIC_AI) -> Dict[str, Any]:
    """Returns options dict for Ollama API/CLI."""
    return {
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "top_k": profile.top_k,
        "seed": profile.seed,
        "repeat_penalty": profile.repeat_penalty
    }
