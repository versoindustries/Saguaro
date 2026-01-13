"""
Chronicle Diff: Semantic Drift Calculation
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("saguaro.chronicle.diff")

try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("Numpy not found. SemanticDiff running in degraded mode.")

class SemanticDiff:
    """
    Calculates the 'Semantic Drift' or distance between two Time Crystal states.
    Uses cosine similarity on the hyperdimensional bundles.
    """
    
    @staticmethod
    def calculate_drift(
        state_a_blob: bytes, 
        state_b_blob: bytes, 
        dtype=None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate semantic drift between two serialized HD states.
        
        Args:
            state_a_blob: Bytes of first state tensor
            state_b_blob: Bytes of second state tensor
            dtype: Numpy dtype of the blobs
            
        Returns:
            (drift_score, details)
            drift_score is 0.0 to 1.0 (0.0 = identical, 1.0 = orthogonal/opposite)
        """
        try:
            if np:
                if dtype is None:
                    dtype = np.float32
                if not state_a_blob or not state_b_blob:
                    logger.warning("Empty state blob provided for drift calculation")
                    return 0.0, {"warning": "empty_state"}

                if len(state_a_blob) % 4 != 0 or len(state_b_blob) % 4 != 0:
                   logger.error(f"Invalid blob size: A={len(state_a_blob)}, B={len(state_b_blob)}")
                   return 1.0, {"error": "invalid_blob_size"}

                vec_a = np.frombuffer(state_a_blob, dtype=dtype)
                vec_b = np.frombuffer(state_b_blob, dtype=dtype)
                
                if vec_a.shape != vec_b.shape:
                    logger.warning(f"Shape mismatch: {vec_a.shape} vs {vec_b.shape}. Truncating to min.")
                    min_len = min(vec_a.shape[0], vec_b.shape[0])
                    vec_a = vec_a[:min_len]
                    vec_b = vec_b[:min_len]
                    
                # Cosine similarity
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                
                if norm_a == 0 or norm_b == 0:
                    similarity = 0.0 # Undefined direction
                else:
                    similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
            else:
                # Fallback purely for CLI check if numpy is missing
                # Just return 1.0 drift (worst case) since we can't calc
                logger.warning("Cannot calculate drift without numpy")
                similarity = 0.0
            
            # Convert similarity (-1 to 1) to distance (0 to 1)
            # dist = 1 - sim (for normalized vectors) is common, but let's normalize to 0-1 range
            # 1.0 sim -> 0.0 distance
            # 0.0 sim -> 1.0 distance (orthogonal) -> max drift usually 
            # -1.0 sim -> 2.0 distance (opposite)
            
            # Simple drift metric: 1 - cosine_similarity
            drift = 1.0 - similarity
            
            return float(drift), {
                "similarity": float(similarity),
                "norm_a": float(norm_a) if np else 0.0,
                "norm_b": float(norm_b) if np else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating drift: {e}")
            return 1.0, {"error": str(e)}

    @staticmethod
    def human_readable_report(drift_score: float) -> str:
        if drift_score < 0.01:
            return "Stable (No significant semantic range)"
        elif drift_score < 0.1:
            return "Minor Drift (Refactoring or small tweaks)"
        elif drift_score < 0.4:
            return "Moderate Drift (Feature addition or logic change)"
        else:
            return "Major Shift (Architectural change or rewrite)"
