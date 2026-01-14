"""
SAGUARO Vector Store
Simple storage backend for Hyperdimensional Vectors.

NOTE: This module now re-exports from memmap_vector_store for new indexes.
The original pickle-based class is renamed to LegacyVectorStore for backward compatibility.
"""

import os
import json
import logging
import pickle
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class LegacyVectorStore:
    def __init__(self, storage_path: str, dim: int, dark_space_ratio: float = 0.4):
        self.storage_path = storage_path
        self.dim = dim
        self.dark_space_ratio = dark_space_ratio

        self.index_path = os.path.join(storage_path, "index.pkl")
        self.metadata_path = os.path.join(storage_path, "metadata.json")

        self.vectors = []
        self.metadata = []

        self._load()

    def _load(self):
        """Load vectors and metadata from disk with validation.

        Performs dimension validation to prevent inhomogeneous shape errors.
        Corrupt or mismatched entries are filtered out with warnings.
        """
        raw_vectors = []
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    raw_vectors = pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                logger.warning(f"Failed to load index: {e}. Starting fresh.")
                raw_vectors = []

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to load metadata: {e}. Starting fresh.")
                self.metadata = []
        else:
            self.metadata = []

        # Validate and filter vectors to ensure homogeneous dimensions
        valid_vectors = []
        valid_metadata = []
        invalid_count = 0

        for i, vec in enumerate(raw_vectors):
            try:
                vec_array = np.asarray(vec, dtype=np.float32)
                if vec_array.ndim == 1 and vec_array.shape[0] == self.dim:
                    valid_vectors.append(vec_array)
                    if i < len(self.metadata):
                        valid_metadata.append(self.metadata[i])
                    else:
                        valid_metadata.append(
                            {"name": f"unknown_{i}", "type": "unknown"}
                        )
                else:
                    invalid_count += 1
                    logger.debug(
                        f"Filtered vector {i}: shape {vec_array.shape} != expected ({self.dim},)"
                    )
            except (ValueError, TypeError) as e:
                invalid_count += 1
                logger.debug(f"Filtered invalid vector {i}: {e}")

        if invalid_count > 0:
            logger.warning(
                f"Filtered {invalid_count} vectors with inconsistent dimensions. "
                f"Valid: {len(valid_vectors)}, Original: {len(raw_vectors)}"
            )

        self.vectors = valid_vectors
        self.metadata = valid_metadata

    def save(self):
        os.makedirs(self.storage_path, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.vectors, f)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def add(self, vector: np.ndarray, meta: Dict[str, Any]):
        """
        Adds a vector to the store.
        """
        # Ensure vector matches dim
        if vector.shape[0] != self.dim:
            # Pad or truncate (shouldn't happen if pipeline is correct)
            if vector.shape[0] < self.dim:
                vector = np.pad(vector, (0, self.dim - vector.shape[0]))
            else:
                vector = vector[: self.dim]

        self.vectors.append(vector)
        self.metadata.append(meta)

    def query(
        self, query_vec: np.ndarray, k: int = 5, boost_map: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Naive linear scan for prototype.
        Returns top-K results with explanations.
        """
        if not self.vectors:
            return []

        # Cosine similarity
        # query_vec: [D]
        # database: [N, D]

        db = np.array(self.vectors)  # [N, D]

        # Norms
        q_norm = np.linalg.norm(query_vec)
        db_norm = np.linalg.norm(db, axis=1)

        scores = np.dot(db, query_vec) / (db_norm * q_norm + 1e-9)

        if boost_map:
            # Apply graph-based boosting
            for i, meta in enumerate(self.metadata):
                name = meta.get("name")
                if name and name in boost_map:
                    scores[i] += (
                        boost_map[name] * 0.2
                    )  # Weighted boost using 0.2 factor

        # Top K
        indices = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(indices):
            res = self.metadata[idx].copy()
            score = float(scores[idx])
            res["score"] = score

            # --- Explainability Layer ---
            # Basic heuristics to explain WHY this was retrieved
            reasons = []

            # 1. Score-based Confidence
            if score > 0.8:
                reasons.append("High semantic similarity match.")
            elif score > 0.5:
                reasons.append("Moderate similarity; likely contextually relevant.")
            else:
                reasons.append("Low confidence match; potential conceptual overlap.")

            # 2. Type-based Context
            entity_type = res.get("type", "unknown")
            if entity_type == "file":
                reasons.append("Core module match.")
            elif entity_type == "class":
                reasons.append("Structural definition match.")
            elif entity_type == "function":
                reasons.append("Functional logic match.")

            # 3. Path heuristic
            file_path = res.get("file", "")
            if "tests" in file_path:
                reasons.append("Provides usage examples via tests.")
            elif "docs" in file_path:
                reasons.append("Documentation source.")

            res["reason"] = " ".join(reasons)
            res["rank"] = rank + 1

            results.append(res)

        return results

    def clear(self):
        self.vectors = []
        self.metadata = []
        self.save()
    
    def __len__(self) -> int:
        """Return number of stored vectors."""
        return len(self.vectors)


# =============================================================================
# AUTO-DETECT FORMAT AND USE APPROPRIATE IMPLEMENTATION
# =============================================================================

def VectorStore(storage_path: str, dim: int, dark_space_ratio: float = 0.4, **kwargs):
    """
    Factory function that auto-detects the vector store format.
    
    - If index_meta.json exists with format='memmap', use MemoryMappedVectorStore
    - If index.pkl exists (legacy), use LegacyVectorStore
    - Otherwise, use MemoryMappedVectorStore (new default)
    
    Args:
        storage_path: Directory containing the vector index
        dim: Vector dimension
        dark_space_ratio: Reserved space ratio (for compatibility)
        
    Returns:
        Appropriate VectorStore implementation
    """
    from saguaro.storage.memmap_vector_store import MemoryMappedVectorStore
    
    index_meta_path = os.path.join(storage_path, "index_meta.json")
    legacy_index_path = os.path.join(storage_path, "index.pkl")
    
    # Check for memmap format
    if os.path.exists(index_meta_path):
        try:
            with open(index_meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get('format') == 'memmap':
                logger.debug(f"Using MemoryMappedVectorStore for {storage_path}")
                return MemoryMappedVectorStore(
                    storage_path, dim, dark_space_ratio, **kwargs
                )
        except Exception:
            pass
    
    # Check for legacy pickle format
    if os.path.exists(legacy_index_path):
        logger.debug(f"Using LegacyVectorStore for {storage_path} (pickle format)")
        return LegacyVectorStore(storage_path, dim, dark_space_ratio)
    
    # New index - use memmap by default
    logger.debug(f"Creating new MemoryMappedVectorStore in {storage_path}")
    return MemoryMappedVectorStore(storage_path, dim, dark_space_ratio, **kwargs)
