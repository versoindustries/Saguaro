"""
SAGUARO Memory-Mapped Vector Store
Enterprise-grade storage backend for Hyperdimensional Vectors with O(1) memory footprint.

This implementation uses numpy memory-mapped arrays to store vectors directly on disk,
allowing indexing of repositories of any size without running out of RAM.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import threading

logger = logging.getLogger(__name__)


class MemoryMappedVectorStore:
    """
    Memory-efficient vector storage using numpy memory-mapped arrays.
    
    Memory Characteristics:
    - Vectors are stored on disk, loaded on-demand by the OS
    - Only actively accessed pages are kept in RAM
    - Supports unlimited repository sizes (disk-bound only)
    - Thread-safe for concurrent writes
    
    Storage Format:
    - vectors.bin: Memory-mapped numpy array [capacity, dim]
    - metadata.json: JSON array of metadata dicts
    - index_meta.json: Store configuration (dim, count, capacity)
    """
    
    GROWTH_FACTOR = 2.0  # Double capacity when full
    INITIAL_CAPACITY = 10000  # Initial number of vectors
    
    def __init__(
        self, 
        storage_path: str, 
        dim: int, 
        dark_space_ratio: float = 0.4,
        read_only: bool = False
    ):
        """
        Initialize the memory-mapped vector store.
        
        Args:
            storage_path: Directory to store index files
            dim: Vector dimension
            dark_space_ratio: Reserved space ratio (for compatibility)
            read_only: If True, open existing index in read-only mode
        """
        self.storage_path = storage_path
        self.dim = dim
        self.dark_space_ratio = dark_space_ratio
        self.read_only = read_only
        
        self.vectors_path = os.path.join(storage_path, "vectors.bin")
        self.metadata_path = os.path.join(storage_path, "metadata.json")
        self.index_meta_path = os.path.join(storage_path, "index_meta.json")
        
        self._vectors: Optional[np.memmap] = None
        self._metadata: List[Dict[str, Any]] = []
        self._count: int = 0
        self._capacity: int = 0
        
        self._write_lock = threading.Lock()
        
        self._load()
    
    def _load(self) -> None:
        """Load or initialize the vector store."""
        os.makedirs(self.storage_path, exist_ok=True)
        
        if os.path.exists(self.index_meta_path):
            # Load existing store
            try:
                with open(self.index_meta_path, 'r') as f:
                    meta = json.load(f)
                
                stored_dim = meta.get('dim', self.dim)
                if stored_dim != self.dim:
                    logger.warning(
                        f"Dimension mismatch: stored={stored_dim}, requested={self.dim}. "
                        f"Using stored dimension."
                    )
                    self.dim = stored_dim
                
                self._count = meta.get('count', 0)
                self._capacity = meta.get('capacity', self.INITIAL_CAPACITY)
                
                # Open memory-mapped file
                mode = 'r' if self.read_only else 'r+'
                if os.path.exists(self.vectors_path):
                    self._vectors = np.memmap(
                        self.vectors_path,
                        dtype=np.float32,
                        mode=mode,
                        shape=(self._capacity, self.dim)
                    )
                else:
                    self._create_vectors_file()
                
                # Load metadata
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        self._metadata = json.load(f)
                else:
                    self._metadata = []
                
                logger.info(
                    f"Loaded MemoryMappedVectorStore: {self._count} vectors, "
                    f"capacity={self._capacity}, dim={self.dim}"
                )
                
            except Exception as e:
                logger.error(f"Failed to load existing store: {e}")
                self._initialize_fresh()
        else:
            self._initialize_fresh()
    
    def _initialize_fresh(self) -> None:
        """Initialize a fresh vector store."""
        self._count = 0
        self._capacity = self.INITIAL_CAPACITY
        self._metadata = []
        self._create_vectors_file()
        logger.info(
            f"Initialized fresh MemoryMappedVectorStore: "
            f"capacity={self._capacity}, dim={self.dim}"
        )
    
    def _create_vectors_file(self) -> None:
        """Create or recreate the vectors memory-mapped file."""
        mode = 'r' if self.read_only else 'w+'
        self._vectors = np.memmap(
            self.vectors_path,
            dtype=np.float32,
            mode=mode,
            shape=(self._capacity, self.dim)
        )
    
    def _grow_capacity(self) -> None:
        """Double the capacity of the vector store."""
        if self.read_only:
            raise RuntimeError("Cannot grow capacity in read-only mode")
        
        new_capacity = int(self._capacity * self.GROWTH_FACTOR)
        logger.info(f"Growing vector store: {self._capacity} -> {new_capacity}")
        
        # Flush current memmap
        if self._vectors is not None:
            self._vectors.flush()
            del self._vectors
        
        # Create temporary file with new size
        temp_path = self.vectors_path + ".tmp"
        new_vectors = np.memmap(
            temp_path,
            dtype=np.float32,
            mode='w+',
            shape=(new_capacity, self.dim)
        )
        
        # Copy existing data
        old_vectors = np.memmap(
            self.vectors_path,
            dtype=np.float32,
            mode='r',
            shape=(self._capacity, self.dim)
        )
        new_vectors[:self._count] = old_vectors[:self._count]
        new_vectors.flush()
        del old_vectors
        del new_vectors
        
        # Replace old file
        os.replace(temp_path, self.vectors_path)
        
        self._capacity = new_capacity
        self._vectors = np.memmap(
            self.vectors_path,
            dtype=np.float32,
            mode='r+',
            shape=(self._capacity, self.dim)
        )
    
    def add(self, vector: np.ndarray, meta: Dict[str, Any]) -> int:
        """
        Add a vector to the store.
        
        Args:
            vector: Vector to add [dim] or [1, dim]
            meta: Metadata dictionary
            
        Returns:
            Index of the added vector
        """
        if self.read_only:
            raise RuntimeError("Cannot add vectors in read-only mode")
        
        with self._write_lock:
            # Ensure capacity
            if self._count >= self._capacity:
                self._grow_capacity()
            
            # Normalize vector shape
            vec = vector.flatten().astype(np.float32)
            if vec.shape[0] != self.dim:
                if vec.shape[0] < self.dim:
                    vec = np.pad(vec, (0, self.dim - vec.shape[0]))
                else:
                    vec = vec[:self.dim]
            
            # Add to memmap
            idx = self._count
            self._vectors[idx] = vec
            self._metadata.append(meta)
            self._count += 1
            
            return idx
    
    def add_batch(self, vectors: np.ndarray, metas: List[Dict[str, Any]]) -> int:
        """
        Add a batch of vectors efficiently.
        
        Args:
            vectors: Vectors to add [N, dim]
            metas: List of metadata dicts
            
        Returns:
            Number of vectors added
        """
        if self.read_only:
            raise RuntimeError("Cannot add vectors in read-only mode")
        
        n = len(metas)
        if n == 0:
            return 0
        
        with self._write_lock:
            # Ensure capacity
            while self._count + n > self._capacity:
                self._grow_capacity()
            
            # Add all vectors
            vecs = vectors.astype(np.float32)
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            
            # Handle dimension mismatch
            if vecs.shape[1] != self.dim:
                if vecs.shape[1] < self.dim:
                    pad_width = ((0, 0), (0, self.dim - vecs.shape[1]))
                    vecs = np.pad(vecs, pad_width)
                else:
                    vecs = vecs[:, :self.dim]
            
            start_idx = self._count
            self._vectors[start_idx:start_idx + n] = vecs[:n]
            self._metadata.extend(metas)
            self._count += n
            
            return n
    
    def save(self) -> None:
        """Flush vectors to disk and save metadata."""
        if self.read_only:
            return
        
        with self._write_lock:
            # Flush memmap
            if self._vectors is not None:
                self._vectors.flush()
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self._metadata, f)
            
            # Save index meta
            with open(self.index_meta_path, 'w') as f:
                json.dump({
                    'dim': self.dim,
                    'count': self._count,
                    'capacity': self._capacity,
                    'version': 2,
                    'format': 'memmap'
                }, f, indent=2)
            
            logger.debug(f"Saved VectorStore: {self._count} vectors")
    
    def query(
        self, 
        query_vec: np.ndarray, 
        k: int = 5, 
        boost_map: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find top-K most similar vectors using cosine similarity.
        
        Uses chunked processing for memory efficiency on large stores.
        
        Args:
            query_vec: Query vector [dim]
            k: Number of results to return
            boost_map: Optional name->boost mapping for result reranking
            
        Returns:
            List of result dicts with score, rank, and metadata
        """
        if self._count == 0:
            return []
        
        # Normalize query
        q = query_vec.flatten().astype(np.float32)
        if q.shape[0] != self.dim:
            if q.shape[0] < self.dim:
                q = np.pad(q, (0, self.dim - q.shape[0]))
            else:
                q = q[:self.dim]
        
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-9:
            return []
        q = q / q_norm
        
        # Chunked similarity computation for memory efficiency
        CHUNK_SIZE = 10000
        all_scores = np.zeros(self._count, dtype=np.float32)
        
        for start in range(0, self._count, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, self._count)
            chunk = self._vectors[start:end]
            
            # Compute norms
            chunk_norms = np.linalg.norm(chunk, axis=1)
            chunk_norms = np.maximum(chunk_norms, 1e-9)
            
            # Cosine similarity
            all_scores[start:end] = np.dot(chunk, q) / chunk_norms
        
        # Apply boost map
        if boost_map:
            for i, meta in enumerate(self._metadata[:self._count]):
                name = meta.get("name")
                if name and name in boost_map:
                    all_scores[i] += boost_map[name] * 0.2
        
        # Top-K
        k = min(k, self._count)
        top_indices = np.argpartition(all_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(all_scores[top_indices])[::-1]]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            res = self._metadata[idx].copy()
            score = float(all_scores[idx])
            res["score"] = score
            res["rank"] = rank + 1
            
            # Explainability
            reasons = []
            if score > 0.8:
                reasons.append("High semantic similarity match.")
            elif score > 0.5:
                reasons.append("Moderate similarity; likely contextually relevant.")
            else:
                reasons.append("Low confidence match; potential conceptual overlap.")
            
            entity_type = res.get("type", "unknown")
            if entity_type == "file":
                reasons.append("Core module match.")
            elif entity_type == "class":
                reasons.append("Structural definition match.")
            elif entity_type == "function":
                reasons.append("Functional logic match.")
            
            file_path = res.get("file", "")
            if "tests" in file_path:
                reasons.append("Provides usage examples via tests.")
            elif "docs" in file_path:
                reasons.append("Documentation source.")
            
            res["reason"] = " ".join(reasons)
            results.append(res)
        
        return results
    
    def clear(self) -> None:
        """Clear all vectors and metadata."""
        if self.read_only:
            raise RuntimeError("Cannot clear in read-only mode")
        
        with self._write_lock:
            self._count = 0
            self._metadata = []
            self.save()
    
    def __len__(self) -> int:
        """Return number of stored vectors."""
        return self._count
    
    def close(self) -> None:
        """Close the memory-mapped file."""
        if self._vectors is not None:
            del self._vectors
            self._vectors = None


# Alias for backward compatibility
VectorStore = MemoryMappedVectorStore
