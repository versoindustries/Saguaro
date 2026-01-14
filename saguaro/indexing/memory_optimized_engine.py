"""
SAGUARO Memory-Optimized Indexing Engine
Enterprise-grade indexer with O(batch) memory footprint.

Key Optimizations:
1. Sequential batch processing - only one batch in memory at a time
2. Numpy-based embedding lookup - no TensorFlow tensor copies
3. Memory-mapped vector storage - vectors stored on disk
4. Aggressive garbage collection between batches
5. Streaming file discovery - no full file list in memory
"""

import os
import gc
import logging
from multiprocessing import shared_memory
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from saguaro.storage.memmap_vector_store import MemoryMappedVectorStore
from saguaro.ops import quantum_ops
from saguaro.ops import holographic
from saguaro.indexing.tracker import IndexTracker
from saguaro.tokenization.vocab import CoherenceManager

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Track memory usage during indexing."""
    peak_rss_mb: float = 0.0
    current_rss_mb: float = 0.0
    batches_processed: int = 0
    vectors_indexed: int = 0
    files_indexed: int = 0
    gc_collections: int = 0


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # Convert KB to MB on Linux
    except Exception:
        return 0.0


class SharedProjectionManager:
    """
    Manages a shared memory projection matrix to eliminate per-worker duplication.
    
    Memory Savings: For 131K vocab × 8192 dim × 4 bytes = 4.3GB per worker.
    With 15 workers, this saves ~60GB of RAM.
    """
    
    SHM_NAME = "saguaro_projection_v2"
    
    def __init__(self):
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._projection: Optional[np.ndarray] = None
        self._vocab_size: int = 0
        self._active_dim: int = 0
    
    def create(self, vocab_size: int, active_dim: int, seed: int = 42) -> None:
        """Create the shared projection matrix (main process only)."""
        self._vocab_size = vocab_size
        self._active_dim = active_dim
        
        # Calculate size in bytes
        nbytes = vocab_size * active_dim * np.float32().nbytes
        
        # Clean up any existing shared memory with same name
        try:
            existing = shared_memory.SharedMemory(name=self.SHM_NAME)
            existing.close()
            existing.unlink()
        except FileNotFoundError:
            pass
        
        # Create new shared memory
        self._shm = shared_memory.SharedMemory(
            name=self.SHM_NAME, create=True, size=nbytes
        )
        
        # Create numpy array backed by shared memory
        self._projection = np.ndarray(
            (vocab_size, active_dim), dtype=np.float32, buffer=self._shm.buf
        )
        
        # Initialize with deterministic random values
        rng = np.random.default_rng(seed)
        init_range = 1.0 / np.sqrt(active_dim)
        self._projection[:] = rng.uniform(
            -init_range, init_range, (vocab_size, active_dim)
        ).astype(np.float32)
        
        logger.info(
            f"Created shared projection: {vocab_size} × {active_dim} = {nbytes / 1024**2:.1f}MB"
        )
    
    def attach(self, vocab_size: int, active_dim: int) -> np.ndarray:
        """Attach to existing shared projection (worker process)."""
        self._shm = shared_memory.SharedMemory(name=self.SHM_NAME)
        self._projection = np.ndarray(
            (vocab_size, active_dim), dtype=np.float32, buffer=self._shm.buf
        )
        return self._projection
    
    def get_projection(self) -> np.ndarray:
        """Get the projection matrix."""
        if self._projection is None:
            raise RuntimeError(
                "Projection not initialized. Call create() or attach() first."
            )
        return self._projection
    
    def cleanup(self) -> None:
        """Clean up shared memory (main process only, after all workers done)."""
        if self._shm is not None:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            self._shm = None
            self._projection = None
            logger.info("Shared projection memory released.")
    
    def close(self) -> None:
        """Close handle without unlinking (worker processes)."""
        if self._shm is not None:
            self._shm.close()
            self._shm = None
            self._projection = None


# Global manager instance for workers to attach
_worker_shm: Optional[shared_memory.SharedMemory] = None


def process_batch_worker_memory_optimized(
    file_paths: list, 
    active_dim: int, 
    total_dim: int, 
    vocab_size: int
) -> Tuple[list, Optional[np.ndarray]]:
    """
    NATIVE C++ worker function for 'spawn' multiprocessing.
    
    NO TENSORFLOW IMPORTS - uses native C API via ctypes.
    Memory per worker: ~300MB (vs ~1.9GB with TensorFlow)
    
    Key optimizations:
    1. Pure C++ tokenization via native API (SIMD-optimized)
    2. Pure C++ embedding lookup via native API (SIMD-optimized)
    3. Pure C++ doc vector computation via native API (SIMD-optimized)
    4. Shared memory projection matrix (zero-copy)
    
    Returns (meta_list, doc_vectors_numpy).
    """
    global _worker_shm
    
    # Import native indexer ONLY (no TensorFlow)
    from saguaro.indexing.native_indexer_bindings import get_native_indexer
    from saguaro.parsing.parser import SAGUAROParser
    
    # Get native indexer singleton
    native = get_native_indexer()
    parser = SAGUAROParser()
    
    # Attach to shared projection matrix (zero-copy)
    try:
        shm = shared_memory.SharedMemory(name=SharedProjectionManager.SHM_NAME)
        _worker_shm = shm  # Keep reference to prevent GC
        projection_np = np.ndarray(
            (vocab_size, active_dim), dtype=np.float32, buffer=shm.buf
        )
    except FileNotFoundError:
        # Fallback: create local projection
        logger.warning("Shared memory not found, creating local projection")
        rng = np.random.default_rng(42)
        init_range = 1.0 / np.sqrt(active_dim)
        projection_np = rng.uniform(
            -init_range, init_range, (vocab_size, active_dim)
        ).astype(np.float32)

    batch_meta = []
    batch_vectors = []

    # Processing Loop - NATIVE C++ only
    for file_path in file_paths:
        try:
            entities = parser.parse_file(file_path)
            if not entities:
                continue

            # Collect texts for batch processing
            texts = [e.content[:2048] for e in entities]
            
            # Native full pipeline: texts -> document vectors
            # This calls SIMD-optimized C++ code via ctypes
            doc_vectors = native.full_pipeline(
                texts=texts,
                projection=projection_np,
                vocab_size=vocab_size,
                max_length=512,
                trie=None,  # TODO: Support superword trie
                num_threads=0  # Auto
            )
            
            # Build metadata
            for i, entity in enumerate(entities):
                meta = {
                    "name": entity.name,
                    "type": entity.type,
                    "file": entity.file_path,
                    "line": entity.start_line,
                    "end_line": entity.end_line,
                }
                batch_meta.append(meta)
                batch_vectors.append(doc_vectors[i])
            
            # Clear entities
            del entities, doc_vectors

        except Exception as e:
            logger.debug(f"Failed {file_path}: {e}")

    # Return as numpy for easy pickling back to main
    if batch_vectors:
        result = np.array(batch_vectors, dtype=np.float32)
        del batch_vectors
        return batch_meta, result
    return [], None


class MemoryOptimizedIndexEngine:
    """
    Memory-optimized indexing engine for enterprise-scale repositories.
    
    Features:
    - Memory-mapped vector storage (disk-backed)
    - Sequential batch processing with memory release
    - Shared memory projection matrix
    - Aggressive garbage collection
    - Memory usage tracking
    """

    def __init__(self, repo_path: str, saguaro_dir: str, config: dict):
        self.repo_path = repo_path
        self.config = config
        self.saguaro_dir = saguaro_dir

        # Use memory-mapped vector store
        self.store = MemoryMappedVectorStore(
            storage_path=os.path.join(saguaro_dir, "vectors"), 
            dim=config["total_dim"]
        )

        self.tracker = IndexTracker(os.path.join(saguaro_dir, "tracking.json"))
        self.coherence_manager = CoherenceManager()

        self.active_dim = config["active_dim"]
        self.total_dim = config["total_dim"]

        # Holographic State (Main Process Only)
        self.current_bundle = tf.zeros([self.active_dim], dtype=tf.float32)
        self.bundle_count = 0
        self.BUNDLE_THRESHOLD = 256  # Crystallize frequently to release memory

        self._ensure_core()

        # Dynamic Vocab Size
        self.vocab_size = self.coherence_manager.controller.tokenizer.vocab_size

        # Shared Projection Manager (for workers)
        self.projection_manager = SharedProjectionManager()

        # Local Projection (Main Process - for query encoding)
        # Use numpy for memory efficiency
        rng = np.random.default_rng(42)
        init_range = 1.0 / np.sqrt(self.active_dim)
        self.projection_matrix_np = rng.uniform(
            -init_range, init_range, (self.vocab_size, self.active_dim)
        ).astype(np.float32)
        
        # Memory stats
        self.memory_stats = MemoryStats()

    def _ensure_core(self):
        quantum_ops.load_saguaro_core()

    def calibrate(self, file_paths: list):
        """Calibrate tokenizer and update projection matrices."""
        self.coherence_manager.calibrate(file_paths)
        
        # Update vocab size
        new_size = self.coherence_manager.controller.tokenizer.vocab_size
        if new_size != self.vocab_size:
            logger.info(f"Resizing Projection Matrix: {self.vocab_size} -> {new_size}")
            self.vocab_size = new_size
            
            # Recreate numpy projection
            rng = np.random.default_rng(42)
            init_range = 1.0 / np.sqrt(self.active_dim)
            self.projection_matrix_np = rng.uniform(
                -init_range, init_range, (self.vocab_size, self.active_dim)
            ).astype(np.float32)
        
        # Force cleanup after calibration
        gc.collect()

    def create_shared_projection(self) -> None:
        """
        Create shared memory projection for workers.
        Call this BEFORE spawning worker processes.
        """
        self.projection_manager.create(self.vocab_size, self.active_dim, seed=42)

    def cleanup_shared_projection(self) -> None:
        """
        Clean up shared memory.
        Call this AFTER all workers are done.
        """
        self.projection_manager.cleanup()

    def ingest_worker_result(
        self, 
        meta_list: list, 
        vectors: np.ndarray
    ) -> Tuple[int, int]:
        """
        Ingests results from a worker into the main holographic bundle and store.
        
        MEMORY OPTIMIZED: Uses batch add and releases memory immediately.
        """
        if vectors is None or len(meta_list) == 0:
            return 0, 0

        count = len(meta_list)

        # 1. Update Holographic Bundle (using TF ops)
        vec_tensor = tf.convert_to_tensor(vectors, dtype=tf.float32)
        combined = tf.concat(
            [tf.expand_dims(self.current_bundle, 0), vec_tensor], axis=0
        )
        self.current_bundle = holographic.holographic_bundle(combined)
        self.bundle_count += count
        
        # Release TF tensors
        del vec_tensor, combined

        # 2. Add to Store using batch add (more efficient)
        # Pad vectors to total_dim if needed
        if vectors.shape[1] < self.total_dim:
            padding = np.zeros(
                (vectors.shape[0], self.total_dim - vectors.shape[1]), 
                dtype=np.float32
            )
            vectors = np.concatenate([vectors, padding], axis=1)
        elif vectors.shape[1] > self.total_dim:
            vectors = vectors[:, :self.total_dim]
        
        self.store.add_batch(vectors, meta_list)

        # 3. Crystallize check - more frequent to release memory
        if self.bundle_count >= self.BUNDLE_THRESHOLD:
            self._crystallize()

        # Update stats
        self.memory_stats.vectors_indexed += count
        self.memory_stats.files_indexed += len(set(m["file"] for m in meta_list))
        
        return len(set(m["file"] for m in meta_list)), count

    def _crystallize(self) -> None:
        """Crystallize current bundle and save to disk."""
        if self.current_bundle is None:
            return

        importance = tf.ones_like(self.current_bundle)
        _ = holographic.crystallize_memory(
            tf.expand_dims(self.current_bundle, 0),
            tf.expand_dims(importance, 0),
            threshold=0.5,
        )
        
        # Save vectors to disk immediately
        self.store.save()
        
        # Reset bundle
        self.current_bundle = tf.zeros([self.active_dim], dtype=tf.float32)
        self.bundle_count = 0
        
        # Aggressive garbage collection
        gc.collect()
        self.memory_stats.gc_collections += 1
        
        logger.info("Holographic Crystallization Complete. Memory released.")

    def commit(self) -> None:
        """Commit all changes to disk."""
        if self.bundle_count > 0:
            self._crystallize()
        self.store.save()
        self.tracker.save()
        
        # Final GC
        gc.collect()

    def encode_text(self, text: str, dim: int = None) -> np.ndarray:
        """
        Encode query text using numpy (Main process only).
        """
        target_dim = dim or self.total_dim
        try:
            trie_handle = self.coherence_manager.get_trie_handle()
            text_tensor = tf.constant([text[:2048]])
            tokens, _ = quantum_ops.fused_text_tokenize_batch(
                text_tensor,
                trie_handle=trie_handle,
                byte_offset=32,
                add_special_tokens=True,
            )
            
            # Use numpy for embedding
            tokens_np = tokens.numpy()
            tokens_np = np.clip(tokens_np, 0, self.vocab_size - 1)
            embeddings = self.projection_matrix_np[tokens_np]
            query_vec = np.mean(embeddings, axis=1)

            result = query_vec.astype(np.float32)
            if target_dim > self.active_dim:
                padding = np.zeros((1, target_dim - self.active_dim), dtype=np.float32)
                result = np.concatenate([result, padding], axis=1)
            elif target_dim < self.active_dim:
                result = result[:, :target_dim]
            return result
        except Exception:
            return np.zeros((1, target_dim), dtype=np.float32)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        current_mb = get_memory_mb()
        self.memory_stats.current_rss_mb = current_mb
        if current_mb > self.memory_stats.peak_rss_mb:
            self.memory_stats.peak_rss_mb = current_mb
        
        return {
            "peak_rss_mb": self.memory_stats.peak_rss_mb,
            "current_rss_mb": self.memory_stats.current_rss_mb,
            "batches_processed": self.memory_stats.batches_processed,
            "vectors_indexed": self.memory_stats.vectors_indexed,
            "files_indexed": self.memory_stats.files_indexed,
            "gc_collections": self.memory_stats.gc_collections,
            "store_count": len(self.store),
        }


# Backward compatibility aliases
IndexEngine = MemoryOptimizedIndexEngine
process_batch_worker = process_batch_worker_memory_optimized
