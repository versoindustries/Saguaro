"""
SAGUARO Indexing Engine (Holographic Edition)
Connects Parsing -> C++ Ops -> Storage
Uses specialized Quantum Holographic ops to maintain constant memory footprint.
Refactored for stateless parallel processing with shared memory optimization.
"""

import os
import logging
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from saguaro.parsing.parser import SAGUAROParser
from saguaro.storage.vector_store import VectorStore
from saguaro.ops import quantum_ops
from saguaro.ops import holographic
from saguaro.indexing.tracker import IndexTracker
from saguaro.tokenization.vocab import CoherenceManager

logger = logging.getLogger(__name__)


def make_shm_name(repo_path: str, version: str = "v1") -> str:
    """Generate a repo-specific POSIX shared memory name.
    
    Prevents collisions when indexing multiple repos concurrently.
    POSIX shm names must be <= 255 chars and contain no slashes.
    """
    import hashlib
    path_hash = hashlib.md5(os.path.abspath(repo_path).encode()).hexdigest()[:12]
    return f"saguaro_proj_{version}_{path_hash}"


# --- Shared Memory Projection Manager ---
class SharedProjectionManager:
    """
    Manages a shared memory projection matrix to eliminate per-worker duplication.
    
    Memory Savings: For 131K vocab × 8192 dim × 4 bytes = 4.3GB per worker.
    With 15 workers, this saves ~60GB of RAM.
    """
    
    # Default name — static for compatibility with spawn-based multiprocessing.
    # Workers reference this class constant directly, so it cannot be overridden
    # per-instance (spawned processes import the module fresh).
    SHM_NAME = "saguaro_projection_v1"
    
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


def process_batch_worker(
    file_paths: list, active_dim: int, total_dim: int, vocab_size: int
) -> Tuple[list, Optional[np.ndarray]]:
    """
    Stateless worker function designed for 'spawn' multiprocessing.
    Parses, Tokenizes, and Embeds a batch of files.
    
    MEMORY OPTIMIZED: Uses shared memory projection matrix instead of creating
    a 4GB+ matrix per worker.
    
    Returns (meta_list, doc_vectors_numpy).
    """
    global _worker_shm
    
    # 1. Initialize Thread-Local Components
    quantum_ops.load_saguaro_core()
    parser = SAGUAROParser()
    cm = CoherenceManager()
    
    # 2. Attach to shared projection matrix (zero-copy!)
    try:
        shm = shared_memory.SharedMemory(name=SharedProjectionManager.SHM_NAME)
        _worker_shm = shm  # Keep reference to prevent GC
        projection_np = np.ndarray(
            (vocab_size, active_dim), dtype=np.float32, buffer=shm.buf
        )
        # Convert to TF tensor (this is a view, not a copy for read-only ops)
        projection_matrix = tf.constant(projection_np)
    except FileNotFoundError:
        # Fallback: create local projection (legacy behavior)
        print("[Worker Warning] Shared memory not found, creating local projection")
        init_range = 1.0 / np.sqrt(active_dim)
        tf.random.set_seed(42)
        projection_matrix = tf.random.uniform(
            [vocab_size, active_dim], -init_range, init_range, seed=42
        )

    batch_meta = []
    batch_vectors = []

    # 3. Processing Loop
    for file_path in file_paths:
        try:
            entities = parser.parse_file(file_path)
            if not entities:
                continue

            texts = [e.content[:2048] for e in entities]  # Truncate for safety

            # Tokenize
            trie_handle = cm.get_trie_handle()
            text_tensor = tf.constant(texts)
            tokens, lengths = quantum_ops.fused_text_tokenize_batch(
                text_tensor,
                trie_handle=trie_handle,
                byte_offset=32,
                add_special_tokens=True,
                max_length=512,
                num_threads=0,
            )

            # Clip tokens to vocab range to prevent OOB
            tokens = tf.clip_by_value(tokens, 0, vocab_size - 1)

            # Embed using shared projection
            embeddings = tf.nn.embedding_lookup(projection_matrix, tokens)

            # Positional Encoding
            seq_len = tf.shape(embeddings)[1]
            positions = tf.linspace(0.0, 1.0, seq_len)
            pos_enc = tf.tile(
                tf.reshape(positions, [1, seq_len, 1]), [len(texts), 1, active_dim]
            )
            embeddings = embeddings + 0.1 * pos_enc

            # Document Vectors [batch, dim]
            doc_vecs = tf.reduce_mean(embeddings, axis=1)
            vecs_np = doc_vecs.numpy()

            for i, entity in enumerate(entities):
                meta = {
                    "name": entity.name,
                    "type": entity.type,
                    "file": entity.file_path,
                    "line": entity.start_line,
                    "end_line": entity.end_line,
                }
                batch_meta.append(meta)
                batch_vectors.append(vecs_np[i])

        except Exception as e:
            print(f"[Worker Error] Failed {file_path}: {e}")

    # Return as numpy for easy pickling back to main
    if batch_vectors:
        return batch_meta, np.array(batch_vectors)
    return [], None


class IndexEngine:
    """Main indexing engine with shared memory projection support."""

    def __init__(self, repo_path: str, saguaro_dir: str, config: dict):
        self.repo_path = repo_path
        self.config = config

        self.store = VectorStore(
            storage_path=os.path.join(saguaro_dir, "vectors"), dim=config["total_dim"]
        )

        self.tracker = IndexTracker(os.path.join(saguaro_dir, "tracking.json"))
        self.coherence_manager = CoherenceManager()  # Main process instance

        self.active_dim = config["active_dim"]
        self.total_dim = config["total_dim"]

        # Holographic State (Main Process Only)
        self.current_bundle = tf.zeros([self.active_dim], dtype=tf.float32)
        self.bundle_count = 0
        self.BUNDLE_THRESHOLD = 512  # Increased for batch efficiency

        self._ensure_core()

        # Dynamic Vocab Size
        self.vocab_size = self.coherence_manager.controller.tokenizer.vocab_size

        # Shared Projection Manager (for workers)
        self.projection_manager = SharedProjectionManager()

        # Local Projection (Main Process - for query encoding)
        init_range = 1.0 / np.sqrt(self.active_dim)
        tf.random.set_seed(42)
        self.projection_matrix = tf.Variable(
            tf.random.uniform(
                [self.vocab_size, self.active_dim], -init_range, init_range, seed=42
            ),
            trainable=False,
            name="holographic_basis",
        )

    def _ensure_core(self):
        quantum_ops.load_saguaro_core()

    def calibrate(self, file_paths: list):
        """Calibrate tokenizer and update projection matrices."""
        self.coherence_manager.calibrate(file_paths)
        
        # Update vocab size and projection matrix
        new_size = self.coherence_manager.controller.tokenizer.vocab_size
        if new_size != self.vocab_size:
            logger.info(f"Resizing Projection Matrix: {self.vocab_size} -> {new_size}")
            self.vocab_size = new_size
            init_range = 1.0 / np.sqrt(self.active_dim)
            tf.random.set_seed(42)
            self.projection_matrix = tf.Variable(
                tf.random.uniform(
                    [self.vocab_size, self.active_dim], -init_range, init_range, seed=42
                ),
                trainable=False,
                name="holographic_basis",
            )

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

    # --- Main Process Aggregation ---

    def ingest_worker_result(self, meta_list: list, vectors: np.ndarray):
        """
        Ingests results from a worker into the main holographic bundle and store.
        """
        if vectors is None or len(meta_list) == 0:
            return 0, 0

        count = len(meta_list)

        # 1. Update Holographic Bundle
        vec_tensor = tf.convert_to_tensor(vectors, dtype=tf.float32)

        combined = tf.concat(
            [tf.expand_dims(self.current_bundle, 0), vec_tensor], axis=0
        )

        self.current_bundle = holographic.holographic_bundle(combined)
        self.bundle_count += count

        # 2. Add to Store
        for i, meta in enumerate(meta_list):
            self.store.add(vectors[i], meta=meta)

        # 3. Crystallize check
        if self.bundle_count >= self.BUNDLE_THRESHOLD:
            self._crystallize()

        return len(set(m["file"] for m in meta_list)), count

    def _crystallize(self):
        if self.current_bundle is None:
            return

        importance = tf.ones_like(self.current_bundle)
        _ = holographic.crystallize_memory(
            tf.expand_dims(self.current_bundle, 0),
            tf.expand_dims(importance, 0),
            threshold=0.5,
        )
        self.store.save()
        self.current_bundle = tf.zeros([self.active_dim], dtype=tf.float32)
        self.bundle_count = 0
        logger.info("Holographic Crystallization Complete.")

    def get_state(self) -> tf.Tensor:
        """Returns the current holographic bundle state."""
        return self.current_bundle

    def commit(self):
        if self.bundle_count > 0:
            self._crystallize()
        self.store.save()
        self.tracker.save()

    def encode_text(self, text: str, dim: int = None) -> np.ndarray:
        """
        Encode query text (Main process only).
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
            embeddings = tf.nn.embedding_lookup(self.projection_matrix, tokens)
            query_vec = tf.reduce_mean(embeddings, axis=1)

            result = query_vec.numpy()
            if target_dim > self.active_dim:
                padding = np.zeros((1, target_dim - self.active_dim), dtype=np.float32)
                result = np.concatenate([result, padding], axis=1)
            elif target_dim < self.active_dim:
                result = result[:, :target_dim]
            return result
        except Exception:
            return np.zeros((1, target_dim), dtype=np.float32)


# =============================================================================
# BACKWARD COMPATIBILITY: Import memory-optimized versions as defaults
# =============================================================================
# The memory-optimized engine is now the default for new indexing operations.
# The original classes above are kept for compatibility and fallback.
# =============================================================================

try:
    from saguaro.indexing.memory_optimized_engine import (
        MemoryOptimizedIndexEngine,
        process_batch_worker_memory_optimized,
    )
    
    # Use memory-optimized as default
    IndexEngine = MemoryOptimizedIndexEngine
    process_batch_worker = process_batch_worker_memory_optimized
    
except ImportError:
    # Fallback to original implementation if memory-optimized not available
    pass  # Original classes already defined above

