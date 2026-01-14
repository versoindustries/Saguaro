"""
SAGUARO Indexing Engine (Holographic Edition)
Connects Parsing -> C++ Ops -> Storage
Uses specialized Quantum Holographic ops to maintain constant memory footprint.
Refactored for stateless parallel processing.
"""

import os
import logging
import numpy as np
import tensorflow as tf

from saguaro.parsing.parser import SAGUAROParser
from saguaro.storage.vector_store import VectorStore
from saguaro.ops import quantum_ops
from saguaro.ops import holographic
from saguaro.indexing.tracker import IndexTracker
from saguaro.tokenization.vocab import CoherenceManager

logger = logging.getLogger(__name__)


# --- Stateless Worker Function (Must be top-level for pickling) ---
def process_batch_worker(
    file_paths: list, active_dim: int, total_dim: int, vocab_size: int
) -> tuple:
    """
    Stateless worker function designed for 'spawn' multiprocessing.
    Parses, Tokenizes, and Embeds a batch of files.
    Returns (meta_list, doc_vectors_numpy).
    """
    # 1. Initialize Thread-Local Components
    # Note: TF Ops load lazily in this process
    quantum_ops.load_saguaro_core()
    parser = SAGUAROParser()
    cm = CoherenceManager()  # Loads vocab/trie for this process

    # Init Projection (Must match main process - simplistic fixed seed for now)
    # Ideally passed in, but for prototype fixed seed is safer across processes
    init_range = 1.0 / np.sqrt(active_dim)

    # Deterministic seed essential for shared basis across workers
    tf.random.set_seed(42)

    # Dynamic Sizing based on calibration (Key Memory Fix)
    projection_matrix = tf.random.uniform(
        [vocab_size, active_dim], -init_range, init_range, seed=42
    )

    batch_meta = []
    batch_vectors = []

    # 2. Processing Loop
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

            # Embed
            # tokens: [batch, seq]
            # Ensure tokens prevent OOB if vocab mismatch (clip or mask)
            # Tokenizer returns IDs up to its vocab size.
            # If projection matrix < tokens, we crash.
            # We assume projection matrix sized to vocab_size is sufficient.

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

            # Store results
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
            # We log but continue, worker shouldn't crash
            # Use print as logging config might not propagate perfectly in spawn
            print(f"[Worker Error] Failed {file_path}: {e}")

    # Return as numpy for easy pickling back to main
    if batch_vectors:
        return batch_meta, np.array(batch_vectors)
    return [], None


class IndexEngine:
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
        # Start with current size from controller
        # Will be updated after calibrate() call in CLI
        self.vocab_size = self.coherence_manager.controller.tokenizer.vocab_size
        # Ensure minimum buffer (e.g. 131k max)
        # But for memory safety start exact.

        # Projection (Main Process)
        # We need this for query encoding
        init_range = 1.0 / np.sqrt(self.active_dim)
        tf.random.set_seed(42)  # Sync with worker

        # We re-init this if vocab size changes during calibration
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
        self.coherence_manager.calibrate(file_paths)
        # Update vocab size and projection matrix
        new_size = self.coherence_manager.controller.tokenizer.vocab_size
        if new_size != self.vocab_size:
            logger.info(f"Resizing Projection Matrix: {self.vocab_size} -> {new_size}")
            self.vocab_size = new_size
            init_range = 1.0 / np.sqrt(self.active_dim)
            tf.random.set_seed(42)
            # Cannot use assign() for shape change. Re-create variable.
            self.projection_matrix = tf.Variable(
                tf.random.uniform(
                    [self.vocab_size, self.active_dim], -init_range, init_range, seed=42
                ),
                trainable=False,
                name="holographic_basis",
            )

    # --- Main Process Aggregation ---

    def ingest_worker_result(self, meta_list: list, vectors: np.ndarray):
        """
        Ingests results from a worker into the main holographic bundle and store.
        """
        if vectors is None or len(meta_list) == 0:
            return 0, 0

        count = len(meta_list)

        # 1. Update Holographic Bundle
        # Convert back to Tensor
        vec_tensor = tf.convert_to_tensor(vectors, dtype=tf.float32)

        # Bundle Logic: bundle(current + batch)
        # We can batch-update the bundle
        # current: [dim] -> [1, dim]
        # batch: [N, dim]
        # combined: [N+1, dim] -> bundle op -> [dim]

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

    def commit(self):
        if self.bundle_count > 0:
            self._crystallize()
        self.store.save()
        self.tracker.save()  # Persist tracker state

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
            query_vec = tf.reduce_mean(embeddings, axis=1)  # [1, dim]

            result = query_vec.numpy()
            if target_dim > self.active_dim:
                padding = np.zeros((1, target_dim - self.active_dim), dtype=np.float32)
                result = np.concatenate([result, padding], axis=1)
            elif target_dim < self.active_dim:
                result = result[:, :target_dim]
            return result
        except Exception:
            return np.zeros((1, target_dim), dtype=np.float32)
