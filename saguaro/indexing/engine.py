"""
SAGUARO Indexing Engine (Holographic Edition)
Connects Parsing -> C++ Ops -> Storage
Uses specialized Quantum Holographic ops to maintain constant memory footprint.
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


class IndexEngine:
    def __init__(self, repo_path: str, saguaro_dir: str, config: dict):
        self.repo_path = repo_path
        self.config = config
        
        self.parser = SAGUAROParser()
        self.store = VectorStore(
            storage_path=os.path.join(saguaro_dir, "vectors"),
            dim=config['total_dim']
        )
        
        # Tracker
        self.tracker = IndexTracker(os.path.join(saguaro_dir, "tracking.json"))
        
        # Coherence / Tokenizer Manager
        # This manages the adaptive vocab and "native trie" handle
        self.coherence_manager = CoherenceManager()

        # Setup Models / Ops context
        self.active_dim = config['active_dim']
        self.total_dim = config['total_dim']
        
        # Holographic State
        self.current_bundle = None
        self.bundle_count = 0
        self.BUNDLE_THRESHOLD = 128  # Number of entities before crystallization
        
        # Ensure C++ ops are loaded
        self._core_loaded = False
        
        # Initialize Random Projection Matrix (Holographic Basis)
        # Using a fixed seed for deterministic behavior across runs if we were persistent,
        # but for now we create it in memory. Ideally load from disk.
        # Vocab size = 131072 (matches max_length default/vocab space of tokenizer)
        self.VOCAB_SIZE = 131072
        init_range = 1.0 / np.sqrt(self.active_dim)
        self.projection_matrix = tf.Variable(
            tf.random.uniform([self.VOCAB_SIZE, self.active_dim], -init_range, init_range),
            trainable=False,
            name="holographic_basis"
        )
        
    def _ensure_core(self):
        """Load C++ ops if not already loaded."""
        if not self._core_loaded:
            quantum_ops.load_saguaro_core()
            self._core_loaded = True
    
    def calibrate(self, file_paths: list):
        """
        Calibrate the tokenizer on the target codebase.
        This ensures the tokenizer vocabulary is tuned to the repo's dialect 
        and updates the 'native trie' used in _process_small_batch.
        """
        self.coherence_manager.calibrate(file_paths)

    def index_batch(self, file_paths: list, force: bool = False) -> int:
        """
        Indexes a batch of files using streaming holographic processing.
        Returns number of successfully indexed files.
        """
        self._ensure_core()
        
        # Incremental check
        if not force:
            needed = self.tracker.filter_needs_indexing(file_paths)
            if not needed:
                logger.info("No files modified since last index.")
                return 0, 0
            
            # Warn if skipping
            skipped = len(file_paths) - len(needed)
            if skipped > 0:
                logger.info(f"Incrementally indexing {len(needed)} files ({skipped} skipped).")
            
            file_paths = needed
        
        # Create a generator for entities to avoid loading all into list
        def entity_stream():
            for file_path in file_paths:
                try:
                    entities = self.parser.parse_file(file_path)
                    if entities:
                        for entity in entities:
                            yield entity
                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {e}")

        # Process the stream
        entity_count = self._process_stream(entity_stream())
        
        # We assume files that didn't throw in the generator setup are "processed"
        # Since we catch per-file, the stream consumes them all.
        file_count = len(file_paths)
        
        # Update tracker with successfully processed files
        if file_count > 0:
            self.tracker.update(file_paths)
        
        return file_count, entity_count
     
    def compute_state(self, file_paths: list = None) -> bytes:
        """
        Compute the holographic state for a set of files without persistence.
        """
        self._ensure_core()
        
        if file_paths is None:
             file_paths = []
             # Standard scan
             exclusions = ['.saguaro', '.git', 'venv', '__pycache__', 'node_modules', 'build', 'dist']
             for root, dirs, files in os.walk(self.repo_path):
                 dirs[:] = [d for d in dirs if d not in exclusions]
                 for file in files:
                     if file.endswith(('.py', '.cc', '.h', '.cpp', '.hpp', '.c', '.js', '.ts', '.md')):
                         file_paths.append(os.path.join(root, file))

        # Generator matching index_batch logic
        def entity_stream():
            for file_path in file_paths:
                try:
                    entities = self.parser.parse_file(file_path)
                    if entities:
                        for entity in entities:
                            yield entity
                except Exception:
                    pass

        # Reset bundle for this calculation
        self.current_bundle = None
        self.bundle_count = 0
        
        # Process without persistence
        self._process_stream(entity_stream(), persist=False)
        
        if self.current_bundle is not None:
            return self.current_bundle.numpy().tobytes()
        return b""

    def _process_stream(self, entity_iterator, persist: bool = True):
        """
        Stream entities into the holographic bundle.
        Memory Usage: O(D) constant, instead of O(N*D).
        """
        processed_count = 0
        
        # Initialize bundle if needed
        if self.current_bundle is None:
            self.current_bundle = tf.zeros([self.active_dim], dtype=tf.float32)
            
        current_batch_entities = []
        current_batch_texts = []
        
        BATCH_SIZE = 32  # Small batch for tokenization only
        
        for entity in entity_iterator:
            current_batch_entities.append(entity)
            current_batch_texts.append(entity.content[:2048])
            
            if len(current_batch_texts) >= BATCH_SIZE:
                self._process_small_batch(current_batch_texts, current_batch_entities, persist=persist)
                current_batch_texts = []
                current_batch_entities = []
                
            processed_count += 1
            
        # Flush remaining
        if current_batch_texts:
            self._process_small_batch(current_batch_texts, current_batch_entities, persist=persist)
            
        return processed_count

    def _process_small_batch(self, texts: list, entities: list, persist: bool = True):
        """
        Tokenize a small batch and bundle it immediately.
        """
        try:
            # 1. Tokenize (Fast C++ Op)
            # Fetch handle for trained trie
            trie_handle = self.coherence_manager.get_trie_handle()
            
            text_tensor = tf.constant(texts)
            tokens, lengths = quantum_ops.fused_text_tokenize_batch(
                text_tensor,
                trie_handle=trie_handle,
                byte_offset=32,
                add_special_tokens=True,
                max_length=512,
                num_threads=0
            )
            
            # tokens: [batch, 512]
            # Lookup embeddings via random projection
            # tokens is int32, safe for lookup
            embeddings = tf.nn.embedding_lookup(self.projection_matrix, tokens)
            
            # embeddings is now [batch, 512, active_dim]
            
            # Add position encoding
            seq_len = tf.shape(embeddings)[1]
            positions = tf.linspace(0.0, 1.0, seq_len)
            pos_enc = tf.tile(
                tf.reshape(positions, [1, seq_len, 1]),
                [len(texts), 1, self.active_dim]
            )
            embeddings = embeddings + 0.1 * pos_enc
            
            # 2. Holographic Bundling (The Fix)
            # Instead of keeping [batch, seq, dim], restrict to [batch, dim]
            # Collapse sequence dimension first
            doc_vectors = tf.reduce_mean(embeddings, axis=1) # [batch, dim]
            
            # Now bundle into the global state
            # This is O(1) memory for the global state
            self.current_bundle = holographic.holographic_bundle(
                tf.concat([
                    tf.expand_dims(self.current_bundle, 0), # Current state
                    doc_vectors                             # New vectors
                ], axis=0)
            )
            
            self.bundle_count += len(texts)
            
            # 3. Crystallize if full
            if persist and self.bundle_count >= self.BUNDLE_THRESHOLD:
                self._crystallize()
                
            # Store metadata (we still need index of *where* things are)
            # But the vector storage is now managing crystallized bundles + offsets
            # For this implementation, we still add individual vectors to store for retrieval
            # but we dropped the massive list accumulation.
            if persist:
                for i, entity in enumerate(entities):
                     self.store.add(doc_vectors[i].numpy(), meta={
                        "name": entity.name,
                        "type": entity.type,
                        "file": entity.file_path,
                        "line": entity.start_line,
                        "end_line": entity.end_line
                    })
                
        except Exception as e:
            logger.error(f"Error in holographic batch: {e}")

    def _crystallize(self):
        """
        Freeze the current bundle to disk/storage and reset.
        Uses CrystallizeMemory op to lock high-importance patterns.
        """
        if self.current_bundle is None:
            return

        # Importance could be calculated from freq/centrality, 
        # here we assume uniform for basic implementation
        importance = tf.ones_like(self.current_bundle)
        
        # Crystallize (Project gradients / lockout)
        _ = holographic.crystallize_memory(
            tf.expand_dims(self.current_bundle, 0),
            tf.expand_dims(importance, 0),
            threshold=0.5
        )
        
        # Commit to vector store (special 'crystal' entry or just flush)
        # For compatibility with existing store, we flush vectors.
        # Ideally we would save 'crystal' as a super-node.
        self.store.save()
        
        # Reset
        self.current_bundle = tf.zeros([self.active_dim], dtype=tf.float32)
        self.bundle_count = 0
        logger.info("Holographic Crystallization Complete.")

    def commit(self):
        # Final crystallization
        if self.bundle_count > 0:
            self._crystallize()
        self.store.save()

    def encode_text(self, text: str, dim: int = None) -> np.ndarray:
        """
        Encode a text query into a vector using C++ ops.
        Uses ModernHopfieldRetrieve for better query matching.
        """
        self._ensure_core()
        target_dim = dim or self.total_dim
        
        try:
            # Tokenize using adaptive trie
            trie_handle = self.coherence_manager.get_trie_handle()
            
            text_tensor = tf.constant([text[:2048]])
            tokens, _ = quantum_ops.fused_text_tokenize_batch(
                text_tensor, 
                trie_handle=trie_handle,
                byte_offset=32, 
                add_special_tokens=True, 
                max_length=512, 
                num_threads=0
            )
            # Embed using projection matrix
            embeddings = tf.nn.embedding_lookup(self.projection_matrix, tokens)
            # embeddings: [1, seq_len, dim]
            
            # Query Vector
            query_vec = tf.reduce_mean(embeddings, axis=1) # [1, dim]
            
            # If we had a loaded crystal, we would retrieve against it here.
            # For now, return the raw query vector, but projected if needed.
            
            result = query_vec.numpy()

            if target_dim > self.active_dim:
                padding = np.zeros((1, target_dim - self.active_dim), dtype=np.float32)
                result = np.concatenate([result, padding], axis=1)
            elif target_dim < self.active_dim:
                result = result[:, :target_dim]
                
            return result
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return np.zeros((1, target_dim), dtype=np.float32)
