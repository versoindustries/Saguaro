"""
SAGUARO Native Indexer Bindings

Python ctypes bindings for the native C API in _saguaro_core.so.
These functions can be called WITHOUT loading TensorFlow, reducing
worker memory from ~1.9GB to ~300MB.

Usage:
    from saguaro.ops.native_indexer import NativeIndexer
    indexer = NativeIndexer()
    tokens, lengths = indexer.tokenize_batch(["hello world"])
"""

import ctypes
import os
import logging
from typing import Optional, Tuple, List

import numpy as np
from numpy.ctypeslib import ndpointer

logger = logging.getLogger(__name__)

# Type aliases
c_int32_ptr = ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
c_float_ptr = ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')


class NativeIndexerError(Exception):
    """Error from native indexer operations."""
    pass


class NativeIndexer:
    """
    Native C++ indexer - NO TensorFlow required.
    
    Loads _saguaro_core.so and calls C functions directly via ctypes,
    bypassing TensorFlow entirely for massive memory savings.
    """
    
    _lib = None
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to avoid reloading the library."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._load_library()
        self._bind_functions()
        self._initialized = True
        
        # Check if native API is available
        if not self.is_available():
            raise NativeIndexerError(
                "Native API not available in _saguaro_core.so. "
                "Please rebuild with native_indexer_api.cc"
            )
        
        logger.info(f"Native indexer initialized (version: {self.version()})")
    
    def _find_library(self) -> str:
        """Find _saguaro_core.so in various locations."""
        # Current file is in saguaro/ops/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        saguaro_dir = os.path.dirname(current_dir)
        repo_dir = os.path.dirname(saguaro_dir)
        
        search_paths = [
            # PRIORITY: TF-free native lib (for worker processes)
            os.path.join(repo_dir, "build", "_saguaro_native.so"),
            os.path.join(saguaro_dir, "_saguaro_native.so"),
            os.path.join(repo_dir, "_saguaro_native.so"),
            "_saguaro_native.so",
            # FALLBACK: Full core lib (requires TF runtime)
            os.path.join(repo_dir, "build", "_saguaro_core.so"),
            os.path.join(saguaro_dir, "_saguaro_core.so"),
            os.path.join(repo_dir, "_saguaro_core.so"),
            "_saguaro_core.so",
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        raise NativeIndexerError(
            f"Could not find _saguaro_core.so. Searched: {search_paths}"
        )
    
    def _load_library(self):
        """Load the shared library."""
        if NativeIndexer._lib is not None:
            return
        
        lib_path = self._find_library()
        logger.debug(f"Loading native library from: {lib_path}")
        
        try:
            NativeIndexer._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise NativeIndexerError(f"Failed to load {lib_path}: {e}")
    
    def _bind_functions(self):
        """Bind C functions with proper type signatures."""
        lib = NativeIndexer._lib
        
        # Version / availability
        lib.saguaro_native_version.argtypes = []
        lib.saguaro_native_version.restype = ctypes.c_char_p
        
        lib.saguaro_native_available.argtypes = []
        lib.saguaro_native_available.restype = ctypes.c_int
        
        # Trie management
        lib.saguaro_native_trie_create.argtypes = []
        lib.saguaro_native_trie_create.restype = ctypes.c_void_p
        
        lib.saguaro_native_trie_destroy.argtypes = [ctypes.c_void_p]
        lib.saguaro_native_trie_destroy.restype = None
        
        lib.saguaro_native_trie_insert.argtypes = [
            ctypes.c_void_p,  # trie
            c_int32_ptr,      # ngram
            ctypes.c_int,     # ngram_len
            ctypes.c_int32,   # superword_id
        ]
        lib.saguaro_native_trie_insert.restype = None
        
        lib.saguaro_native_trie_build_from_table.argtypes = [
            ctypes.c_void_p,  # trie
            c_int32_ptr,      # offsets
            c_int32_ptr,      # tokens
            c_int32_ptr,      # superword_ids
            ctypes.c_int,     # num_ngrams
        ]
        lib.saguaro_native_trie_build_from_table.restype = None
        
        # Tokenization
        lib.saguaro_native_tokenize_batch.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),  # texts
            ctypes.POINTER(ctypes.c_int),     # text_lengths
            ctypes.c_int,                     # batch_size
            c_int32_ptr,                      # output_tokens
            c_int32_ptr,                      # output_lengths
            ctypes.c_int,                     # max_length
            ctypes.c_int,                     # byte_offset
            ctypes.c_int,                     # add_special_tokens
            ctypes.c_void_p,                  # trie
            ctypes.c_int,                     # num_threads
        ]
        lib.saguaro_native_tokenize_batch.restype = ctypes.c_int
        
        # Embedding lookup
        lib.saguaro_native_embed_lookup.argtypes = [
            c_int32_ptr,      # tokens
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # seq_len
            c_float_ptr,      # projection
            ctypes.c_int,     # vocab_size
            ctypes.c_int,     # dim
            c_float_ptr,      # output
        ]
        lib.saguaro_native_embed_lookup.restype = None
        
        # Document vectors
        lib.saguaro_native_compute_doc_vectors.argtypes = [
            c_float_ptr,      # embeddings
            c_int32_ptr,      # lengths
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # seq_len
            ctypes.c_int,     # dim
            c_float_ptr,      # output
        ]
        lib.saguaro_native_compute_doc_vectors.restype = None
        
        # Holographic bundling
        lib.saguaro_native_holographic_bundle.argtypes = [
            c_float_ptr,      # vectors
            ctypes.c_int,     # num_vectors
            ctypes.c_int,     # dim
            c_float_ptr,      # output
        ]
        lib.saguaro_native_holographic_bundle.restype = None
        
        lib.saguaro_native_crystallize.argtypes = [
            c_float_ptr,      # knowledge
            c_float_ptr,      # importance
            ctypes.c_int,     # num_vectors
            ctypes.c_int,     # dim
            ctypes.c_float,   # threshold
            c_float_ptr,      # output
        ]
        lib.saguaro_native_crystallize.restype = None
        
        # Full pipeline
        lib.saguaro_native_full_pipeline.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),  # texts
            ctypes.POINTER(ctypes.c_int),     # text_lengths
            ctypes.c_int,                     # batch_size
            c_float_ptr,                      # projection
            ctypes.c_int,                     # vocab_size
            ctypes.c_int,                     # dim
            ctypes.c_int,                     # max_length
            ctypes.c_void_p,                  # trie
            c_float_ptr,                      # output
            ctypes.c_int,                     # num_threads
        ]
        lib.saguaro_native_full_pipeline.restype = ctypes.c_int
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def version(self) -> str:
        """Get native library version."""
        return NativeIndexer._lib.saguaro_native_version().decode('utf-8')
    
    def is_available(self) -> bool:
        """Check if native API is available."""
        return NativeIndexer._lib.saguaro_native_available() == 1
    
    def create_trie(self) -> ctypes.c_void_p:
        """Create a new superword trie."""
        return NativeIndexer._lib.saguaro_native_trie_create()
    
    def destroy_trie(self, trie: ctypes.c_void_p):
        """Destroy a superword trie."""
        if trie:
            NativeIndexer._lib.saguaro_native_trie_destroy(trie)
    
    def trie_insert(self, trie: ctypes.c_void_p, ngram: np.ndarray, superword_id: int):
        """Insert an n-gram into the trie."""
        ngram = np.ascontiguousarray(ngram, dtype=np.int32)
        NativeIndexer._lib.saguaro_native_trie_insert(
            trie, ngram, len(ngram), superword_id
        )
    
    def tokenize_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        byte_offset: int = 32,
        add_special_tokens: bool = True,
        trie: Optional[ctypes.c_void_p] = None,
        num_threads: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of UTF-8 strings.
            max_length: Maximum sequence length.
            byte_offset: Byte offset (typically 32).
            add_special_tokens: Whether to add CLS/EOS tokens.
            trie: Optional superword trie for merging.
            num_threads: Number of threads (0 = auto).
            
        Returns:
            (tokens, lengths) where tokens is [batch_size, max_length]
            and lengths is [batch_size].
        """
        batch_size = len(texts)
        if batch_size == 0:
            return np.zeros((0, max_length), dtype=np.int32), np.zeros(0, dtype=np.int32)
        
        # Encode texts
        encoded = [t.encode('utf-8') for t in texts]
        text_ptrs = (ctypes.c_char_p * batch_size)(*encoded)
        text_lengths = (ctypes.c_int * batch_size)(*[len(t) for t in encoded])
        
        # Allocate outputs
        output_tokens = np.zeros((batch_size, max_length), dtype=np.int32)
        output_lengths = np.zeros(batch_size, dtype=np.int32)
        
        ret = NativeIndexer._lib.saguaro_native_tokenize_batch(
            text_ptrs,
            text_lengths,
            batch_size,
            output_tokens,
            output_lengths,
            max_length,
            byte_offset,
            1 if add_special_tokens else 0,
            trie,
            num_threads,
        )
        
        if ret != 0:
            raise NativeIndexerError(f"Tokenization failed with code {ret}")
        
        return output_tokens, output_lengths
    
    def embed_lookup(
        self,
        tokens: np.ndarray,
        projection: np.ndarray,
        vocab_size: int,
    ) -> np.ndarray:
        """
        Perform embedding lookup.
        
        Args:
            tokens: Token IDs [batch_size, seq_len].
            projection: Projection matrix [vocab_size, dim].
            vocab_size: Vocabulary size.
            
        Returns:
            Embeddings [batch_size, seq_len, dim].
        """
        tokens = np.ascontiguousarray(tokens, dtype=np.int32)
        projection = np.ascontiguousarray(projection, dtype=np.float32)
        
        batch_size, seq_len = tokens.shape
        dim = projection.shape[1]
        
        output = np.zeros((batch_size, seq_len, dim), dtype=np.float32)
        
        NativeIndexer._lib.saguaro_native_embed_lookup(
            tokens, batch_size, seq_len,
            projection, vocab_size, dim,
            output,
        )
        
        return output
    
    def compute_doc_vectors(
        self,
        embeddings: np.ndarray,
        lengths: np.ndarray,
    ) -> np.ndarray:
        """
        Compute document vectors via mean pooling.
        
        Args:
            embeddings: [batch_size, seq_len, dim].
            lengths: [batch_size].
            
        Returns:
            Document vectors [batch_size, dim].
        """
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        lengths = np.ascontiguousarray(lengths, dtype=np.int32)
        
        batch_size, seq_len, dim = embeddings.shape
        output = np.zeros((batch_size, dim), dtype=np.float32)
        
        NativeIndexer._lib.saguaro_native_compute_doc_vectors(
            embeddings, lengths,
            batch_size, seq_len, dim,
            output,
        )
        
        return output
    
    def holographic_bundle(
        self,
        vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Bundle multiple vectors into one.
        
        Args:
            vectors: [num_vectors, dim].
            
        Returns:
            Bundled vector [dim].
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        num_vectors, dim = vectors.shape
        output = np.zeros(dim, dtype=np.float32)
        
        NativeIndexer._lib.saguaro_native_holographic_bundle(
            vectors, num_vectors, dim, output,
        )
        
        return output
    
    def crystallize(
        self,
        knowledge: np.ndarray,
        importance: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Crystallize memory.
        
        Args:
            knowledge: [num_vectors, dim].
            importance: [num_vectors, dim].
            threshold: Crystallization threshold.
            
        Returns:
            Crystallized output [num_vectors, dim].
        """
        knowledge = np.ascontiguousarray(knowledge, dtype=np.float32)
        importance = np.ascontiguousarray(importance, dtype=np.float32)
        
        num_vectors, dim = knowledge.shape
        output = np.zeros_like(knowledge)
        
        NativeIndexer._lib.saguaro_native_crystallize(
            knowledge, importance,
            num_vectors, dim, threshold,
            output,
        )
        
        return output
    
    def full_pipeline(
        self,
        texts: List[str],
        projection: np.ndarray,
        vocab_size: int,
        max_length: int = 512,
        trie: Optional[ctypes.c_void_p] = None,
        num_threads: int = 0,
    ) -> np.ndarray:
        """
        Full pipeline: texts -> document vectors.
        
        Args:
            texts: List of UTF-8 strings.
            projection: Projection matrix [vocab_size, dim].
            vocab_size: Vocabulary size.
            max_length: Maximum sequence length.
            trie: Optional superword trie.
            num_threads: Number of threads (0 = auto).
            
        Returns:
            Document vectors [batch_size, dim].
        """
        batch_size = len(texts)
        if batch_size == 0:
            dim = projection.shape[1]
            return np.zeros((0, dim), dtype=np.float32)
        
        projection = np.ascontiguousarray(projection, dtype=np.float32)
        dim = projection.shape[1]
        
        # Encode texts
        encoded = [t.encode('utf-8') for t in texts]
        text_ptrs = (ctypes.c_char_p * batch_size)(*encoded)
        text_lengths = (ctypes.c_int * batch_size)(*[len(t) for t in encoded])
        
        # Allocate output
        output = np.zeros((batch_size, dim), dtype=np.float32)
        
        ret = NativeIndexer._lib.saguaro_native_full_pipeline(
            text_ptrs,
            text_lengths,
            batch_size,
            projection,
            vocab_size,
            dim,
            max_length,
            trie,
            output,
            num_threads,
        )
        
        if ret != 0:
            raise NativeIndexerError(f"Full pipeline failed with code {ret}")
        
        return output


# Global singleton instance
_indexer: Optional[NativeIndexer] = None


def get_native_indexer() -> NativeIndexer:
    """Get the global native indexer instance."""
    global _indexer
    if _indexer is None:
        _indexer = NativeIndexer()
    return _indexer


# Convenience functions
def tokenize_batch(texts: List[str], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenize texts without TensorFlow."""
    return get_native_indexer().tokenize_batch(texts, **kwargs)


def embed_and_pool(
    tokens: np.ndarray,
    lengths: np.ndarray,
    projection: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    """Embed tokens and compute document vectors."""
    indexer = get_native_indexer()
    embeddings = indexer.embed_lookup(tokens, projection, vocab_size)
    return indexer.compute_doc_vectors(embeddings, lengths)


def full_pipeline(texts: List[str], projection: np.ndarray, vocab_size: int, **kwargs) -> np.ndarray:
    """Full text -> document vector pipeline."""
    return get_native_indexer().full_pipeline(texts, projection, vocab_size, **kwargs)


def holographic_bundle(vectors: np.ndarray) -> np.ndarray:
    """Bundle vectors without TensorFlow."""
    return get_native_indexer().holographic_bundle(vectors)
