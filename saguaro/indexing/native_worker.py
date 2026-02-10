"""
SAGUARO Native C++ Worker
Process isolation layer to prevent TensorFlow runtime conflicts.

This module MUST NOT import tensorflow.
It uses only:
- standard library (os, json, logging)
- numpy
- saguaro.ops.native_indexer (ctypes)
- saguaro.parsing.parser (tree-sitter)
"""

import os
import json
import logging
import numpy as np
from multiprocessing import shared_memory
from typing import Tuple, Optional

from saguaro.indexing.native_indexer_bindings import get_native_indexer
from saguaro.parsing.parser import SAGUAROParser

logger = logging.getLogger(__name__)

# Global worker state
_worker_shm = None
_native_indexer = None
_parser = None
_trie_handle = None

def _initialize_worker():
    """Initialize thread-local/process-local components."""
    global _native_indexer, _parser
    
    if _native_indexer is None:
        _native_indexer = get_native_indexer()
    
    if _parser is None:
        _parser = SAGUAROParser()

def _load_codebook_manual(path: str):
    """
    Manually load codebook JSON and build native trie.
    Avoids importing IntelligentVocabController (which imports TF usually).
    """
    global _trie_handle, _native_indexer
    
    if _trie_handle is not None:
        return _trie_handle

    try:
        if not os.path.exists(path):
            return None
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        superwords = data.get("superwords", [])
        if not superwords:
            return None
            
        # Extract table data
        offsets = [0]
        all_tokens = []
        ids = []
        
        current_offset = 0
        for entry in superwords:
            tokens = entry["token_ids"]
            all_tokens.extend(tokens)
            current_offset += len(tokens)
            offsets.append(current_offset)
            ids.append(entry["superword_id"])
            
        # Create trie via native API
        _trie_handle = _native_indexer.create_trie()
        
        # Build from table
        # We need to construct C-compatible arrays
        # native_indexer.py handles the ctypes conversion
        _native_indexer.build_trie_from_table(
            _trie_handle, 
            offsets, 
            all_tokens, 
            ids
        )
        
        return _trie_handle
        
    except Exception as e:
        logger.warning(f"Failed to load native codebook manually: {e}")
        return None

# Standard shared memory name - must match memory_optimized_engine.py
# When used standalone, defaults to "saguaro_projection_v2".
# The engine passes the repo-specific name via the shm_name parameter.
SHM_NAME = "saguaro_projection_v2"

def process_batch_worker_native(
    file_paths: list, 
    active_dim: int, 
    total_dim: int, 
    vocab_size: int,
    shm_name: str = SHM_NAME,
    codebook_path: Optional[str] = None
) -> Tuple[list, Optional[np.ndarray]]:
    """
    NATIVE C++ worker function.
    
    Args:
        file_paths: List of files to index
        active_dim: Embedding dimension
        total_dim: (Unused)
        vocab_size: Vocabulary size for projection
        shm_name: Name of shared memory block
        codebook_path: Path to codebook JSON (optional)
        
    Returns:
        (meta_list, doc_vectors_numpy)
    """
    global _worker_shm, _trie_handle
    
    _initialize_worker()
    
    # Load trie if needed
    trie = None
    if codebook_path:
        trie = _load_codebook_manual(codebook_path)
    
    # Attach to shared projection data
    try:
        # Only attach if not already attached
        if _worker_shm is None:
            shm = shared_memory.SharedMemory(name=shm_name)
            _worker_shm = shm  # Keep alive
        else:
            shm = _worker_shm
            
        projection_np = np.ndarray(
            (vocab_size, active_dim), dtype=np.float32, buffer=shm.buf
        )
    except Exception as e:
        logger.warning(f"Shared memory attach failed: {e}")
        # Fallback local projection
        rng = np.random.default_rng(42)
        init_range = 1.0 / np.sqrt(active_dim)
        projection_np = rng.uniform(
            -init_range, init_range, (vocab_size, active_dim)
        ).astype(np.float32)

    batch_meta = []
    batch_vectors = []
    
    # Process files
    for file_path in file_paths:
        try:
            entities = _parser.parse_file(file_path)
            if not entities:
                continue
                
            # Extract content
            texts = [e.content[:2048] for e in entities]
            
            # Run C++ pipeline
            # FORCE num_threads=1 to prevent nested parallelism explosion.
            # We are already running in a process pool of 15+ workers.
            # Spawning 16+ threads per worker would create ~240+ threads, causing contention and potential OOM/Stack issues.
            doc_vectors = _native_indexer.full_pipeline(
                texts=texts,
                projection=projection_np,
                vocab_size=vocab_size,
                max_length=512,
                trie=trie,
                num_threads=1  # FIXED: Safe single-threaded C++ execution per worker
            )
            
            # Map back to entities
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
                
        except Exception as e:
            logger.debug(f"Worker failed on {file_path}: {e}")
            
    if batch_vectors:
        result = np.array(batch_vectors, dtype=np.float32)
        return batch_meta, result
        
    return [], None


# Backward compatibility alias
# This allows cli.py to import this function by its old name
process_batch_worker_memory_optimized = process_batch_worker_native
