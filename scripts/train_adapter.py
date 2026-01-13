#!/usr/bin/env python3
# scripts/train_adapter.py
# SAGUARO Phase 2: Coherence (Repo-Specific Accuracy)

import os
import sys
import glob
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saguaro.ops.fused_text_tokenizer import (
    StreamingNgramCounterHandle,
    fused_text_tokenize_batch,
    ops_available
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("saguaro.train_adapter")

def train_adapter(repo_path: str, vocab_size: int = 1000):
    if not ops_available():
        logger.error("SAGUARO C++ Ops not available. Cannot train adapter.")
        return

    logger.info(f"Scanning repo: {repo_path}")
    
    # 1. Initialize Counter
    counter = StreamingNgramCounterHandle(min_ngram=2, max_ngram=8)
    
    # 2. Collect files
    # Simple recursive glob for code files
    extensions = ["py", "cc", "h", "js", "ts", "md", "txt"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(f"{repo_path}/**/*.{ext}", recursive=True))
    
    # Filter out venv, build, .git
    files = [f for f in files if "/venv/" not in f and "/build/" not in f and "/.git/" not in f]
    logger.info(f"Found {len(files)} files to scan.")
    
    batch_size = 32
    total_tokens = 0
    
    # 3. Process in batches
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        batch_texts = []
        
        for fpath in batch_files:
            try:
                with open(fpath, "rb") as f:
                    content = f.read()
                    # Skip large files > 1MB
                    if len(content) > 1_000_000:
                        continue
                    batch_texts.append(content)
            except Exception as e:
                logger.warning(f"Failed to read {fpath}: {e}")
                continue
        
        if not batch_texts:
            continue
            
        # Tokenize (base level, no trie)
        tokens, lengths = fused_text_tokenize_batch(batch_texts)
        
        # Count n-grams
        # Convert tensor to flat tokens and lengths
        # Ops expects (tokens_flat, lengths_flat)
        # But `fused_text_tokenize_batch` returns tokens [B, MAX], lengths [B]
        # We need to flatten relevant parts.
        # Actually, `count_batch` in python wrapper takes `tokens` and `lengths`.
        # However, `StreamingNgramCountOp` C++ expects 2D tokens?
        # Let's check `src/ops/fused_text_tokenizer_op.cc`:
        # REGISTER_OP("StreamingNgramCount").Input("tokens: int32").Input("lengths: int32")
        # Compute: tokens dims==2.
        # So yes, it expects [Batch, MaxLen].
        
        counter.count_batch(tokens, lengths)
        
        batch_total = int(sum(lengths.numpy()))
        total_tokens += batch_total
        print(f"\rProcessed {i+len(batch_texts)}/{len(files)} files ({total_tokens} tokens)...", end="")
        
    print("\nScanning complete.")
    
    # 4. Extract Top-K
    logger.info(f"Extracting top {vocab_size} n-grams...")
    ngrams_flat, counts, splits = counter.get_top_k(k=vocab_size, min_freq=5)
    
    # 5. Save/Print
    # We need to reconstruct n-grams from flat + splits
    num_ngrams = len(counts)
    logger.info(f"Found {num_ngrams} frequent n-grams.")
    
    ngrams_data = ngrams_flat.numpy()
    splits_data = splits.numpy()
    counts_data = counts.numpy()
    
    output_path = os.path.join(repo_path, "adapter.vocab")
    with open(output_path, "w") as f:
        f.write("# SAGUARO Adapter Vocab\n")
        f.write("# Frequency | N-gram (Bytes)\n")
        for idx in range(num_ngrams):
            start = splits_data[idx]
            end = splits_data[idx+1]
            ngram_ids = ngrams_data[start:end]
            
            # Convert back to bytes (assuming offset=32)
            # IDs < 256+32 are raw bytes.
            # IDs >= 256+32 are... wait, we only fed raw bytes (offset 32).
            # So all IDs should be byte tokens.
            ngram_bytes = bytes([b - 32 for b in ngram_ids if 32 <= b < 288])
            
            try:
                s = ngram_bytes.decode("utf-8")
                # Escape newlines for CSV
                s = s.replace("\n", "\\n")
                f.write(f"{counts_data[idx]}\t{s}\n")
            except Exception:
                # Binary sequence
                f.write(f"{counts_data[idx]}\t{ngram_bytes}\n")
                
    logger.info(f"Saved adapter vocab to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".", help="Repo path")
    parser.add_argument("--size", type=int, default=1000, help="Vocab size")
    args = parser.parse_args()
    
    train_adapter(args.path, args.size)
