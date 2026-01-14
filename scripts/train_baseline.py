#!/usr/bin/env python3
# scripts/train_baseline.py
# Copyright 2026 Verso Industries
#
# Train the Verso Baseline tokenizer codebook from curriculum datasets.
# This is OPTIONAL - Saguaro works without it using byte-level tokenization.
# The codebook improves tokenization efficiency by ~30-50%.
#
# Usage:
#   python scripts/train_baseline.py --curriculum verso-baseline
#   python scripts/train_baseline.py --corpus /path/to/code --fast

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saguaro.tokenization.intelligent_controller import IntelligentVocabController, VocabControllerConfig
from saguaro.data.dataset_loader import DatasetLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainBaseline")

def train_baseline(corpus_path: str = None, curriculum_name: str = None, output_path: str = "", fast: bool = False):
    """
    Train the Verso Baseline tokenizer.
    Can train from a local corpus path OR a HuggingFace curriculum.
    """
    if not corpus_path and not curriculum_name:
        logger.error("Must specify either --corpus or --curriculum")
        return

    logger.info("Initializing IntelligentVocabController...")
    
    # Config for a robust baseline
    config = VocabControllerConfig(
        model_max_length=8192,
        min_ngram_freq=50,       # High frequency for baseline
        sample_size=20_000,      # Matched to HighNoon standards (was 1M)
        min_ngram_size=2,
        max_ngram_size=5
    )
    
    controller = IntelligentVocabController(config)
    samples = []
    
    # Limit logic
    # Fast: 10MB, Full: 256MB default
    max_bytes = 10 * 1024 * 1024 if fast else 256 * 1024 * 1024 
    current_bytes = 0

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    # 1. Load Data
    if curriculum_name:
        logger.info(f"Loading Curriculum: {curriculum_name}")
        loader = DatasetLoader()
        
        logger.info(f"Streaming samples (Limit: {config.sample_size} samples)...")
        
        stream_iter = loader.stream_samples(curriculum_name, total_limit_bytes=max_bytes)
        if has_tqdm:
            stream_iter = tqdm(stream_iter, total=config.sample_size, unit='seq', desc="Streaming Dataset")
            
        for text in stream_iter:
             if len(samples) >= config.sample_size:
                 break
             samples.append(text)
             current_bytes += len(text)
             if has_tqdm:
                 stream_iter.update(1)
                 
    elif corpus_path:
        files = []
        logger.info(f"Scanning corpus at {corpus_path}...")
        for root, _, filenames in os.walk(corpus_path):
            for f in filenames:
                if f.endswith(('.py', '.js', '.ts', '.cpp', '.h', '.cc', '.md', '.java', '.go', '.rs')):
                    files.append(os.path.join(root, f))
        
        logger.info(f"Found {len(files)} source files.")
        if len(files) == 0:
            logger.error("No valid source files found!")
            return

        import random
        random.shuffle(files)
        
        logger.info(f"Reading samples (Limit: {max_bytes/1024/1024:.0f} MB)...")
        if has_tqdm:
            pbar = tqdm(total=max_bytes, unit='B', unit_scale=True, desc="Reading Corpus")
            
        for fpath in files:
            if current_bytes >= max_bytes:
                break
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    if len(text) > 100:
                        samples.append(text)
                        len_text = len(text)
                        current_bytes += len_text
                        if has_tqdm:
                            pbar.update(len_text)
            except Exception:
                pass
        if has_tqdm:
            pbar.close()
            
    logger.info(f"\nCollected {len(samples)} samples ({current_bytes/1024/1024:.2f} MB).")
    
    if not samples:
        logger.error("No samples collected. Aborting.")
        return

    # 3. Train
    logger.info("Starting training (this may take a while)...")
    
    train_callback = None
    if has_tqdm:
        # We don't know total superwords exactly, but we know sample count
        train_pbar = tqdm(total=len(samples), desc="Training Codebook", unit="seq")
        def _cb(current, total):
            train_pbar.update(current - train_pbar.n)
        train_callback = _cb
        
    controller.train_codebook(samples, min_freq=config.min_ngram_freq, progress_callback=train_callback)
    
    if has_tqdm:
        train_pbar.close()
    
    # 4. Save
    if not output_path:
        output_path = "saguaro/artifacts/codebooks/verso_baseline.json"
        
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    controller.save_codebook(output_path)
    logger.info(f"✅ Baseline saved to: {output_path}")
    logger.info(f"Vocab Stats: {controller.get_stats()}")

def main():
    parser = argparse.ArgumentParser(description="Train Saguaro Baseline Tokenizer")
    parser.add_argument("--corpus", help="Path to local corpus directory")
    parser.add_argument("--curriculum", help="Name of curriculum preset (e.g. verso-baseline)")
    parser.add_argument("--output", default="saguaro/artifacts/codebooks/verso_baseline.json", help="Output path")
    parser.add_argument("--fast", action="store_true", help="Fast mode (smaller sample)")
    
    args = parser.parse_args()
    
    train_baseline(args.corpus, args.curriculum, args.output, args.fast)

if __name__ == "__main__":
    main()
