# saguaro/tokenization/train_baseline.py
# Copyright 2026 Verso Industries

import os
import json
import logging
import random
from typing import Optional, List, Dict, Any
from pathlib import Path

from saguaro.tokenization.intelligent_controller import IntelligentVocabController, VocabControllerConfig

logger = logging.getLogger(__name__)

# Try to import datasets (HuggingFace)
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

def train_baseline(
    corpus_path: Optional[str] = None,
    curriculum_name: str = "verso-baseline",
    output_path: Optional[str] = None,
    fast_mode: bool = False
):
    """
    Train a baseline codebook using either a local corpus OR a HuggingFace curriculum.
    
    Args:
        corpus_path: Path to local corpus (optional).
        curriculum_name: Name of the preset in curriculum_presets.json (default: 'verso-baseline').
        output_path: Path to save the trained codebook (default: artifacts/codebooks/verso_baseline.json).
        fast_mode: If true, drastically reduces sample size for testing.
    """
    
    samples = []
    
    # 1. Load from Curriculum (Priority)
    if HAS_DATASETS:
        print(f"Loading curriculum preset: {curriculum_name}")
        try:
            # Find presets file
            base_dir = Path(__file__).parent.parent.parent # saguaro root
            presets_path = base_dir / "saguaro" / "artifacts" / "curriculum_presets.json"
            
            if presets_path.exists():
                with open(presets_path, 'r') as f:
                    presets = json.load(f).get('presets', {})
                
                preset = presets.get(curriculum_name)
                if preset:
                    hf_datasets = preset.get('hf_datasets', [])
                    print(f"Found {len(hf_datasets)} datasets in preset.")
                    
                    # Limit samples per dataset
                    samples_per_dataset = 500 if fast_mode else 5000
                    
                    for ds_name in hf_datasets:
                        try:
                            print(f"  Streaming from {ds_name}...")
                            # Use streaming to avoid downloading TBs of data
                            dataset = load_dataset(ds_name, split="train", streaming=True)
                            
                            count = 0
                            for item in dataset:
                                if count >= samples_per_dataset:
                                    break
                                
                                # Extract text content (heuristic for common column names)
                                text = item.get('content') or item.get('text') or item.get('code')
                                
                                if text:
                                    # Truncate enormous files
                                    samples.append(text[:100000])
                                    count += 1
                            print(f"    Collected {count} samples.")
                            
                        except Exception as e:
                            print(f"    Failed to load {ds_name}: {e}")
                else:
                    print(f"Warning: Preset '{curriculum_name}' not found.")
            else:
                print(f"Warning: Presets file not found at {presets_path}")
                
        except Exception as e:
             print(f"Error loading curriculum: {e}")
    else:
        print("Warning: 'datasets' library not found. Skipping HuggingFace curriculum.")

    # 2. Augment with Local Corpus (if provided)
    if corpus_path:
        corpus_path = os.path.abspath(corpus_path)
        if os.path.exists(corpus_path):
            print(f"Scanning local corpus: {corpus_path}")
            local_samples = []
            files = []
            for root, _, filenames in os.walk(corpus_path):
                for filename in filenames:
                    if filename.endswith(('.py', '.cc', '.h', '.md', '.txt')):
                        files.append(os.path.join(root, filename))
            
            random.shuffle(files)
            
            # Local limit
            byte_limit = 5 * 1024 * 1024 if fast_mode else 50 * 1024 * 1024
            total_bytes = 0
            
            for fpath in files:
                if total_bytes >= byte_limit:
                    break
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(100000)
                        if content:
                            local_samples.append(content)
                            total_bytes += len(content)
                except Exception:
                    pass
            print(f"Collected {len(local_samples)} local samples.")
            samples.extend(local_samples)
        else:
            print(f"Warning: Local corpus path {corpus_path} does not exist.")

    if not samples:
        print("Error: No samples collected from curriculum or local corpus.")
        return

    print(f"Total training samples: {len(samples)}")

    # Configure Controller
    config = VocabControllerConfig(
        min_ngram_freq=10, 
        sample_size=len(samples),
        min_ngram_size=2,
        max_ngram_size=5
    )
    
    controller = IntelligentVocabController(config=config)
    
    print("Training Codebook (this may take a while)...")
    try:
        learned_count = controller.train_codebook(samples)
        print(f"Training complete. Learned {learned_count} superwords.")
        
        # Determine output path
        if output_path:
            target_path = output_path
        else:
            # Default location
            base_dir = Path(__file__).parent.parent.parent
            target_path = base_dir / "saguaro" / "artifacts" / "codebooks" / "verso_baseline.json"
            
        print(f"Saving to: {target_path}")
        controller.save_codebook(target_path)
        print("Done.")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
