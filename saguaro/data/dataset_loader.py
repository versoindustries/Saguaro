# saguaro/data/dataset_loader.py
# Copyright 2026 Verso Industries

import logging
import json
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Adapts HuggingFace Datasets for Saguaro Tokenizer Training.
    Handles streaming from the 'Verso Baseline' curriculum.
    """
    
    # Datasets requiring specific configurations to avoid hanging on massive metadata
    # Format: "dataset_name": ["config1", "config2", ...] or "dataset_name": "single_config"
    DATASET_CONFIGS = {
        # The Stack v2 requires language subset - uses PascalCase config names
        # See full list: https://huggingface.co/datasets/bigcode/the-stack-v2-dedup
        "bigcode/the-stack-v2-dedup": [
            "Python", "JavaScript", "TypeScript", "Go", "Rust", "Java",
            "C", "C++", "C-Sharp",  # Core systems languages
            "JSX", "TSX",  # React
            "Shell", "Bash", "PowerShell",  # Shell scripting
            "SQL", "PLSQL",  # Database
            "Ruby", "PHP", "Scala", "Kotlin", "Swift",  # Other popular
            "HTML", "CSS", "SCSS",  # Web fundamentals
        ],
        "bigcode/the-stack-v2": [
            "Python", "JavaScript", "TypeScript", "Go", "Rust", "Java",
            "C", "C++", "C-Sharp", "JSX", "TSX", "Shell",
        ],
        "bigcode/starcoderdata": "python",
        # code_contests was renamed
        "google/code_contests": None,  # No config needed
        "deepmind/code_contests": None,  # Alias
        # Code search net needs language
        "code-search-net/code_search_net": ["python", "javascript", "go", "java", "ruby", "php"],
    }
    
    # Text field mapping for datasets with non-standard field names
    TEXT_FIELDS = {
        "bigcode/the-stack-v2-dedup": "content",
        "bigcode/the-stack-v2": "content", 
        "bigcode/starcoderdata": "content",
        "code-search-net/code_search_net": "func_code_string",
        "google/code_contests": "description",
        "deepmind/code_contests": "description",
        "lighteval/MATH": "problem",
        "meta-math/MetaMathQA": "query",
        "microsoft/orca-math-word-problems-200k": "question",
        "TIGER-Lab/MathInstruct": "instruction",
        "math-ai/StackMathQA": "question",
        "open-r1/Math-Verify": "problem",
        "HuggingFaceH4/ultrafeedback_binarized": "prompt",
        "Salesforce/xlam-function-calling-60k": "query",
        "HuggingFaceH4/ultrachat_200k": "prompt",
        "OpenAssistant/oasst_top1_2023-08-25": "text",
    }
    
    def __init__(self, curriculum_path: Optional[str] = None):
        if not curriculum_path:
            # Default to artifacts/curriculum_presets.json relative to this file
            base_dir = Path(__file__).parent.parent
            curriculum_path = base_dir / "artifacts" / "curriculum_presets.json"
            
        self.curriculum_path = Path(curriculum_path)
        self.presets = {}
        self._load_curriculum()
        
    def _load_curriculum(self):
        if not self.curriculum_path.exists():
            logger.warning(f"Curriculum file not found: {self.curriculum_path}")
            return
            
        try:
            with open(self.curriculum_path, 'r') as f:
                data = json.load(f)
                self.presets = data.get("presets", {})
        except Exception as e:
            logger.error(f"Failed to load curriculum: {e}")

    def _get_text_from_item(self, ds_name: str, item: dict) -> Optional[str]:
        """Extract text from dataset item using known field mappings."""
        # Check if we have a known text field for this dataset
        if ds_name in self.TEXT_FIELDS:
            field = self.TEXT_FIELDS[ds_name]
            text = item.get(field)
            if text:
                return text if isinstance(text, str) else str(text)
        
        # Fallback to common field names
        for field in ["text", "content", "code", "prompt", "question", "instruction"]:
            if field in item and item[field]:
                val = item[field]
                return val if isinstance(val, str) else str(val)
        
        return None

    def _stream_to_queue(self, ds_name: str, config: Optional[str], 
                          budget_bytes: int, queue, stop_event, load_dataset_fn):
        """Stream from a single dataset into a queue (for parallel streaming)."""
        config_str = f" (config: {config})" if config else ""
        logger.info(f"  → Streaming: {ds_name}{config_str}...")
        
        try:
            if config:
                ds = load_dataset_fn(ds_name, config, split="train", streaming=True, trust_remote_code=True)
            else:
                ds = load_dataset_fn(ds_name, split="train", streaming=True, trust_remote_code=True)
            
            ds_bytes = 0
            for item in ds:
                if stop_event.is_set() or ds_bytes >= budget_bytes:
                    break
                
                text = self._get_text_from_item(ds_name, item)
                if not text or len(text) < 50:
                    continue
                
                queue.put(text)
                ds_bytes += len(text)
                
        except Exception as e:
            logger.warning(f"  ✗ Failed {ds_name}{config_str}: {e}")
        finally:
            queue.put(None)  # Signal completion

    def stream_samples(self, preset_name: str, total_limit_bytes: int = 100_000_000,
                       max_workers: int = 8) -> Iterator[str]:
        """
        Stream text samples from the specified curriculum preset using parallel I/O.
        
        Uses ThreadPoolExecutor to stream from multiple datasets concurrently,
        overlapping network I/O to significantly reduce wall-clock time.
        
        Args:
            preset_name: Name of preset in curriculum (e.g., 'verso-baseline')
            total_limit_bytes: Soft limit on total bytes to yield
            max_workers: Maximum number of concurrent dataset streams (default: 8)
        
        Yields:
            Text strings
        """
        import queue
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        if preset_name not in self.presets:
            logger.error(f"Unknown preset: {preset_name}")
            return
            
        hf_datasets = self.presets[preset_name].get("hf_datasets", [])
        if not hf_datasets:
            logger.warning(f"No HF datasets defined for {preset_name}")
            return
            
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("HuggingFace 'datasets' library not installed.")
            logger.error("Run: pip install datasets")
            return

        # Expand datasets with configs into individual streams
        expanded_streams = []
        for ds_name in hf_datasets:
            if ds_name in self.DATASET_CONFIGS:
                configs = self.DATASET_CONFIGS[ds_name]
                if configs is None:
                    expanded_streams.append((ds_name, None))
                elif isinstance(configs, list):
                    for cfg in configs:
                        expanded_streams.append((ds_name, cfg))
                else:
                    expanded_streams.append((ds_name, configs))
            else:
                expanded_streams.append((ds_name, None))

        num_streams = len(expanded_streams)
        logger.info(f"Parallel streaming from {num_streams} streams ({len(hf_datasets)} datasets) with {min(max_workers, num_streams)} workers...")
        
        budget_per_stream = total_limit_bytes // max(num_streams, 1)
        
        # Create queues and stop event for each stream
        queues = [queue.Queue(maxsize=100) for _ in range(num_streams)]
        stop_event = threading.Event()
        
        current_bytes = 0
        active_streams = num_streams
        
        # Start parallel streaming
        with ThreadPoolExecutor(max_workers=min(max_workers, num_streams)) as executor:
            # Submit all stream tasks
            futures = []
            for i, (ds_name, config) in enumerate(expanded_streams):
                future = executor.submit(
                    self._stream_to_queue,
                    ds_name, config, budget_per_stream,
                    queues[i], stop_event, load_dataset
                )
                futures.append(future)
            
            # Round-robin collect from queues
            stream_done = [False] * num_streams
            
            while active_streams > 0 and current_bytes < total_limit_bytes:
                collected_this_round = False
                
                for i in range(num_streams):
                    if stream_done[i]:
                        continue
                    
                    try:
                        # Non-blocking get with small timeout
                        text = queues[i].get(timeout=0.01)
                        
                        if text is None:
                            # Stream finished
                            stream_done[i] = True
                            active_streams -= 1
                            continue
                        
                        yield text
                        current_bytes += len(text)
                        collected_this_round = True
                        
                        if current_bytes >= total_limit_bytes:
                            stop_event.set()
                            break
                            
                    except queue.Empty:
                        continue
                
                # Small sleep if no data collected to avoid busy-waiting
                if not collected_this_round and active_streams > 0:
                    import time
                    time.sleep(0.01)
            
            # Signal all streams to stop
            stop_event.set()

        logger.info(f"Finished parallel streaming. Total bytes: {current_bytes:,}")
