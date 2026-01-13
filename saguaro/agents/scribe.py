
import os
import json
import logging
import time
from typing import Dict, Any, List
from saguaro.indexing.engine import IndexEngine
from saguaro.indexing.auto_scaler import get_repo_stats_and_config
from saguaro.tokenization.vocab import CoherenceManager

logger = logging.getLogger(__name__)

class Scribe:
    """
    The Generative Engine (Phase 2).
    Connects Perception (Index) to Generation (LLM).
    Now powered by HighNoon CodexRunner.
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.saguaro_dir = os.path.join(self.repo_path, ".saguaro")
        
        # Initialize components
        self.stats = get_repo_stats_and_config(self.repo_path)
        self.engine = IndexEngine(self.repo_path, self.saguaro_dir, self.stats)
        
        self.coherence = CoherenceManager()
        self.coherence.initialize()
        
        # Initialize HighNoon
        self.hn_model = None
        self.hn_runner = None
        self._init_highnoon()

    def _init_highnoon(self):
        """Initialize HighNoon components with fallback."""
        try:
            import highnoon as hn
            from highnoon.cli.runner import CodexRunner
            from highnoon.cli.manifest import ToolManifest
            
            logger.info("HighNoon Framework detected.")
            
            # Try to load model - simplistic approach for now
            # In a real scenario, we might check env vars or config
            try:
                # Placeholder for model loading
                # self.hn_model = hn.create_model("highnoon-3b-lite") 
                # For now, we default to None -> Dry Run mode unless configured
                pass
            except Exception as e:
                logger.warning(f"Could not load HighNoon model: {e}")
                
            # Create Manifest for Saguaro Tools
            self.manifest = ToolManifest()
            # We will register a 'submit_patch' tool that the agent can call
            from highnoon.cli.manifest import ToolRegistration
            self.manifest.register(ToolRegistration(
                name="submit_patch",
                handler=self._tool_submit_patch,
                description="Submit a code patch. Args: target_file, operations=[{op, content, ...}]"
            ))
            
            self.hn_runner = CodexRunner(
                model=self.hn_model,
                manifest=self.manifest,
                dry_run=(self.hn_model is None) # Auto-fallback to dry-run
            )
            
            if self.hn_model is None:
                logger.info("Scribe running in HighNoon DRY-RUN mode (no model loaded).")
                
        except ImportError:
            logger.warning("HighNoon framework not found. Scribe crippled.")
            self.hn_runner = None

    def _tool_submit_patch(self, target_file: str, operations: List[Dict[str, Any]], invocation_id: str = None) -> str:
        """
        Tool exposed to HighNoon agent to submit a patch.
        We capture this return value in generate_patch.
        """
        # Validate roughly
        return json.dumps({
            "status": "success", 
            "patch": {
                "target_file": target_file,
                "operations": operations
            }
        })

    def generate_patch(self, task_description: str, context_files: List[str] = None) -> Dict[str, Any]:
        """
        Generates a Semantic Patch using HighNoon.
        """
        # 1. Retrieve Context
        logger.info(f"Scribe retrieving context for: {task_description}")
        
        # Determine relevant files if not provided
        if not context_files:
            # Query the index to find files
            # For simplicity, we use the engine to encode and search (simulated here since we don't have scalar search handy in this class without EscalationLadder)
            # But we can use the CLI logic or just assume the user provides context for now.
             pass

        # ... (Context retrieval logic would go here) ...
        context_block = ""
        if context_files:
             context_block = f"Context Files: {context_files}\n"

        # 2. Tokenize / Prepare Prompt via Coherence (for stats mainly)
        vocab_size = getattr(self.coherence, 'vocab_size', 0)
        logger.info(f"Using Coherence Vocab (Size: {vocab_size})")
        
        # 3. Generate via HighNoon Runner
        if self.hn_runner:
            prompt = (
                f"Task: {task_description}\n"
                f"{context_block}\n"
                "Please analyze the task and context, then call the 'submit_patch' tool "
                "with the necessary operations to complete the task."
            )
            
            logger.info("Invoking HighNoon CodexRunner...")
            try:
                # The runner returns the FINAL text response. 
                # But we want the TOOL CALL payload. 
                # The CodexRunner as implemented in the inspection executes tools and injects results.
                # If we want to capture the generated patch, we need to intercept the tool call 
                # OR parse the result.
                
                # Hack: We can attach a 'captured_patch' to self temporarily or return it from the runner?
                # or simpler: The 'submit_patch' tool modifies a state we can read.
                self._latest_patch = None
                
                # We need to wrap the internal tool to capture the patch
                # Re-register with capture
                def capture_patch(args: Dict[str, Any]):
                    target_file = args.get("target_file")
                    operations = args.get("operations")
                    self._latest_patch = {
                        "target_file": target_file,
                        "operations": operations
                    }
                    return "Patch submitted successfully."
                
                from highnoon.cli.manifest import ToolRegistration
                self.hn_runner.manifest.register(ToolRegistration(
                    name="submit_patch", 
                    handler=capture_patch,
                    description="Submit a code patch."
                ))
                
                # If dry_run is True involved, CodexRunner._dry_run_generate makes a fake tool call.
                # We need to ensure that matches our expectaton if we want to test it?
                # The inspected CodexRunner._dry_run_generate creates a "run_unit_tests" call.
                # We might need to subclass or modify CodexRunner behavior for Scribe if dry_run logic is hardcoded.
                
                # For this integration, let's trust the runner. 
                # IF dry_run is active (no model), the standard runner returns a hardcoded "run_unit_tests".
                # This won't trigger our "submit_patch".
                # So for the "No Model" requirement, we must handle the simulation OURSELVES if runner is dry.
                
                if self.hn_runner.dry_run:
                     logger.info("Simulating patch generation (Dry Run)...")
                     # Manual fallback simulation since the Runner's dry-run is hardcoded for unit tests
                     return {
                        "target_file": context_files[0] if context_files else "generated.py",
                        "operations": [
                            {
                                "op": "insert",
                                "content": f"# Generated by Scribe (HighNoon DryRun) for task: {task_description}\n# Timestamp: {time.time()}\n\ndef generated_function():\n    pass\n"
                            }
                        ]
                    }

                # Real execution (if model existed)
                self.hn_runner.run(prompt)
                
                if self._latest_patch:
                    return self._latest_patch
                else:
                    logger.warning("Agent finished but produced no patch.")
                    return {}
                    
            except Exception as e:
                logger.error(f"HighNoon generation failed: {e}")
                return {}
        
        else:
             # Fallback if HighNoon is missing entirely
             logger.warning("HighNoon not available. Returning empty patch.")
             return {}

    def refine_patch(self, original_patch: Dict, feedback: str) -> Dict[str, Any]:
        """
        Refines a patch based on feedback (e.g. linter errors).
        """
        logger.info(f"Refining patch based on feedback: {feedback}")
        # Mock refinement
        new_patch = original_patch.copy()
        new_patch['operations'][0]['content'] += f"\n# Fixed: {feedback}\n"
        return new_patch
