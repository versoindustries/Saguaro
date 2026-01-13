"""
SAGUARO DNI Server
JSON-RPC 2.0 implementation over Stdio.
Allows agents (like Claude/Antigravity) to query the index.
"""

import sys
import json
import logging
import traceback
import os

from saguaro.indexing.engine import IndexEngine
from saguaro.storage.vector_store import VectorStore

# Configure logging to file so we don't pollute stdout (which is for JSON-RPC)
logging.basicConfig(
    filename='saguaro_dni.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DNIServer:
    def __init__(self):
        self.engine = None
        self.store = None
        self.session = None
        
    def handle_request(self, line: str):
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            self.send_error(None, -32700, "Parse error")
            return

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        try:
            if method == "initialize":
                result = self.initialize(params)
            elif method == "index":
                result = self.index(params)
            elif method == "query":
                result = self.query(params)
            elif method == "status":
                result = self.status(params)
            elif method == "read_node":
                result = self.read_node(params)
            else:
                self.send_error(req_id, -32601, "Method not found")
                return
                
            self.send_result(req_id, result)
            
        except Exception as e:
            logger.error(f"Error handling {method}: {traceback.format_exc()}")
            self.send_error(req_id, -32603, f"Internal error: {str(e)}")

    def initialize(self, params):
        repo_path = params.get("path", ".")
        self.saguaro_dir = os.path.join(repo_path, ".saguaro")
        
        # Load config
        config_path = os.path.join(self.saguaro_dir, "config.yaml")
        # Defaults
        config = {
            "active_dim": 8192, 
            "total_dim": 16384
        }
        
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                    # Update config with file values if present
                    if 'indexing' in file_config:
                        # If indexing block exists, might contain dims?
                        # auto_scaler produces dict with 'active_dim', 'total_dim' at top level
                        pass
                    
                    # We expect top level or under 'indexing'?
                    # Let's support both for robustness or check how we save it.
                    # For now, we will save at top level in CLI.
                    if 'active_dim' in file_config:
                        config['active_dim'] = file_config['active_dim']
                    if 'total_dim' in file_config:
                        config['total_dim'] = file_config['total_dim']
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        # Determine dimensions (try to load from config if exists, else defaults)
        # For prototype we assume standard defaults if not found
        
        self.store = VectorStore(
            storage_path=os.path.join(self.saguaro_dir, "vectors"),
            dim=config['total_dim']
        )
        
        # Initialize Engine purely for encoding utility (lightweight mode)
        # We don't want to re-index, just use its helpers
        self.engine = IndexEngine(repo_path, self.saguaro_dir, config)
        
        # Initialize Session if ID provided
        session_id = params.get("session_id")
        if session_id:
            from saguaro.memory.session import AgentSession
            self.session = AgentSession(session_id, repo_path)
        
        return {
            "status": "ready", 
            "repo": repo_path, 
            "vectors": len(self.store.vectors),
            "session": session_id if self.session else None
        }

    def index(self, params):
        # Trigger re-index
        # Would require instantiating Engine
        return {"status": "not_implemented_in_prototype"}

    def query(self, params):
        text = params.get("text")
        k = params.get("k", 5)
        
        if not self.store or not self.engine:
            raise RuntimeError("Server not initialized")
            
        # 1. Encode Query
        # We use the same pipeline: Text -> QWT/Mock -> Vector
        # We get [S, D] tokens from encode_text
        # Ensure we request total_dim so it matches store
        tokens = self.engine.encode_text(text, dim=self.store.dim)
        
        # Collapse to single query vector (Mean Pool)
        # [S, D] -> [D]
        import tensorflow as tf
        query_vec = tf.reduce_mean(tokens, axis=0).numpy()
        
        # 2. Search
        results = self.store.query(query_vec, k=k)
        
        # 3. Memory
        if self.session:
            self.session.add_interaction(text, results)
            
        return {"results": results}

    def read_node(self, params):
        """
        Reads source code for a node/file.
        Params:
            - file (str): Absolute path to file
            - start_line (int, optional): 1-indexed start
            - end_line (int, optional): 1-indexed end
        """
        file_path = params.get("file")
        start = params.get("start_line")
        end = params.get("end_line")
        
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
        if start is not None and end is not None:
            # Adjust for 0-indexing
            # start_line 1 -> index 0
            start_idx = max(0, start - 1)
            end_idx = min(len(lines), end)
            content = "".join(lines[start_idx:end_idx])
            range_info = f"{start}-{end}"
        else:
            content = "".join(lines)
            range_info = "all"
            
        return {
            "file": file_path,
            "range": range_info,
            "content": content,
            "loc": len(lines)
        }

    def status(self, params):
        return {
            "state": "running", 
            "indexed_docs": len(self.store.vectors) if self.store else 0,
            "session_active": bool(self.session)
        }

    def send_result(self, req_id, result):
        res = {"jsonrpc": "2.0", "result": result, "id": req_id}
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

    def send_error(self, req_id, code, message):
        res = {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": req_id}
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

def main():
    server = DNIServer()
    logger.info("SAGUARO DNI Server Started")
    
    # Simple line-based JSON-RPC reader
    for line in sys.stdin:
        if not line:
            break
        server.handle_request(line.strip())

if __name__ == "__main__":
    main()
