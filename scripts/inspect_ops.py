
import inspect
from saguaro.ops.quantum_ops import load_saguaro_core

def inspect_ops():
    mod = load_saguaro_core()
    op = getattr(mod, "quantum_embedding_forward", None)
    if not op:
         op = getattr(mod, "QuantumEmbeddingForward", None)
    
    if op:
        print(f"Op: {op}")
        try:
            print(f"Sig: {inspect.signature(op)}")
        except Exception as e:
            print(f"Could not inspect: {e}")
            
    # Also check via tf.raw_ops if possible (requires graph mode context to see generic sig?)
    # best is to inspect the wrapper function
    
if __name__ == "__main__":
    inspect_ops()
