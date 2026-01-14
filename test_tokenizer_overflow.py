
import tensorflow as tf
import saguaro.ops.fused_text_tokenizer as tok
import numpy as np

def test_overflow():
    print("Testing tokenizer with large input to trigger potential heap corruption...")
    
    # Create a string larger than default max_length (128KB)
    # 200KB string
    large_text = "a" * (200 * 1024) 
    
    texts = [large_text, "short text"]
    
    try:
        print(f"Tokenizing batch with input size {len(large_text)}...")
        tokens, lengths = tok.fused_text_tokenize_batch(
            texts,
            byte_offset=32,
            add_special_tokens=True,
            max_length=131072, # Default max
            num_threads=1 # Force single thread to be deterministic, or >1 to test threading
        )
        print("Success! No crash.")
        print("Output shape:", tokens.shape)
        print("Lengths:", lengths.numpy())
        
        # Verify truncation happened correctly
        assert lengths[0] == 131072
        assert lengths[1] > 0
        
    except Exception as e:
        print(f"Caught exception: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    if not tok._ops_available:
        print("Ops not available, skipping test.")
        exit(0)
    test_overflow()
