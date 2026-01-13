
import tensorflow as tf
try:
    from saguaro.ops.quantum_ops import fused_text_tokenize_batch, _get_empty_resource_handle
    
    print("[-] Testing fused_text_tokenize_batch...")
    text_input = tf.constant(["hello world", "saguaro test"])
    
    # Try with default empty handle
    trie = _get_empty_resource_handle()
    
    tokens = fused_text_tokenize_batch(
        input_texts=text_input,
        trie_handle=trie
    )
    
    print(f"[+] Tokenizer Output Type: {type(tokens)}")
    print(f"[+] Tokens: {tokens}")
    print("SUCCESS: Tokenizer bindings are working.")

except Exception as e:
    print(f"FAILURE: Tokenizer bindings failed: {e}")
