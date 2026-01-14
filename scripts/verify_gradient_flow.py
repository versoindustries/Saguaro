import tensorflow as tf
from saguaro.ops.quantum_ops import quantum_embedding, _get_empty_resource_handle


def verify_gradient_flow():
    print("=== Saguaro Gradient Flow Verification (Simplified) ===")

    vocab_size = 1000
    embedding_dim = 64
    batch_size = 2
    seq_len = 10

    # 1. Setup Trainable Weights
    embedding_matrix = tf.Variable(
        tf.random.normal([vocab_size, embedding_dim]),
        name="quantum_embeddings",
        dtype=tf.float32,
    )
    print(f"[-] Params Shape: {embedding_matrix.shape}")

    # 2. Setup Inputs (Dummy)
    # Token IDs (int32)
    token_ids = tf.random.uniform(
        (batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32
    )
    print(f"[-] Token IDs Shape: {token_ids.shape}")

    # keys (Float vectors) - In real usage, these come from tokenizer or hash.
    # We gather them from a "key table" or just generate randoms matching input shape.
    # Let's assume input keys correspond to the token_ids.
    dummy_keys_table = tf.random.normal([vocab_size, 64])
    token_keys = tf.gather(dummy_keys_table, token_ids)
    print(f"[-] Token Keys Shape: {token_keys.shape}")

    # Store Resource (Empty/Dummy)
    try:
        _get_empty_resource_handle()
    except Exception as e:
        print(f"[-] Warning: Count not get resource handle: {e}")

    # 3. Gradient Tape
    with tf.GradientTape() as tape:
        tape.watch(embedding_matrix)

        print("[-] Running Quantum Embedding...")
        # Check signature: (input_tensor, token_keys, holographic_store, vocab_size, hd_dim)
        embeddings = quantum_embedding(
            input_tensor=token_ids,
            token_keys=token_keys,  # Key vectors
            holographic_store=embedding_matrix,  # Weights as store (Float!)
            vocab_size=vocab_size,
            hd_dim=embedding_dim,
        )

        print(f"[-] Embeddings Output Shape: {embeddings.shape}")

        # Projection for Scalar Loss
        # Simple reduce sum or projection
        loss = tf.reduce_mean(embeddings**2)
        print(f"[-] Loss: {loss.numpy()}")

    # 4. Compute Gradients
    print("[-] Computing Gradients...")
    grads = tape.gradient(loss, embedding_matrix)

    if grads is None:
        print("❌ FAILED: Gradients are None")
        exit(1)

    grad_norm = tf.norm(grads).numpy()
    print(f"[-] Gradient Norm: {grad_norm}")

    if grad_norm > 0.0:
        print("✅ SUCCESS: Gradient Flow Verified!")
    else:
        print("⚠️ WARNING: Gradient Norm is 0")


if __name__ == "__main__":
    verify_gradient_flow()
