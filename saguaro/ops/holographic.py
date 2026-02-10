"""
Holographic Memory Operations for SAGUARO.
Wraps the C++ kernels exported by saguaro_memory_ops.cc.
"""

import tensorflow as tf
from . import quantum_ops


def holographic_bundle(vectors: tf.Tensor) -> tf.Tensor:
    """
    Compresses N vectors into a single holographic superposition state.

    Args:
        vectors: [N, D] tensor of vectors to bundle.

    Returns:
        bundle: [D] tensor representing the superposition.
    """
    return quantum_ops.holographic_bundle(vectors)


def modern_hopfield_retrieve(
    query: tf.Tensor, memory: tf.Tensor, beta: float = 1.0
) -> tf.Tensor:
    """
    Retrieves patterns from memory using Modern Hopfield Network dynamics.

    Args:
        query: [batch, D] query vectors.
        memory: [M, D] memory patterns.
        beta: Inverse temperature (higher means sharper retrieval).

    Returns:
        retrieved: [batch, D] retrieved patterns.
    """
    return quantum_ops.modern_hopfield_retrieve(query, memory, beta=beta)


def crystallize_memory(
    knowledge: tf.Tensor, importance: tf.Tensor, threshold: float = 0.5
) -> tf.Tensor:
    """
    Crystallizes memory patterns based on importance.

    Args:
        knowledge: [batch, D] raw memory.
        importance: [batch, D] importance weights.
        threshold: Threshold for crystallization.

    Returns:
        crystal: [batch, D] crystallized memory.
    """
    return quantum_ops.crystallize_memory(knowledge, importance, threshold)


def serialize_bundle(tensor: tf.Tensor) -> bytes:
    """Serializes a holographic bundle tensor to bytes."""
    return tf.io.serialize_tensor(tensor).numpy()


def deserialize_bundle(blob: bytes) -> tf.Tensor:
    """Deserializes a holographic bundle from bytes."""
    return tf.io.parse_tensor(blob, out_type=tf.float32)
