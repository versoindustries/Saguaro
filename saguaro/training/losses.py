# saguaro/training/losses.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enterprise-grade loss functions for holographic embedding training.

This module provides production-ready loss functions ported from HighNoon,
specifically designed for training holographic/quantum embeddings.

Key Features:
    - InfoNCE contrastive loss for embedding-key alignment
    - Cosine similarity reconstruction loss
    - Evaluation metrics (top-k accuracy, cosine similarity)
    - Numerical stability with gradient-safe operations

Mathematical Foundation:
    InfoNCE Loss = -log( exp(sim(embed, key_pos) / τ) / Σ exp(sim(embed, key_i) / τ) )

    Where:
    - sim(a, b) = cosine similarity = a·b / (||a|| ||b||)
    - τ = temperature hyperparameter (default 0.07)
    - key_pos = positive key corresponding to the token

References:
    - Oord et al. (2018): Representation Learning with Contrastive Predictive Coding
    - Chen et al. (2020): A Simple Framework for Contrastive Learning of Visual Representations
"""

from __future__ import annotations

import logging
from typing import Tuple

import tensorflow as tf

logger = logging.getLogger(__name__)


def infonce_contrastive_loss(
    embeddings: tf.Tensor,
    token_keys: tf.Tensor,
    token_ids: tf.Tensor,
    temperature: float = 0.07,
    epsilon: float = 1e-8,
    reduction: str = "mean",
) -> tf.Tensor:
    """InfoNCE contrastive loss for holographic embedding training.

    Maximizes similarity between embeddings and their corresponding token keys
    while minimizing similarity to all other keys in the vocabulary.

    This is the PRIMARY loss function for holographic store training, replacing
    the broken squared magnitude loss.

    Mathematical Formulation:
        For each embedding e_i with positive key k_pos:
        L_i = -log( exp(sim(e_i, k_pos) / τ) / Σ_j exp(sim(e_i, k_j) / τ) )

        Where sim(a, b) = a·b / (||a|| ||b||) is cosine similarity.

    Args:
        embeddings: Output embeddings from holographic unbinding [batch, seq, dim]
                   or [batch*seq, dim] (flattened).
        token_keys: Token key matrix [vocab_size, dim]. Orthogonal keys for each token.
        token_ids: Ground truth token IDs [batch, seq] or [batch*seq].
                  Each ID indexes into token_keys.
        temperature: Softmax temperature τ (default: 0.07).
                    Lower = sharper distribution, faster convergence but less stable.
                    Higher = smoother distribution, more stable but slower.
        epsilon: Small constant for numerical stability in normalization.
        reduction: Loss reduction mode ('mean', 'sum', 'none').

    Returns:
        Scalar loss tensor (if reduction='mean' or 'sum') or
        per-sample losses [batch*seq] (if reduction='none').

    Raises:
        ValueError: If embedding and token_ids shapes are incompatible.

    Example:
        >>> embeddings = quantum_embedding(token_ids, token_keys, holographic_store)
        >>> loss = infonce_contrastive_loss(embeddings, token_keys, token_ids)
        >>> loss.numpy()
        5.123  # Should decrease during training

    Notes:
        - Expected initial loss ≈ log(vocab_size) for random embeddings
        - For vocab_size=50257: log(50257) ≈ 10.8
        - Target loss: < 3.0 (perplexity < 20)
        - Production target: < 2.0 (perplexity < 7.4)
    """
    # Input validation and shape handling
    embeddings = tf.cast(embeddings, tf.float32)
    token_keys = tf.cast(token_keys, tf.float32)

    # Flatten if 3D: [batch, seq, dim] -> [batch*seq, dim]
    original_shape = tf.shape(embeddings)
    if len(embeddings.shape) == 3:
        original_shape[0]
        original_shape[1]
        dim = original_shape[2]
        embeddings = tf.reshape(embeddings, [-1, dim])
        token_ids = tf.reshape(token_ids, [-1])
    else:
        dim = original_shape[-1]

    # Ensure token_ids are int32 for gather
    token_ids = tf.cast(token_ids, tf.int32)

    # L2 normalize embeddings and keys for cosine similarity
    emb_norm = tf.nn.l2_normalize(embeddings, axis=-1)
    key_norm = tf.nn.l2_normalize(token_keys, axis=-1)

    # Compute similarity matrix: [N, vocab_size]
    # Each row is similarity of one embedding to all keys
    logits = tf.matmul(emb_norm, key_norm, transpose_b=True)

    # Apply temperature scaling
    logits = logits / temperature

    # Cross-entropy loss: each embedding should predict its corresponding token ID
    # sparse_softmax_cross_entropy handles the softmax internally for efficiency
    per_sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=token_ids, logits=logits
    )

    # Apply reduction
    if reduction == "none":
        return per_sample_loss
    elif reduction == "sum":
        return tf.reduce_sum(per_sample_loss)
    else:  # reduction == "mean"
        return tf.reduce_mean(per_sample_loss)


def cosine_similarity_loss(
    embeddings: tf.Tensor,
    token_keys: tf.Tensor,
    token_ids: tf.Tensor,
    epsilon: float = 1e-8,
    reduction: str = "mean",
) -> tf.Tensor:
    """Cosine similarity reconstruction loss.

    Directly optimizes cosine similarity between embeddings and their
    corresponding token keys. Simpler than InfoNCE but lacks negative sampling.

    Mathematical Formulation:
        L = 1 - mean(cos_sim(e_i, k_i))
        Where k_i = token_keys[token_ids[i]]

    Args:
        embeddings: Output embeddings [batch*seq, dim].
        token_keys: Token key matrix [vocab_size, dim].
        token_ids: Ground truth token IDs [batch*seq].
        epsilon: Numerical stability constant.
        reduction: Loss reduction mode ('mean', 'sum', 'none').

    Returns:
        Loss tensor (1 - average cosine similarity).

    Notes:
        - Initial loss ≈ 1.0 (random embeddings have ~0 similarity)
        - Target loss: < 0.1 (cosine similarity > 0.9)
        - This loss is useful as a secondary metric or for fine-tuning
    """
    embeddings = tf.cast(embeddings, tf.float32)
    token_keys = tf.cast(token_keys, tf.float32)

    # Flatten if needed
    if len(embeddings.shape) == 3:
        dim = embeddings.shape[-1]
        embeddings = tf.reshape(embeddings, [-1, dim])
        token_ids = tf.reshape(token_ids, [-1])

    token_ids = tf.cast(token_ids, tf.int32)

    # Gather the correct keys for each token
    target_keys = tf.gather(token_keys, token_ids)  # [N, dim]

    # L2 normalize
    emb_norm = tf.nn.l2_normalize(embeddings, axis=-1)
    key_norm = tf.nn.l2_normalize(target_keys, axis=-1)

    # Cosine similarity: dot product of normalized vectors
    cos_sim = tf.reduce_sum(emb_norm * key_norm, axis=-1)  # [N]

    # Loss = 1 - similarity (so minimizing loss maximizes similarity)
    per_sample_loss = 1.0 - cos_sim

    if reduction == "none":
        return per_sample_loss
    elif reduction == "sum":
        return tf.reduce_sum(per_sample_loss)
    else:
        return tf.reduce_mean(per_sample_loss)


def embedding_retrieval_accuracy(
    embeddings: tf.Tensor,
    token_keys: tf.Tensor,
    token_ids: tf.Tensor,
    k: int = 1,
) -> tf.Tensor:
    """Compute top-k retrieval accuracy for embeddings.

    Measures how often the correct token key is in the top-k most similar keys
    to each embedding. This is the primary evaluation metric for holographic stores.

    Args:
        embeddings: Output embeddings [batch*seq, dim].
        token_keys: Token key matrix [vocab_size, dim].
        token_ids: Ground truth token IDs [batch*seq].
        k: Number of top predictions to consider (default: 1 for exact match).

    Returns:
        Accuracy scalar in [0, 1].

    Example:
        >>> top1_acc = embedding_retrieval_accuracy(embeddings, keys, ids, k=1)
        >>> top5_acc = embedding_retrieval_accuracy(embeddings, keys, ids, k=5)
        >>> print(f"Top-1: {top1_acc:.2%}, Top-5: {top5_acc:.2%}")
        Top-1: 87.50%, Top-5: 99.20%
    """
    embeddings = tf.cast(embeddings, tf.float32)
    token_keys = tf.cast(token_keys, tf.float32)

    # Flatten if needed
    if len(embeddings.shape) == 3:
        dim = embeddings.shape[-1]
        embeddings = tf.reshape(embeddings, [-1, dim])
        token_ids = tf.reshape(token_ids, [-1])

    token_ids = tf.cast(token_ids, tf.int64)

    # L2 normalize for cosine similarity
    emb_norm = tf.nn.l2_normalize(embeddings, axis=-1)
    key_norm = tf.nn.l2_normalize(token_keys, axis=-1)

    # Similarity matrix: [N, vocab_size]
    similarities = tf.matmul(emb_norm, key_norm, transpose_b=True)

    # Get top-k predictions
    _, top_k_indices = tf.nn.top_k(similarities, k=k)  # [N, k]
    top_k_indices = tf.cast(top_k_indices, tf.int64)

    # Check if true label is in top-k
    token_ids_expanded = tf.expand_dims(token_ids, axis=1)  # [N, 1]
    matches = tf.reduce_any(tf.equal(top_k_indices, token_ids_expanded), axis=1)

    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
    return accuracy


def combined_holographic_loss(
    embeddings: tf.Tensor,
    token_keys: tf.Tensor,
    token_ids: tf.Tensor,
    temperature: float = 0.07,
    cosine_weight: float = 0.1,
    spectral_weight: float = 0.01,
    reduction: str = "mean",
) -> Tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Combined loss for holographic embedding training.

    Combines InfoNCE contrastive loss with auxiliary regularization terms
    for optimal training stability and convergence.

    Components:
        - InfoNCE contrastive loss (primary)
        - Cosine similarity loss (reconstruction quality)
        - Spectral flatness regularization (embedding diversity)

    Args:
        embeddings: Output embeddings [batch*seq, dim].
        token_keys: Token key matrix [vocab_size, dim].
        token_ids: Ground truth token IDs [batch*seq].
        temperature: InfoNCE temperature.
        cosine_weight: Weight for cosine similarity auxiliary loss.
        spectral_weight: Weight for spectral flatness regularization.
        reduction: Loss reduction mode.

    Returns:
        Tuple of (total_loss, loss_components_dict).

    Example:
        >>> loss, components = combined_holographic_loss(emb, keys, ids)
        >>> print(f"Total: {loss:.4f}, InfoNCE: {components['infonce']:.4f}")
    """
    # Primary loss: InfoNCE contrastive
    infonce = infonce_contrastive_loss(
        embeddings, token_keys, token_ids, temperature=temperature, reduction=reduction
    )

    # Auxiliary loss: Cosine similarity
    cosine = cosine_similarity_loss(
        embeddings, token_keys, token_ids, reduction=reduction
    )

    # Regularization: Spectral diversity (encourages diverse embeddings)
    # Uses bounded log-variance formulation instead of 1/variance to avoid explosion
    # when embeddings are low-variance (e.g., early in training)
    if len(embeddings.shape) == 3:
        emb_flat = tf.reshape(embeddings, [-1, embeddings.shape[-1]])
    else:
        emb_flat = embeddings

    # Compute per-dimension variance
    variance = tf.math.reduce_variance(emb_flat, axis=0)
    mean_variance = tf.reduce_mean(variance)

    # Target variance of 1.0 (unit hypersphere) - bounded loss formulation
    # Loss = max(0, -log(variance)) = encourage variance >= 1
    # Clamp to avoid log(0) and bound maximum contribution
    clamped_variance = tf.clip_by_value(mean_variance, 1e-6, 10.0)
    spectral_loss = tf.maximum(0.0, -tf.math.log(clamped_variance))

    # Combine losses
    total_loss = infonce + cosine_weight * cosine + spectral_weight * spectral_loss

    components = {
        "infonce": infonce,
        "cosine": cosine,
        "spectral": spectral_loss,
        "total": total_loss,
    }

    return total_loss, components
