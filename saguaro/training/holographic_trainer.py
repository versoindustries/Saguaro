# saguaro/training/holographic_trainer.py
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

"""Enterprise-grade training infrastructure for holographic embeddings.

This module provides production-ready training components with:
- AdamW optimizer with weight decay
- Cosine annealing learning rate with warmup
- Gradient clipping for stability
- Checkpointing and early stopping
- Comprehensive logging and metrics

Architecture ported from HighNoon training infrastructure.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
import tensorflow as tf

from saguaro.training.losses import (
    infonce_contrastive_loss,
    embedding_retrieval_accuracy,
    combined_holographic_loss,
)
from saguaro.ops.quantum_ops import quantum_embedding

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for holographic embedding training.
    
    Attributes:
        vocab_size: Vocabulary size for token keys.
        embedding_dim: Embedding dimension (should be power of 2).
        num_bundles: Number of holographic bundles in store.
        
        batch_size: Training batch size.
        seq_length: Sequence length for training samples.
        max_steps: Maximum training steps.
        
        learning_rate: Peak learning rate for AdamW.
        weight_decay: L2 weight decay coefficient.
        warmup_steps: Number of warmup steps for LR schedule.
        grad_clip_norm: Maximum gradient norm for clipping.
        
        temperature: InfoNCE temperature parameter.
        
        checkpoint_dir: Directory for checkpoints.
        checkpoint_interval: Steps between checkpoints.
        log_interval: Steps between logging.
        eval_interval: Steps between evaluation.
        
        early_stopping_patience: Epochs without improvement to stop.
        early_stopping_min_delta: Minimum improvement to count.
    """
    # Model architecture
    vocab_size: int = 50257
    embedding_dim: int = 256
    num_bundles: int = 256
    
    # Training batch
    batch_size: int = 32
    seq_length: int = 128
    max_steps: int = 5000
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip_norm: float = 1.0
    
    # Loss
    temperature: float = 0.07
    
    # Checkpointing
    checkpoint_dir: str | None = None
    checkpoint_interval: int = 500
    log_interval: int = 10
    eval_interval: int = 100
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001


class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine annealing learning rate schedule with warmup.
    
    LR starts at 0, linearly warms up to peak_lr, then cosine decays to min_lr.
    """
    
    def __init__(
        self,
        peak_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
    ):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        
        # Warmup phase: linear increase
        warmup_lr = self.peak_lr * (step / (warmup_steps + 1e-8))
        
        # Cosine decay phase
        decay_steps = total_steps - warmup_steps
        decay_progress = (step - warmup_steps) / (decay_steps + 1e-8)
        decay_progress = tf.clip_by_value(decay_progress, 0.0, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * decay_progress))
        decay_lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
        
        # Use warmup LR during warmup, decay LR after
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)
    
    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


class HolographicTrainer:
    """Enterprise-grade trainer for holographic embedding stores.
    
    Implements production training loop with:
    - AdamW optimizer with weight decay
    - Cosine annealing LR with warmup  
    - Gradient clipping
    - Comprehensive metrics
    - Checkpointing and early stopping
    
    Example:
        >>> config = TrainingConfig(vocab_size=50257, embedding_dim=256)
        >>> trainer = HolographicTrainer(config)
        >>> trainer.train(data_generator, eval_data=eval_generator)
        >>> trainer.save("path/to/checkpoint")
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration.
        
        Args:
            config: Training configuration dataclass.
        """
        self.config = config
        
        # Initialize trainable holographic store
        self.holographic_store = tf.Variable(
            tf.random.normal([config.num_bundles, config.embedding_dim], stddev=0.1),
            trainable=True,
            name="holographic_store",
            dtype=tf.float32,
        )
        
        # Initialize token keys (fixed, Haar-random orthogonal)
        self.token_keys = self._init_haar_keys(config.vocab_size, config.embedding_dim)
        
        # Learning rate schedule - store separately for access in tf.function
        self.lr_schedule = CosineAnnealingSchedule(
            peak_lr=config.learning_rate,
            warmup_steps=config.warmup_steps,
            total_steps=config.max_steps,
        )
        
        # Optimizer: AdamW with weight decay
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.lr_schedule,
            weight_decay=config.weight_decay,
            clipnorm=config.grad_clip_norm,
        )
        
        # Training state
        self.global_step = 0
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.training_history: list[dict[str, Any]] = []
        
        # Checkpointing
        if config.checkpoint_dir:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        logger.info(
            f"Initialized HolographicTrainer:\n"
            f"  vocab_size={config.vocab_size}\n"
            f"  embedding_dim={config.embedding_dim}\n"
            f"  num_bundles={config.num_bundles}\n"
            f"  learning_rate={config.learning_rate}\n"
            f"  warmup_steps={config.warmup_steps}"
        )
    
    def _init_haar_keys(self, vocab_size: int, dim: int) -> tf.Tensor:
        """Initialize Haar-random orthogonal token keys.
        
        Generates unit vectors uniformly distributed on the hypersphere.
        These remain fixed during training.
        
        Args:
            vocab_size: Number of tokens.
            dim: Key dimension.
        
        Returns:
            Token keys tensor [vocab_size, dim].
        """
        # Generate random Gaussian vectors
        keys = tf.random.normal([vocab_size, dim], dtype=tf.float32, seed=42)
        # L2 normalize to unit hypersphere
        keys = tf.nn.l2_normalize(keys, axis=-1)
        return keys
    
    @tf.function
    def train_step(
        self,
        token_ids: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        """Execute a single training step.
        
        Args:
            token_ids: Batch of token IDs [batch, seq_len].
        
        Returns:
            Dictionary of metrics (loss, grad_norm, lr, etc.).
        """
        with tf.GradientTape() as tape:
            # Forward pass: holographic embedding lookup
            embeddings = quantum_embedding(
                token_ids,
                self.token_keys,
                holographic_store=self.holographic_store,
                vocab_size=self.config.vocab_size,
                hd_dim=self.config.embedding_dim,
            )
            
            # Compute InfoNCE contrastive loss
            loss, components = combined_holographic_loss(
                embeddings,
                self.token_keys,
                token_ids,
                temperature=self.config.temperature,
            )
        
        # Compute gradients
        gradients = tape.gradient(loss, [self.holographic_store])
        
        # Compute gradient norm before clipping
        grad_norm = tf.linalg.global_norm(gradients)
        
        # Apply gradients (clipping is handled by optimizer)
        self.optimizer.apply_gradients(
            zip(gradients, [self.holographic_store])
        )
        
        # Get current learning rate from schedule
        current_lr = self.lr_schedule(self.optimizer.iterations)
        
        metrics = {
            "loss": loss,
            "infonce": components["infonce"],
            "cosine": components["cosine"],
            "grad_norm": grad_norm,
            "lr": current_lr,
        }
        
        return metrics
    
    def evaluate(
        self,
        eval_data: Iterator[tf.Tensor],
        max_batches: int = 50,
    ) -> dict[str, float]:
        """Evaluate model on validation data.
        
        Args:
            eval_data: Iterator yielding token_ids batches.
            max_batches: Maximum batches to evaluate.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        num_batches = 0
        
        for batch_idx, token_ids in enumerate(eval_data):
            if batch_idx >= max_batches:
                break
            
            # Forward pass
            embeddings = quantum_embedding(
                token_ids,
                self.token_keys,
                holographic_store=self.holographic_store,
                vocab_size=self.config.vocab_size,
                hd_dim=self.config.embedding_dim,
            )
            
            # Compute loss
            loss = infonce_contrastive_loss(
                embeddings, self.token_keys, token_ids,
                temperature=self.config.temperature,
            )
            
            # Compute accuracy
            top1_acc = embedding_retrieval_accuracy(
                embeddings, self.token_keys, token_ids, k=1
            )
            top5_acc = embedding_retrieval_accuracy(
                embeddings, self.token_keys, token_ids, k=5
            )
            
            total_loss += float(loss)
            total_top1 += float(top1_acc)
            total_top5 += float(top5_acc)
            num_batches += 1
        
        if num_batches == 0:
            return {"eval_loss": 0.0, "top1_acc": 0.0, "top5_acc": 0.0}
        
        return {
            "eval_loss": total_loss / num_batches,
            "top1_acc": total_top1 / num_batches,
            "top5_acc": total_top5 / num_batches,
        }
    
    def train(
        self,
        train_data: Iterator[tf.Tensor],
        eval_data: Iterator[tf.Tensor] | Callable[[], Iterator[tf.Tensor]] | None = None,
        callbacks: list[Callable] | None = None,
    ) -> dict[str, Any]:
        """Run full training loop.
        
        Args:
            train_data: Iterator yielding token_ids batches [batch, seq_len].
            eval_data: Optional validation data iterator OR a callable that returns
                      a fresh iterator each time (recommended for multiple evals).
            callbacks: Optional list of callback functions.
        
        Returns:
            Training results dictionary with final metrics and history.
        """
        logger.info(f"Starting training for {self.config.max_steps} steps...")
        start_time = time.time()
        
        for step, token_ids in enumerate(train_data):
            if self.global_step >= self.config.max_steps:
                break
            
            # Training step
            metrics = self.train_step(token_ids)
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                log_msg = (
                    f"step {self.global_step}: "
                    f"loss={float(metrics['loss']):.4f}, "
                    f"grad_norm={float(metrics['grad_norm']):.2f}, "
                    f"lr={float(metrics['lr']):.2e}"
                )
                logger.info(log_msg)
                print(log_msg)
            
            # Evaluation
            if eval_data and self.global_step % self.config.eval_interval == 0:
                # If eval_data is callable, call it to get a fresh iterator
                eval_iter = eval_data() if callable(eval_data) else eval_data
                eval_metrics = self.evaluate(eval_iter)
                eval_msg = (
                    f"  [EVAL] loss={eval_metrics['eval_loss']:.4f}, "
                    f"top1={eval_metrics['top1_acc']:.2%}, "
                    f"top5={eval_metrics['top5_acc']:.2%}"
                )
                logger.info(eval_msg)
                print(eval_msg)
                
                # Early stopping check
                if eval_metrics["eval_loss"] < self.best_loss - self.config.early_stopping_min_delta:
                    self.best_loss = eval_metrics["eval_loss"]
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered at step {self.global_step}")
                    break
            
            # Checkpointing
            if self.config.checkpoint_dir and self.global_step % self.config.checkpoint_interval == 0:
                ckpt_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"checkpoint_step_{self.global_step}"
                )
                self.save(ckpt_path)
            
            # Record history
            self.training_history.append({
                "step": self.global_step,
                "loss": float(metrics["loss"]),
                "grad_norm": float(metrics["grad_norm"]),
                "lr": float(metrics["lr"]),
            })
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, metrics)
        
        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed:.1f}s")
        
        return {
            "final_step": self.global_step,
            "final_loss": float(metrics["loss"]),
            "best_loss": self.best_loss,
            "elapsed_time": elapsed,
            "history": self.training_history,
        }
    
    def save(self, path: str) -> None:
        """Save trainer state and holographic store.
        
        Args:
            path: Directory path for checkpoint.
        """
        os.makedirs(path, exist_ok=True)
        
        # Save holographic store
        store_path = os.path.join(path, "holographic_store.npy")
        np.save(store_path, self.holographic_store.numpy())
        
        # Save token keys
        keys_path = os.path.join(path, "token_keys.npy")
        np.save(keys_path, self.token_keys.numpy())
        
        # Save config and state
        import json
        config_path = os.path.join(path, "config.json")
        state = {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "vocab_size": self.config.vocab_size,
            "embedding_dim": self.config.embedding_dim,
            "num_bundles": self.config.num_bundles,
        }
        with open(config_path, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved checkpoint to {path}")
    
    def load(self, path: str) -> None:
        """Load trainer state from checkpoint.
        
        Args:
            path: Directory path of checkpoint.
        """
        # Load holographic store
        store_path = os.path.join(path, "holographic_store.npy")
        store_data = np.load(store_path)
        self.holographic_store.assign(store_data)
        
        # Load token keys
        keys_path = os.path.join(path, "token_keys.npy")
        if os.path.exists(keys_path):
            self.token_keys = tf.constant(np.load(keys_path), dtype=tf.float32)
        
        # Load config
        import json
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            state = json.load(f)
        
        self.global_step = state.get("global_step", 0)
        self.best_loss = state.get("best_loss", float("inf"))
        
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")


def create_synthetic_data_generator(
    vocab_size: int,
    batch_size: int,
    seq_length: int,
    num_batches: int = 1000,
) -> Iterator[tf.Tensor]:
    """Create synthetic data generator for testing.
    
    Yields random token IDs for training loop verification.
    
    Args:
        vocab_size: Maximum token ID (exclusive).
        batch_size: Batch size.
        seq_length: Sequence length.
        num_batches: Number of batches to generate.
    
    Yields:
        Token ID tensors [batch_size, seq_length].
    """
    for _ in range(num_batches):
        yield tf.random.uniform(
            [batch_size, seq_length],
            minval=0,
            maxval=vocab_size,
            dtype=tf.int32,
        )
