"""
config.py

This module defines the configuration dataclass used to train
the Excel Formula SLM. All hyperparameters, file paths, model
sizes, and training options live here.

This keeps train.py clean and makes it easy to extend or override.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """
    Configuration values for training the Excel SLM.
    
    You can modify these values directly here, or override them
    by passing custom arguments in a CLI wrapper.
    """

    # -----------------------------
    # DATA SETTINGS
    # -----------------------------
    
    # Path to the training dataset (NL → formula pairs)
    data_path: str = "data/excel_pairs.txt"

    # Maximum sequence length (prompt + formula).
    # If examples exceed this length, they are skipped.
    block_size: int = 256

    # Batch size per step
    batch_size: int = 16


    # -----------------------------
    # TRAINING SETTINGS
    # -----------------------------
    
    # Number of epochs to train
    num_epochs: int = 10

    # Learning rate for AdamW optimizer
    lr: float = 3e-4

    # L2 weight decay
    weight_decay: float = 0.01

    # Gradient clipping to stabilize training
    grad_clip: float = 1.0


    # -----------------------------
    # MODEL ARCHITECTURE
    # -----------------------------
    
    # Dimensionality of the Transformer embeddings
    d_model: int = 256

    # Number of Transformer blocks (depth)
    num_layers: int = 4

    # Number of attention heads in each block
    num_heads: int = 8

    # Feed-forward network size inside each block
    d_ff: int = 1024

    # Dropout for stabilizing training
    dropout: float = 0.1


    # -----------------------------
    # SYSTEM SETTINGS
    # -----------------------------
    
    # Force a device: "", "cpu", or "cuda".
    # If empty, training auto-selects CUDA if available.
    device: str = ""

    # Where to save the model checkpoint
    ckpt_path: str = "checkpoints/excel_slm.pt"
