"""
train.py

Training script for the Excel Formula Small Language Model (SLM).

Responsibilities:
- Load training data (natural-language → Excel formula pairs)
- Build a character-level tokenizer from the dataset
- Initialize the GPT-style Transformer model
- Run the training loop (next-token prediction)
- Save a checkpoint with model weights + tokenizer vocab + config

Run from the project root as:

    python -m excel_slm.train

(Assuming your `src` directory is on PYTHONPATH or you’re using
`python -m` from the repository root.)
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import ExcelFormulaDataset, collate_fn, build_tokenizer_from_file
from .model import ExcelSLM


def train(config: TrainConfig):
    """
    Main training routine.

    Args:
        config (TrainConfig): All hyperparameters and paths.

    Steps:
        1. Resolve device (CPU / GPU)
        2. Build tokenizer and dataset
        3. Initialize DataLoader
        4. Build model and optimizer
        5. Run training loop for num_epochs
        6. Save checkpoint
    """

    # ----------------------------------------------------------------------
    # 1. DEVICE SELECTION
    # ----------------------------------------------------------------------
    if config.device:
        # If user forces a device ("cpu" or "cuda")
        device = config.device
    else:
        # Auto-detect GPU if available, otherwise fallback to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Using device: {device}")

    # ----------------------------------------------------------------------
    # 2. TOKENIZER + DATASET
    # ----------------------------------------------------------------------
    # Build tokenizer from entire training file
    tokenizer = build_tokenizer_from_file(config.data_path)
    print(f"[INFO] Vocab size: {tokenizer.vocab_size}")

    # Create dataset of NL → formula examples
    dataset = ExcelFormulaDataset(
        config.data_path,
        tokenizer,
        block_size=config.block_size,
    )
    print(f"[INFO] Loaded {len(dataset)} training examples.")

    # DataLoader with our custom padding collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # ----------------------------------------------------------------------
    # 3. MODEL INITIALIZATION
    # ----------------------------------------------------------------------
    model = ExcelSLM(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_len=config.block_size,
        dropout=config.dropout,
    ).to(device)

    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ----------------------------------------------------------------------
    # 4. OPTIMIZER + LOSS
    # ----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # NOTE:
    #  - We are not currently using a special padding token with ignore_index.
    #  - The attention mask prevents the model from *using* padding tokens,
    #    but the loss still "sees" them (as zeros).
    #  - For a more robust setup, you could:
    #       * introduce an explicit <PAD> token in the tokenizer, and
    #       * set ignore_index=pad_token_id here.
    loss_fn = nn.CrossEntropyLoss()

    # Ensure checkpoint directory exists
    ckpt_dir = Path(config.ckpt_path).parent
    os.makedirs(ckpt_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # 5. TRAINING LOOP
    # ----------------------------------------------------------------------
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for x, y, attn_mask in dataloader:
            # x, y: (B, T), attn_mask: (B, T)
            x = x.to(device)
            y = y.to(device)
            attn_mask = attn_mask.to(device)

            # Forward pass: get logits for each position
            logits = model(x, attn_mask=attn_mask)  # (B, T, V)
            B, T, V = logits.shape

            # Reshape for CrossEntropyLoss: (B*T, V) vs (B*T,)
            loss = loss_fn(
                logits.view(B * T, V),
                y.view(B * T),
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"[EPOCH {epoch+1}/{config.num_epochs}] loss: {avg_loss:.4f}")

    # ----------------------------------------------------------------------
    # 6. SAVE CHECKPOINT
    # ----------------------------------------------------------------------
    ckpt = {
        "model_state_dict": model.state_dict(),
        "itos": tokenizer.itos,      # vocab for reconstruction
        "config": config.__dict__,   # save config for reproducibility
    }
    torch.save(ckpt, config.ckpt_path)
    print(f"[INFO] Saved checkpoint to {config.ckpt_path}")


def main():
    """
    Entry point when running:

        python -m excel_slm.train

    For now, it uses the default TrainConfig.
    You can later extend this to parse CLI arguments and override config.
    """
    config = TrainConfig()
    print("[INFO] Starting training with config:")
    for k, v in config.__dict__.items():
        print(f"  - {k}: {v}")

    train(config)


if __name__ == "__main__":
    main()
