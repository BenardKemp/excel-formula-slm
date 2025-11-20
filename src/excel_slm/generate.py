"""
generate.py

Command-line inference script for the Excel Formula SLM.

This script:
    - Loads a saved checkpoint (model weights + vocabulary + config)
    - Reconstructs the tokenizer and Transformer model
    - Accepts a natural-language instruction from the user
    - Generates the corresponding Excel formula using autoregressive decoding

Usage example:

    python -m excel_slm.generate \
        --instruction "Sum E2:E100 where B2:B100 equals 'North'"

Output:

    Instruction: Sum E2:E100 where B2:B100 equals "North"
    Model output:
    Formula: =SUMIF(B2:B100,"North",E2:E100)<EOS>

This script is intentionally simple and clean for GitHub distribution.
"""

import argparse
from pathlib import Path

import torch

from .config import TrainConfig
from .model import ExcelSLM
from .tokenizer import CharTokenizer


# ============================================================================
#                     CHECKPOINT LOADING FUNCTION
# ============================================================================

def load_checkpoint(path: str):
    """
    Loads a checkpoint file and reconstructs:
        - Tokenizer
        - Model (with proper architecture)
        - Training config

    Checkpoint Structure (saved in train.py):
        {
            "model_state_dict": ...,
            "itos": [...],
            "config": {...}
        }
    """

    ckpt = torch.load(path, map_location="cpu")

    # Restore the vocabulary list
    itos = ckpt["itos"]

    # Restore config if present (fall back to default TrainConfig)
    cfg_dict = ckpt.get("config", {})
    config = TrainConfig(**cfg_dict) if cfg_dict else TrainConfig()

    # ------------------------------------------------------------
    # Rebuild tokenizer:
    # We recreate its structure manually and override stoi/itos.
    # ------------------------------------------------------------
    tokenizer = CharTokenizer("".join(itos))   # initial build
    tokenizer.itos = itos                      # overwrite vocab list
    tokenizer.stoi = {ch: i for i, ch in enumerate(itos)}  # rebuild lookup

    # ------------------------------------------------------------
    # Rebuild model with *exact* architecture used during training
    # ------------------------------------------------------------
    model = ExcelSLM(
        vocab_size=len(itos),
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_len=config.block_size,
        dropout=config.dropout,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    return model, tokenizer, config


# ============================================================================
#                     MAIN GENERATION ENTRY POINT
# ============================================================================

def main():
    """
    CLI interface for generating Excel formulas from instructions.
    """

    # ----------------------------
    # Command-line arguments
    # ----------------------------
    parser = argparse.ArgumentParser(
        description="Generate Excel formulas using the trained SLM."
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/excel_slm.pt",
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Natural language description of the Excel task.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=80,
        help="Maximum number of tokens to generate.",
    )

    args = parser.parse_args()

    # Validate that checkpoint exists
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {args.ckpt}. "
            "Train a model first using: python -m excel_slm.train"
        )

    # ----------------------------
    # Load model + tokenizer
    # ----------------------------
    model, tokenizer, config = load_checkpoint(args.ckpt)

    # Choose compute device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # ----------------------------
    # Format the model prompt
    # ----------------------------
    # This mirrors the dataset format:
    #
    #   Instruction: ...
    #   Formula: ...
    #
    prompt = f"Instruction: {args.instruction}\nFormula: "
    prompt_ids = tokenizer.encode(prompt)

    # ----------------------------
    # Run generation
    # ----------------------------
    output = model.generate(
        prompt_ids,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,      # lower = more deterministic
        top_k=20,             # restrict to top 20 tokens
        stop_token="\n",      # stop after end-of-example
    )

    # ----------------------------
    # Final output to console
    # ----------------------------
    print("\nInstruction:", args.instruction)
    print("\nModel output:\n")
    print(output)


# Allow running via: python -m excel_slm.generate
if __name__ == "__main__":
    main()
