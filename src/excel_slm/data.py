"""
data.py

This module handles:
- Loading NL → Excel formula examples from a text file
- Building a character tokenizer from the dataset
- Preparing sequences for training (input/output pairs)
- Padding and masking batches for the Transformer

Dataset format (excel_pairs.txt):
Each example is separated by a blank line:

Instruction: Sum sales in E2:E100 where region is "North" in B2:B100
Formula: =SUMIF(B2:B100,"North",E2:E100)<EOS>

Instruction: Count values in D2:D100 greater than 1000
Formula: =COUNTIF(D2:D100,">1000")<EOS>

Instruction: ...
Formula: ...<EOS>

This keeps data simple and easy to extend with synthetic examples.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


# ======================================================================
#                       DATASET CLASS
# ======================================================================

class ExcelFormulaDataset(Dataset):
    """
    Loads natural-language → Excel formula examples from a single text file.

    How it works:
    - Splits the file into examples based on blank lines
    - Each example becomes a sequence like:
        "Instruction: ...\nFormula: ...<EOS>\n"
    - Each example is encoded with CharTokenizer
    - Sequence is truncated/ignored if longer than block_size

    The model trains using next-token prediction:
        x = all characters except last
        y = all characters except first
    """

    def __init__(self, path: str, tokenizer: CharTokenizer, block_size: int = 256):
        text = Path(path).read_text(encoding="utf-8")
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Split file by blank lines into training examples
        raw_examples = [e.strip() for e in text.split("\n\n") if e.strip()]

        self.examples = []

        for ex in raw_examples:
            # Add newline so every example terminates cleanly
            ids = tokenizer.encode(ex + "\n")

            # Skip examples that exceed max allowed sequence length
            if len(ids) <= block_size:
                self.examples.append(torch.tensor(ids, dtype=torch.long))

        if len(self.examples) == 0:
            raise ValueError(
                f"No valid examples found. Check file '{path}' for format or increase block_size."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: sequence[:-1]
            y: sequence[1:]

        These are the standard input/target pairs for autoregressive training.
        """
        seq = self.examples[idx]
        x = seq[:-1]   # all tokens except last
        y = seq[1:]    # all tokens except first
        return x, y


# ======================================================================
#                       BATCH COLLATION
# ======================================================================

def collate_fn(batch):
    """
    Pads variable-length sequences in a batch.

    Input batch: list of (x, y) pairs where each x,y is a 1D tensor.

    Output:
        x_padded: (B, T)
        y_padded: (B, T)
        attn_mask: (B, T) boolean mask indicating valid tokens

    Padding tokens = 0 (because tokenizer contains no explicit <PAD>)
    The model uses attn_mask to prevent attending to padding.
    """
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    B = len(xs)

    x_padded = torch.zeros(B, max_len, dtype=torch.long)
    y_padded = torch.zeros(B, max_len, dtype=torch.long)
    attn_mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (x, y) in enumerate(zip(xs, ys)):
        L = len(x)
        x_padded[i, :L] = x
        y_padded[i, :L] = y
        attn_mask[i, :L] = True   # mark real tokens

    return x_padded, y_padded, attn_mask


# ======================================================================
#                     TOKENIZER BUILDER
# ======================================================================

def build_tokenizer_from_file(path: str) -> CharTokenizer:
    """
    Reads full text of training file and builds a character-level tokenizer.

    Useful because:
    - It ensures vocabulary exactly matches characters seen in dataset.
    - The model only sees tokens it can generate.

    If you want special tokens (<PAD>, <EOS>, etc.) you can add them here.
    """
    text = Path(path).read_text(encoding="utf-8")
    return CharTokenizer(text)
