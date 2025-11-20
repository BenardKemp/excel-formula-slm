"""
tokenizer.py

This module contains a simple **character-level tokenizer**.

Why character-level?
- Excel formulas contain symbols like "=", ">", "<", ":", "\", "/", "(", ")", etc.
- Excel function names can be nested, combined, or contain unusual patterns.
- A character tokenizer avoids the need for complex logic or predefined vocab.

The CharTokenizer:
- Builds a vocabulary from all unique characters in the training dataset.
- Encodes strings into lists of integer token IDs.
- Decodes lists of IDs back into strings.
"""

from typing import List, Optional


class CharTokenizer:
    """
    A minimal character-level tokenizer.

    This tokenizer creates:
      - stoi: character → integer ID
      - itos: integer ID → character

    Use cases:
      tokenizer.encode("=SUM(A1:A10)")
      tokenizer.decode([5, 12, 33, ...])
    """

    def __init__(self, text: str, extra_tokens: Optional[List[str]] = None):
        """
        Build a tokenizer from a text corpus.

        Args:
            text (str): The full training dataset as a single string.
                        Every unique character becomes part of the vocabulary.
            extra_tokens (list[str], optional):
                        Useful for adding tokens like <PAD>, <BOS>, <EOS>.
        """
        # Allow user-defined special tokens (optional)
        self.extra_tokens = extra_tokens or []

        # Extract unique characters from data and sort them for stable vocab
        chars = sorted(set(text))

        # itos = ID → character list
        self.itos: List[str] = self.extra_tokens + chars

        # stoi = character → ID dictionary
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def encode(self, s: str) -> List[int]:
        """
        Convert a string into a list of token IDs.

        Args:
            s (str): Input string to tokenize.
        Returns:
            List[int]: Sequence of token IDs.
        """
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: List[int]) -> str:
        """
        Convert a list of token IDs back into a string.

        Args:
            ids (List[int]): Sequence of token IDs.
        Returns:
            str: Reconstructed string.
        """
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        """Return the total number of tokens in the vocabulary."""
        return len(self.itos)
