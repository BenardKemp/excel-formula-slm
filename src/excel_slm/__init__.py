"""
excel_slm package

This package contains everything needed to train and run
a Small Language Model (SLM) for converting natural language
instructions into Excel formulas.
"""

from .config import TrainConfig
from .tokenizer import CharTokenizer
from .data import ExcelFormulaDataset, collate_fn, build_tokenizer_from_file
from .model import ExcelSLM

__all__ = [
    "TrainConfig",
    "CharTokenizer",
    "ExcelFormulaDataset",
    "collate_fn",
    "build_tokenizer_from_file",
    "ExcelSLM",
]
