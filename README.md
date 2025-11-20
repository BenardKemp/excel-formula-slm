# Excel Formula SLM

A tiny **Small Language Model (SLM)** in pure PyTorch that learns to convert
natural-language instructions into **Excel formulas**.

Example:

> Instruction: Sum sales in E2:E100 where region is "North" in B2:B100  
> Formula: `=SUMIF(B2:B100,"North",E2:E100)`

## Features

- ✅ From-scratch GPT-style decoder-only Transformer (no Hugging Face required)  
- ✅ Character-level tokenizer (handles `=`, `"`, `>`, `:` without fuss)  
- ✅ Simple training loop on NL → formula pairs  
- ✅ Generation script for interactive testing  

Perfect as a learning project or as the base for a specialized
**Nano Language Model** for Excel (6SigmaMind / NanoLanguageModels-style).

---

## Quickstart

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

