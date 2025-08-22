from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import torch
import numpy as np
from transformer_lens import HookedTransformer

# -----------------------------
# Loading & tokenization utils
# -----------------------------

def _torch_dtype(dtype_name: str):
    name = dtype_name.lower()
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32

def load_tlens_model(
    model_name: str,
    device: str = "cuda",
    dtype: str = "float16",
    trust_remote_code: bool = True,
) -> HookedTransformer:
    torch_dtype = _torch_dtype(dtype)
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model

def tokenize_with_split(
    model: HookedTransformer,
    question: str,
    cot: str,
    p_char_index: int,
) -> Tuple[torch.Tensor, int, str, str, str]:
    """
    Build teacher-forced input with a precise token split at p_char_index.
    Returns:
        tokens         : LongTensor [1, seq_len]
        split_tok_idx  : int (first "after" token index)
        text_pre       : decoded text up to split point (question + cot[:p_idx])
        text_after     : decoded text from split point onward (cot[p_idx:])
        text_full      : decoded full text for debugging
    """
    if p_char_index is None or p_char_index < 0 or p_char_index > len(cot):
        raise ValueError("Invalid p_char_index for COT.")

    cot_pre = cot[:p_char_index]
    cot_after = cot[p_char_index:]

    text_pre = question + cot_pre
    tok_pre = model.to_tokens(text_pre, prepend_bos=True)     # [1, L1]
    tok_after = model.to_tokens(cot_after, prepend_bos=False) # [1, L2]

    tokens = torch.cat([tok_pre, tok_after], dim=1)           # [1, L1+L2]
    split_tok_idx = tok_pre.shape[1]

    # Re-decode for consistent debug that matches tokenization
    text_full = model.to_string(tokens[0])
    text_pre_dec = model.to_string(tok_pre[0])
    text_after_dec = model.to_string(tok_after[0])

    return tokens, split_tok_idx, text_pre_dec, text_after_dec, text_full

def clamp_layers(requested_layers: List[int], n_layers: int) -> List[int]:
    good = [i for i in requested_layers if 0 <= i < n_layers]
    if not good:
        mid = max(0, n_layers // 2)
        good = [mid]
    return sorted(set(good))

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Batching helpers
# -----------------------------

def get_pad_id(model: HookedTransformer) -> int:
    tok = model.tokenizer
    if getattr(tok, "pad_token_id", None) is not None:
        return tok.pad_token_id
    if getattr(tok, "eos_token_id", None) is not None:
        return tok.eos_token_id
    return 0

def pad_and_stack(token_tensors: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    """
    token_tensors: list of [1, L_i] LongTensors
    returns: [B, max_L] LongTensor
    """
    assert len(token_tensors) > 0
    max_len = max(t.shape[1] for t in token_tensors)
    bsz = len(token_tensors)
    device = token_tensors[0].device
    out = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
    for i, t in enumerate(token_tensors):
        L = t.shape[1]
        out[i, :L] = t[0]
    return out

def make_names_filter(hook_point: str, layers: List[int]) -> Callable[[str], bool]:
    """
    Keep only chosen hook point & layers in the cache.
    TransformerLens names look like 'blocks.{L}.hook_resid_post', etc.
    """
    hook_sub = f"hook_{hook_point}"
    layer_frags = [f"blocks.{L}." for L in layers]

    def _keep(name: str) -> bool:
        if hook_sub not in name:
            return False
        return any(fr in name for fr in layer_frags)
    return _keep

# -----------------------------
# Safe decoding for debug views
# -----------------------------

def decode_slice(model: HookedTransformer, tok_row: torch.Tensor, start: int, end: int) -> str:
    """
    Decode tokens in [start:end) without including any right-padding beyond `end`.
    `tok_row` is a 1D tensor of token ids (no batch dim).
    """
    if end <= start:
        return ""
    return model.to_string(tok_row[start:end])
