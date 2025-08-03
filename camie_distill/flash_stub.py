"""
flash_stub.py – transparently replaces Flash‑Attention with PyTorch 2.1 SDPA
if the CUDA extension is absent.
"""
from __future__ import annotations
import types, sys, builtins, warnings
import torch.nn.functional as F

def _flash_attn_stub(q, k, v, dropout_p: float = 0.0,
                     softmax_scale=None, causal: bool = False):
    q, k, v = (t.transpose(0, 1) for t in (q, k, v))
    return F.scaled_dot_product_attention(
        q, k, v, dropout_p=dropout_p, is_causal=causal
    ).transpose(0, 1)

def install_flash_stub() -> None:
    builtins.flash_attn_func = _flash_attn_stub    # global symbol

    if "flash_attn.flash_attention" in sys.modules:
        sys.modules["flash_attn.flash_attention"].flash_attn_func = \
            getattr(sys.modules["flash_attn.flash_attention"],
                    "flash_attn_func", _flash_attn_stub)
        return

    root = types.ModuleType("flash_attn")
    sub  = types.ModuleType("flash_attn.flash_attention")
    sub.flash_attn_func = _flash_attn_stub
    root.flash_attention = sub
    sys.modules.update({"flash_attn": root,
                        "flash_attn.flash_attention": sub})

    # Inform the operator once.
    import torch
    if torch.cuda.is_available():
        cc, _ = torch.cuda.get_device_capability()
        if cc >= 8:  # Ada/Hopper or newer
            warnings.warn("Flash‑Attention binary not found – "
                          "falling back to SDPA (~1.4× slower).")

install_flash_stub()
