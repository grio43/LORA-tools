#!/usr/bin/env python3
"""
Camie‑Tagger • Training‑set Candidate Generator
------------------------------------------------
Runs Camie‑Tagger in batch mode and produces (image, .txt) pairs
suitable for caption‑based training.  For each input picture the
script keeps only tags whose probability is **≥ --confidence‑threshold**.

• Output directory   : TRAINING_CANDIDATES_DIR
• Output file layout :  <name>.png  +  <name>.txt  (comma‑separated tags)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import types
import builtins
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load
from torch import amp
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Flash‑Attention → SDPA fallback
# ──────────────────────────────────────────────────────────────────────────────
def _flash_attn_stub(q, k, v, dropout_p: float = 0.0,
                     softmax_scale=None, causal: bool = False):
    """Replacement for flash_attn.flash_attention.flash_attn_func."""
    q, k, v = (t.transpose(0, 1) for t in (q, k, v))  # (L, NH, D)
    out = F.scaled_dot_product_attention(q, k, v,
                                         dropout_p=dropout_p,
                                         is_causal=causal)
    return out.transpose(0, 1)                         # (NH, L, D)


def _install_flash_attn_stub() -> None:
    """Expose flash_attn_func even when flash‑attention isn’t installed."""
    builtins.flash_attn_func = _flash_attn_stub  # global fallback

    if "flash_attn.flash_attention" in sys.modules:
        sub = sys.modules["flash_attn.flash_attention"]
        sub.flash_attn_func = getattr(sub, "flash_attn_func", _flash_attn_stub)
        return

    root = types.ModuleType("flash_attn")
    sub  = types.ModuleType("flash_attn.flash_attention")
    sub.flash_attn_func = _flash_attn_stub
    root.flash_attention = sub
    sys.modules.update({
        "flash_attn": root,
        "flash_attn.flash_attention": sub,
    })


_install_flash_attn_stub()

# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────
IMAGE_EXTS: set[str] = {".png", ".jpg", ".jpeg", ".webp"}

INPUT_DIR: Path                 = Path(r"K:\Ready for captions without tags")
TRAINING_CANDIDATES_DIR: Path   = Path(r"D:\TrainingCandidates")

# ──────────────────────────────────────────────────────────────────────────────
# Pre‑processing
# ──────────────────────────────────────────────────────────────────────────────
def _preprocess(path: Path, size: int = 512, *, fp16: bool = False) -> np.ndarray:
    """Resize‑and‑pad to square CHW tensor with values in [0, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    r = w / h
    nw, nh = (size, int(size / r)) if r > 1 else (int(size * r), size)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    arr = np.asarray(canvas, dtype=np.float16 if fp16 else np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))

# ──────────────────────────────────────────────────────────────────────────────
# Dynamic model import
# ──────────────────────────────────────────────────────────────────────────────
def _lazy_import_arch(repo: str, *, token: str | None):
    """Load model_code.py from HF and guarantee flash_attn_func is present."""
    code_path = hf_hub_download(repo, "model/model_code.py", token=token)
    import importlib.util

    spec = importlib.util.spec_from_file_location("camie_model", code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec from {code_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if "flash_attn_func" not in module.__dict__:
        module.flash_attn_func = _flash_attn_stub

    return module

# ──────────────────────────────────────────────────────────────────────────────
# Inference runner
# ──────────────────────────────────────────────────────────────────────────────
class SafetensorRunner:
    def __init__(self, repo: str, device: str, fp16: bool, *, token: str | None):
        print("⇩ Downloading refined weights and metadata…")

        ckpt_path     = hf_hub_download(repo, "model_refined.safetensors", token=token)
        meta_path     = hf_hub_download(repo, "model/model_info_refined.json", token=token)
        tag_meta_path = hf_hub_download(repo, "model/metadata.json",      token=token)

        meta     = json.loads(Path(meta_path).read_text())
        tag_meta = json.loads(Path(tag_meta_path).read_text())
        self.idx2tag: Dict[int, str] = {int(k): v for k, v in tag_meta["idx_to_tag"].items()}
        self.tag2cat: Dict[str, str] = tag_meta["tag_to_category"]

        arch = _lazy_import_arch(repo, token=token)

        dataset = arch.TagDataset(
            total_tags=len(self.idx2tag),
            idx_to_tag=self.idx2tag,
            tag_to_category=self.tag2cat,
        )
        model = arch.ImageTagger(
            total_tags=len(self.idx2tag),
            dataset=dataset,
            num_heads=meta.get("num_heads", 16),
            tag_context_size=meta.get("tag_context_size", 256),
            pretrained=False,
        )

        state = safe_load(ckpt_path, device=device)
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        if fp16:
            model.half()

        self.model  = model
        self.device = device
        self.fp16   = fp16

    def __call__(self, batch: List[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(batch)).to(self.device)
        with torch.no_grad(), amp.autocast(device_type=self.device, enabled=self.fp16):
            _, refined = self.model(x)
        return refined.cpu().numpy()

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def _write_tag_file(target_dir: Path, image_path: Path, tags: List[str]) -> None:
    """Create `<basename>.txt` with comma‑separated tag list."""
    txt_path = target_dir / f"{image_path.stem}.txt"
    txt_path.write_text(", ".join(tags), encoding="utf‑8")


def main(args: argparse.Namespace) -> None:
    # Initialise runner
    runner = SafetensorRunner(
        repo=args.model_repo,
        device=args.device,
        fp16=args.fp16,
        token=args.hf_token
    )

    # Prepare output directory
    TRAINING_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    # Gather images
    images = sorted(
        p for p in INPUT_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise SystemExit(f"No images found in {INPUT_DIR}")

    print(f"✓ Found {len(images)} images. Starting batch processing…")

    # Batch inference
    for i in tqdm(range(0, len(images), args.batch_size), unit="batch"):
        batch_paths = images[i : i + args.batch_size]
        batch_arr   = [_preprocess(p, fp16=args.fp16) for p in batch_paths]

        logits = runner(batch_arr)
        probs  = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

        for pic, pb in zip(batch_paths, probs):
            # ── 1 & 2.  Filter by confidence threshold ────────────────────────
            accepted_tags: List[str] = [
                runner.idx2tag[j]
                for j, prob in enumerate(pb)
                if prob >= args.confidence_threshold
            ]
            if not accepted_tags:      # 2. skip if empty
                continue

            # ── 3. Write .txt sidecar ─────────────────────────────────────────
            _write_tag_file(TRAINING_CANDIDATES_DIR, pic, accepted_tags)

            # ── 4. Copy image ────────────────────────────────────────────────
            shutil.copy2(pic, TRAINING_CANDIDATES_DIR / pic.name)

    print("✓ All done. Training candidates are in", TRAINING_CANDIDATES_DIR)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate (image, .txt) training pairs using Camie‑Tagger."
    )

    # Required credentials
    parser.add_argument("--hf-token", required=True,
                        help="Hugging Face auth token.")

    # New confidence threshold
    parser.add_argument("--confidence-threshold", type=float, default=0.35,
                        help="Minimum probability to include a tag.")

    # Execution & model options
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Images per batch.")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable half‑precision inference.")
    parser.add_argument("--model-repo", default="Camais03/camie-tagger",
                        help="Model repository name.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Inference device.")

    args = parser.parse_args()
    main(args)