#!/usr/bin/env python3
"""
Camie‑Tagger • Dataset Builder
==============================

Batch‑runs the Camie‑Tagger model on a folder of images and produces a
*training‑ready* dataset in exactly the same layout as the public
`val_dataset.csv` that ships with the original repository.

Required arguments
------------------
--input-dir        Path to a directory that contains **only** images
                   (*.png, *.jpg/jpeg, *.webp). Sub‑folders are scanned.
--output-dir       Destination directory. The script creates it (and an
                   `images/` sub‑folder) if it doesn’t exist.
--hf-token         Your personal Hugging Face access token.

Common options
--------------
--confidence-threshold  0.35   Minimum per‑tag probability to write the tag
--batch-size            64     Number of images per forward pass
--fp16                         Enable half‑precision inference (faster / less VRAM)
--device                cuda   Inference device: `cuda` or `cpu`
--model-repo            Camais03/camie-tagger  Repository that hosts the model
--skip-sidecar                  Do **not** generate `<image>.txt` caption files

Typical invocation
------------------
python camie_tagger_dataset_builder.py \
    --input-dir  "/path/to/your/images" \
    --output-dir "/path/to/new/dataset" \
    --hf-token   "hf_xxxxxxxxxxxxxxxxxx" \
    --confidence-threshold 0.35 \
    --batch-size  64 \
    --fp16 \
    --device      cuda \
    --skip-sidecar       # optional

Resulting directory structure
-----------------------------
/path/to/new/dataset/
 ├─ images/                  (original files, unchanged names)
 ├─ train_dataset.csv        (training split – can be renamed or merged)
 └─ *.txt                    (caption‑style tag files, unless skipped)

CSV row format (4 columns)
--------------------------
id, file_name, tag_idx, tag

Example:
0,000001.png,34 712 24 898,cat girl long hair blush thighhighs

Immediate loading in Camie‑Tagger notebooks
-------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
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


# ────────────────────────────────────────────────────────────────────
# Flash‑Attention → SDPA fallback
# ────────────────────────────────────────────────────────────────────
def _flash_attn_stub(q, k, v, dropout_p: float = 0.0,
                     softmax_scale=None, causal: bool = False):
    q, k, v = (t.transpose(0, 1) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v,
                                          dropout_p=dropout_p,
                                          is_causal=causal).transpose(0, 1)


def _install_flash_attn_stub() -> None:
    builtins.flash_attn_func = _flash_attn_stub   # global fallback

    if "flash_attn.flash_attention" in sys.modules:
        sub = sys.modules["flash_attn.flash_attention"]
        sub.flash_attn_func = getattr(sub, "flash_attn_func",
                                      _flash_attn_stub)
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

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
IMAGE_EXTS: set[str] = {".png", ".jpg", ".jpeg", ".webp"}


def _preprocess(path: Path, size: int = 512, *, fp16: bool = False) -> np.ndarray:
    """Resize‑and‑pad to square CHW tensor with values in [0, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    r = w / h
    nw, nh = (size, int(size / r)) if r > 1 else (int(size * r), size)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (size, size))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    arr = np.asarray(canvas,
                     dtype=np.float16 if fp16 else np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


# ────────────────────────────────────────────────────────────────────
# Dynamic model import
# ────────────────────────────────────────────────────────────────────
def _lazy_import_arch(repo: str, *, token: str | None):
    code_path = hf_hub_download(repo, "model/model_code.py", token=token)
    import importlib.util
    spec  = importlib.util.spec_from_file_location("camie_model", code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec from {code_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if "flash_attn_func" not in module.__dict__:
        module.flash_attn_func = _flash_attn_stub
    return module


# ────────────────────────────────────────────────────────────────────
# Inference runner
# ────────────────────────────────────────────────────────────────────
class SafetensorRunner:
    def __init__(self, repo: str, device: str, fp16: bool, *,
                 token: str | None):
        print("⇩ Downloading refined weights and metadata…")
        ckpt_path = hf_hub_download(repo, "model_refined.safetensors",
                                    token=token)
        meta_path = hf_hub_download(repo, "model/model_info_refined.json",
                                    token=token)
        tag_meta_path = hf_hub_download(repo, "model/metadata.json",
                                        token=token)

        meta     = json.loads(Path(meta_path).read_text())
        tag_meta = json.loads(Path(tag_meta_path).read_text())
        self.idx2tag: Dict[int, str] = {int(k): v
                                        for k, v in
                                        tag_meta["idx_to_tag"].items()}
        # speed‑up for lookup in the main loop
        self.tag2idx: Dict[str, int] = {v: k for k, v in self.idx2tag.items()}

        arch = _lazy_import_arch(repo, token=token)

        dataset = arch.TagDataset(total_tags=len(self.idx2tag),
                                  idx_to_tag=self.idx2tag,
                                  tag_to_category=tag_meta["tag_to_category"])

        model = arch.ImageTagger(total_tags=len(self.idx2tag),
                                 dataset=dataset,
                                 num_heads=meta.get("num_heads", 16),
                                 tag_context_size=meta.get("tag_context_size",
                                                            256),
                                 pretrained=False)

        state = safe_load(ckpt_path, device=device)
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        if fp16:
            model.half()

        self.model  = model
        self.device = device
        self.fp16   = fp16

    # ────────────────────────────────────────────────
    def __call__(self, batch: List[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(batch)).to(self.device)
        with torch.no_grad(), amp.autocast(device_type=self.device,
                                           enabled=self.fp16):
            _, refined = self.model(x)
        return refined.cpu().numpy()


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
def _write_sidecar(txt_dir: Path, image_path: Path,
                   tags: List[str]) -> None:
    """Optional `<basename>.txt` with comma‑separated tag list."""
    txt_dir.joinpath(f"{image_path.stem}.txt") \
           .write_text(", ".join(tags), encoding="utf‑8")


def build_dataset(args: argparse.Namespace) -> None:
    # Initialise runner
    runner = SafetensorRunner(repo=args.model_repo,
                              device=args.device,
                              fp16=args.fp16,
                              token=args.hf_token)

    # I/O setup
    out_root        = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    img_out_dir     = out_root / "images"
    img_out_dir.mkdir(exist_ok=True)

    csv_path        = out_root / "train_dataset.csv"
    sidecar_dir     = out_root if not args.skip_sidecar else None

    # Gather input
    images = sorted(p
                    for p in Path(args.input_dir).rglob("*")
                    if p.suffix.lower() in IMAGE_EXTS)

    if not images:
        raise SystemExit(f"No images found under {args.input_dir}")

    print(f"✓ Found {len(images)} images. Starting batch processing…")

    # Prepare CSV writer
    csv_file = csv_path.open("w", newline='', encoding="utf-8")
    writer   = csv.writer(csv_file)
    writer.writerow(["id", "file_name", "tag_idx", "tag"])  # header row

    running_id = 0

    # Batch inference
    for i in tqdm(range(0, len(images), args.batch_size), unit="batch"):
        batch_paths = images[i : i + args.batch_size]
        batch_arr   = [_preprocess(p, fp16=args.fp16) for p in batch_paths]

        logits = runner(batch_arr)
        probs  = 1. / (1. + np.exp(-logits))           # sigmoid

        for pic, pb in zip(batch_paths, probs):
            # 1. threshold filtering
            accepted_indices = np.where(pb >= args.confidence_threshold)[0]
            if not len(accepted_indices):
                continue

            accepted_tags = [runner.idx2tag[j] for j in accepted_indices]

            # 2. write CSV line
            writer.writerow([
                running_id,
                f"images/{pic.name}",
                " ".join(str(j) for j in accepted_indices),
                " ".join(accepted_tags)
            ])
            running_id += 1

            # 3. optional caption side‑car
            if sidecar_dir:
                _write_sidecar(sidecar_dir, pic, accepted_tags)

            # 4. copy image
            shutil.copy2(pic, img_out_dir / pic.name)

    csv_file.close()
    print(f"✓ All done – {running_id} samples written → {csv_path}")


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a Camie‑Tagger training CSV from a "
                    "directory of images."
    )

    # I/O
    parser.add_argument("--input-dir",  required=True,
                        help="Folder that holds *only* images.")
    parser.add_argument("--output-dir", required=True,
                        help="Destination for images + CSV.")
    parser.add_argument("--skip-sidecar", action="store_true",
                        help="Do **not** write individual <name>.txt "
                             "files next to the CSV.")

    # HF credentials
    parser.add_argument("--hf-token", required=True,
                        help="Hugging Face auth token.")

    # Inference parameters
    parser.add_argument("--confidence-threshold", type=float, default=0.35,
                        help="Minimum probability to include a tag.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Images per batch.")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable half‑precision inference.")
    parser.add_argument("--model-repo", default="Camais03/camie-tagger",
                        help="Model repository name.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Inference device.")

    build_dataset(parser.parse_args())
