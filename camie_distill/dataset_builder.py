from __future__ import annotations
import argparse, csv, json, shutil
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch import amp
from tqdm import tqdm
from safetensors.torch import load_file as safe_load
from huggingface_hub import hf_hub_download

from flash_stub import install_flash_stub
from preprocessing import load_and_preprocess, IMAGE_EXTS

install_flash_stub()                       # ensure import before model code

# ────────────────────────────────────────────────────────────────────
def _lazy_import_arch(repo: str, *, token: str | None):
    code_path = hf_hub_download(repo, "model/model_code.py",
                                token=token, revision="main")  # pin SHAs if desired
    import importlib.util
    spec = importlib.util.spec_from_file_location("camie_model", code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Guarantee flash_attn_func exists
    if "flash_attn_func" not in module.__dict__:
        module.flash_attn_func = builtins.flash_attn_func
    return module
# ────────────────────────────────────────────────────────────────────
class SafetensorRunner:
    def __init__(self, repo: str, device: str, fp16: bool, *, token: str | None):
        ckpt = hf_hub_download(repo, "model_refined.safetensors", token=token)
        meta = json.loads(Path(hf_hub_download(repo,
                               "model/model_info_refined.json",
                               token=token)).read_text())
        tag_meta = json.loads(Path(hf_hub_download(repo,
                                   "model/metadata.json",
                                   token=token)).read_text())

        self.idx2tag: Dict[int, str] = {int(k): v
                                        for k, v in tag_meta["idx_to_tag"].items()}
        self.tag2idx: Dict[str, int] = {v: k for k, v in self.idx2tag.items()}

        arch = _lazy_import_arch(repo, token=token)
        dataset = arch.TagDataset(total_tags=len(self.idx2tag),
                                  idx_to_tag=self.idx2tag,
                                  tag_to_category=tag_meta["tag_to_category"])

        self.model = arch.ImageTagger(
            total_tags=len(self.idx2tag),
            dataset=dataset,
            num_heads=meta.get("num_heads", 16),
            tag_context_size=meta.get("tag_context_size", 256),
            pretrained=False
        )

        self.model.load_state_dict(safe_load(ckpt, device="cpu"), strict=False)
        self.model.to(device).eval()
        if fp16:
            self.model.half()

        self.device, self.fp16 = device, fp16

    @torch.no_grad()
    def __call__(self, batch: List[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(batch)).to(self.device)
        with amp.autocast(device_type=self.device, enabled=self.fp16):
            _, refined = self.model(x)
        return refined.cpu().numpy()

# ────────────────────────────────────────────────────────────────────
def build_dataset(cfg):
    runner = SafetensorRunner(cfg.model_repo, cfg.device, cfg.fp16,
                              token=cfg.hf_token)

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    img_out_dir = out_root / "images"; img_out_dir.mkdir(exist_ok=True)

    csv_path    = out_root / "train_dataset.csv"
    sidecar_dir = None if cfg.skip_sidecar else out_root

    images = sorted(p for p in Path(cfg.input_dir).rglob("*")
                    if p.suffix.lower() in IMAGE_EXTS)
    if not images: raise SystemExit("No images found.")

    writer = csv.writer(csv_path.open("w", newline='', encoding="utf-8"))
    writer.writerow(["id", "file_name", "tag_idx", "tag"])

    running_id = 0
    for i in tqdm(range(0, len(images), cfg.batch_size), unit="batch"):
        paths = images[i:i+cfg.batch_size]
        batch = [load_and_preprocess(p, fp16=cfg.fp16,
                                     pad_colour=tuple(cfg.pad_colour))
                 for p in paths]

        logits = runner(batch)
        probs  = 1. / (1. + np.exp(-logits))  # sigmoid

        for pic, pb in zip(paths, probs):
            sel = np.where(pb >= cfg.confidence_threshold)[0]
            if not len(sel): continue
            tags = [runner.idx2tag[j] for j in sel]

            writer.writerow([running_id, f"images/{pic.name}",
                             " ".join(map(str, sel)), " ".join(tags)])
            running_id += 1

            if sidecar_dir:
                (sidecar_dir / f"{pic.stem}.txt").write_text(
                    ", ".join(tags), encoding="utf-8")

            shutil.copy2(pic, img_out_dir / pic.name)

    print(f"✓ {running_id} samples written to {csv_path}")

# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Camie‑Tagger pseudo‑label generator")
    # I/O
    parser.add_argument("--input-dir",  required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-sidecar", action="store_true")
    # HF
    parser.add_argument("--hf-token", required=True)
    # Inference
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--model-repo", default="Camais03/camie-tagger")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    # New argument
    parser.add_argument("--pad-colour", type=int, nargs=3, default=(0,0,0),
                        metavar=("R","G","B"),
                        help="Letter‑box colour (default: black)")
    cfg = parser.parse_args()
    build_dataset(cfg)
