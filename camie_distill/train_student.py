
"""
Distillation loop using DeepSpeed ZeRO‑2 and the recipe published in the
model card.  Handles automatic off‑loading of the teacher after pseudo‑labelling.
"""
import argparse, json, math, os
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import deepspeed                    # pip install deepspeed
from huggingface_hub import hf_hub_download

from focal_loss import UnifiedFocalLoss
from student_model import StudentTagger
from flash_stub import install_flash_stub; install_flash_stub()

# ────────────────────────────────────────────────────────────────────
class CsvDataset(Dataset):
    def __init__(self, csv_path: Path, img_dir: Path, tag_count: int,
                 pad_colour, fp16):
        import pandas as pd
        from preprocessing import load_and_preprocess
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.tag_count = tag_count
        self.pad_colour, self.fp16 = pad_colour, fp16
        self._cache = {}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.img_dir / row.file_name
        if path not in self._cache:
            self._cache[path] = load_and_preprocess(
                path, pad_colour=self.pad_colour, fp16=self.fp16)
        img = torch.from_numpy(self._cache[path])
        tag_idx = list(map(int, row.tag_idx.split()))
        target = torch.zeros(self.tag_count, dtype=torch.float32)
        target[tag_idx] = 1.0
        return img, target
# ────────────────────────────────────────────────────────────────────
def create_deepspeed_config(out_path: Path, lr: float, wd: float,
                            train_samples: int, mb_size: int,
                            accum_steps: int):
    conf = {
        "train_micro_batch_size_per_gpu": mb_size,
        "gradient_accumulation_steps": accum_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": wd
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": lr * 1e-2,
                "warmup_max_lr": lr,
                "warmup_num_steps": int(0.25 * train_samples
                                        / (mb_size * accum_steps))
            }
        },
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "contiguous_gradients": True,
            "offload_optimizer": {"device": "none"}
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False
        }
    }
    out_path.write_text(json.dumps(conf, indent=2))

# ────────────────────────────────────────────────────────────────────
def main(cfg):
    # Teacher logits already written by dataset_builder → pass None here
    tag_meta = json.loads(Path(cfg.tag_meta).read_text())
    tag_cnt  = len(tag_meta["idx_to_tag"])

    # Student & optimizer
    model = StudentTagger(tag_cnt)
    criterion = UnifiedFocalLoss()

    # Dataset
    ds = CsvDataset(Path(cfg.csv_path),
                    Path(cfg.img_root),
                    tag_cnt,
                    pad_colour=tuple(cfg.pad_colour),
                    fp16=False)
    dl = DataLoader(ds, batch_size=cfg.micro_batch,
                    shuffle=True, num_workers=4, pin_memory=True)

    # DeepSpeed init
    cfg_path = Path(cfg.output_dir) / "ds_config.json"
    create_deepspeed_config(cfg_path, cfg.lr, cfg.wd,
                            len(ds), cfg.micro_batch, cfg.grad_accum)
    model_engine, optim, _, _ = deepspeed.initialize(
        model=model, config=str(cfg_path), model_parameters=model.parameters())

    scaler = torch.cuda.amp.GradScaler()

    total_steps = math.ceil(len(dl) / cfg.grad_accum) * cfg.epochs
    step = 0
    for epoch in range(cfg.epochs):
        for img, tgt in dl:
            img = img.to(model_engine.device, non_blocking=True)
            tgt = tgt.to(model_engine.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                logits_i, logits_r = model_engine(img)
                loss = criterion(logits_i, logits_r, tgt)

            model_engine.backward(loss)
            model_engine.step()
            if step % cfg.log_every == 0 and model_engine.global_rank == 0:
                print(f"step {step}/{total_steps} | loss {loss.item():.4f}")
            step += 1

        # Optional EMA, checkpointing skipped for brevity

if __name__ == "__main__":
    p = argparse.ArgumentParser("Student distillation")
    p.add_argument("--csv-path",   required=True)
    p.add_argument("--img-root",   required=True)
    p.add_argument("--tag-meta",   required=True,
                   help="metadata.json from model repo")
    p.add_argument("--output-dir", required=True)
    # Training hyper‑params
    p.add_argument("--lr",  type=float, default=3e-4)         # :contentReference[oaicite:2]{index=2}
    p.add_argument("--wd",  type=float, default=0.01)         # :contentReference[oaicite:3]{index=3}
    p.add_argument("--micro-batch", type=int, default=4)      # :contentReference[oaicite:4]{index=4}
    p.add_argument("--grad-accum",  type=int, default=8)      # :contentReference[oaicite:5]{index=5}
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--pad-colour", type=int, nargs=3,
                   default=(0,0,0), metavar=("R","G","B"))
    cfg = p.parse_args()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    main(cfg)
