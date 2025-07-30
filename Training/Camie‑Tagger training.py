#!/usr/bin/env python3
"""
Finetune Camie‑Tagger on new images + (optional) replay, with
empirical‑Fisher EWC regularisation and focal loss.

▪ No teacher‑student KD is used (kd_lambda hard‑wired to 0).
▪ Tested on a single RTX 5090 (31.5 GiB) with bf16 and DeepSpeed ZeRO‑2.
"""

# ────────────────────────────────────────────────────────────────────────────────
# Imports & CLI
# ────────────────────────────────────────────────────────────────────────────────
import argparse, json, math, os, random, time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import deepspeed

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--base_ckpt',   required=True,  help='camie_v0.pt')
p.add_argument('--output_ckpt', required=True)
p.add_argument('--data_root',   required=True,  help='folder with new_images/  new_tags.jsonl')
p.add_argument('--old_data_root', default=None,
               help='(optional) few original images for replay')
p.add_argument('--replay_ratio', type=float, default=0.1,
               help='fraction of each sample drawn from legacy set')
p.add_argument('--epochs', type=float, default=1.0)
p.add_argument('--ewc_lambda',  type=float, default=0.3)
p.add_argument('--focal_gamma', type=float, default=2.0)
p.add_argument('--legacy_downweight', type=float, default=0.2)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--micro_bs', type=int, default=6)
p.add_argument('--grad_accum', type=int, default=8)
p.add_argument('--warmup_pct', type=float, default=0.02)
p.add_argument('--fisher_path', default='fisher_cls.pt')
args = p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
bf16  = torch.bfloat16
device = torch.device('cuda')

# ────────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────────────────────
seed = 42 + int(os.getenv("RANK", "0"))
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# ────────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────────
def find_last_linear(model: nn.Module):
    """Return (module_ref, attribute_name) of the last nn.Linear (classifier)."""
    last_key, last_mod = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_key, last_mod = name, mod
    if last_key is None:
        raise RuntimeError("No nn.Linear layer found in model; cannot expand classifier.")
    # Traverse to parent
    parent = model
    for tok in last_key.split('.')[:-1]:
        parent = getattr(parent, tok)
    return parent, last_key.split('.')[-1], last_mod

def expand_classifier(model: nn.Module, num_labels: int):
    parent, attr, old = find_last_linear(model)
    if old.out_features >= num_labels:
        return  # Already big enough
    new_fc = nn.Linear(old.in_features, num_labels)
    with torch.no_grad():
        new_fc.weight[:old.out_features].copy_(old.weight)
        new_fc.bias[:old.out_features].copy_(old.bias)
    setattr(parent, attr, new_fc)

def set_trainable(module: nn.Module, trainable: bool):
    for p in module.parameters():
        p.requires_grad_(trainable)

# ────────────────────────────────────────────────────────────────────────────────
# Load Camie
# ────────────────────────────────────────────────────────────────────────────────
def load_camie(path: str, num_labels: int = None, freeze: bool = False):
    mdl = torch.load(path, map_location='cpu')
    if num_labels is not None:
        expand_classifier(mdl, num_labels)
    if freeze:
        set_trainable(mdl, False)
    return mdl

# ────────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────────
class TagDataset(Dataset):
    def __init__(self, manifest: Path, tag2idx: dict, img_root: Path,
                 augment: bool = True):
        self.recs = [json.loads(l) for l in open(manifest, 'r', encoding='utf-8')]
        self.tag2idx = tag2idx
        self.root = Path(img_root)
        if augment:
            self.tf = T.Compose([
                T.RandomResizedCrop(512, scale=(0.8, 1.0),
                                    interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                T.ToTensor()
            ])
        else:
            self.tf = T.Compose([
                T.Resize(512, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(512),
                T.ToTensor()
            ])

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        r = self.recs[idx]
        img = Image.open(self.root / r['image']).convert('RGB')
        img = self.tf(img)
        y   = torch.zeros(len(self.tag2idx), dtype=torch.float32)
        for t in r['tags']:
            if t in self.tag2idx:
                y[self.tag2idx[t]] = 1.
        return img, y

# ────────────────────────────────────────────────────────────────────────────────
# Empirical Fisher diagonal
# ────────────────────────────────────────────────────────────────────────────────
def compute_fisher(model: nn.Module, dataloader: DataLoader, cls_param_names):
    """Empirical Fisher diag over selected params."""
    fisher = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in model.named_parameters()
              if n in cls_param_names}
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, dtype=bf16)
            y = y.to(device, dtype=bf16)
            with torch.cuda.amp.autocast(dtype=bf16):
                logits = model(x)[0] if isinstance(model(x), tuple) else model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y, reduction='sum')
            grads = torch.autograd.grad(
                loss,
                [p for n, p in model.named_parameters() if n in cls_param_names],
                retain_graph=False
            )
            for (n, _), g in zip(filter(lambda np_: np_[0] in cls_param_names,
                                        model.named_parameters()), grads):
                fisher[n] += (g.detach().float() ** 2)
    for n in fisher:
        fisher[n] /= len(dataloader.dataset)
    return fisher

# ────────────────────────────────────────────────────────────────────────────────
# Focal loss
# ────────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels, weight=None):
        p  = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        pt = labels * p + (1 - labels) * (1 - p)
        loss = ((1 - pt) ** self.gamma) * ce
        if weight is not None:
            loss = loss * weight
        return loss.mean()

# ────────────────────────────────────────────────────────────────────────────────
# Tag maps
# ────────────────────────────────────────────────────────────────────────────────
root = Path(args.data_root)
tag2idx_old = json.load(open(root / 'tag_maps' / 'tag2idx_old.json'))
tag2idx_new = json.load(open(root / 'tag_maps' / 'tag2idx_new.json'))
tag2idx = {**tag2idx_old, **tag2idx_new}           # order preserved in CPython>=3.7
num_labels = len(tag2idx)
old_mask   = torch.zeros(num_labels, dtype=torch.bool)
old_mask[:len(tag2idx_old)] = True

# ────────────────────────────────────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────────────────────────────────────
# Student model (trainable)
model = load_camie(args.base_ckpt, num_labels=num_labels).to(device, dtype=bf16)
model.train()

# Frozen reference for Fisher & EWC (snapshot of initial weights)
reference = load_camie(args.base_ckpt, num_labels=num_labels, freeze=True).to(device, dtype=bf16)

# Track which parameters get regularised
cls_param_names = {n for n, _ in model.named_parameters() if n.endswith('.weight') or n.endswith('.bias')}
init_params = {n: p.detach().clone() for n, p in model.named_parameters() if n in cls_param_names}

# ────────────────────────────────────────────────────────────────────────────────
# Data loaders
# ────────────────────────────────────────────────────────────────────────────────
new_ds = TagDataset(root / 'new_tags.jsonl', tag2idx, root / 'new_images', augment=True)

if args.old_data_root and args.replay_ratio > 0:
    old_root = Path(args.old_data_root)
    old_ds   = TagDataset(old_root / 'tags.jsonl', tag2idx_old, old_root / 'images', augment=True)

    class Mixed(Dataset):
        def __len__(self):  # choose epoch length as new_ds length
            return len(new_ds)

        def __getitem__(self, i):
            if random.random() < args.replay_ratio:
                return old_ds[random.randrange(len(old_ds))]
            return new_ds[i]
    train_ds = Mixed()
else:
    train_ds = new_ds

# Small validation set: last 5 % of new_ds
val_split = max(1, int(0.05 * len(new_ds)))
val_ds    = torch.utils.data.Subset(new_ds, list(range(-val_split, 0)))
train_ds2 = torch.utils.data.Subset(train_ds, list(range(0, len(train_ds) - val_split)))

dl_train = DataLoader(train_ds2, batch_size=args.micro_bs, shuffle=True,
                      num_workers=8, pin_memory=True, drop_last=True)
dl_val   = DataLoader(val_ds,  batch_size=args.micro_bs*2, shuffle=False,
                      num_workers=4, pin_memory=True)

# ────────────────────────────────────────────────────────────────────────────────
# Fisher computation or load
# ────────────────────────────────────────────────────────────────────────────────
if Path(args.fisher_path).exists():
    fisher = torch.load(args.fisher_path, map_location='cpu')
    print(f"[Info] Loaded Fisher from {args.fisher_path}")
else:
    print("[Info] Computing Fisher diagonal on reference model…")
    fisher_loader = DataLoader(new_ds, batch_size=4, shuffle=False, num_workers=4)
    fisher = compute_fisher(reference, fisher_loader, cls_param_names)
    torch.save(fisher, args.fisher_path)
    print(f"[Info] Saved Fisher to {args.fisher_path}")

# ────────────────────────────────────────────────────────────────────────────────
# DeepSpeed engine
# ────────────────────────────────────────────────────────────────────────────────
deepspeed.init_distributed()

optimizer_grouped_params = [p for p in model.parameters() if p.requires_grad]
engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=optimizer_grouped_params,
    config={
        "train_batch_size": args.micro_bs * args.grad_accum,
        "gradient_accumulation_steps": args.grad_accum,
        "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01}},
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2},
        "gradient_clipping": 1.0
    }
)

total_steps = math.ceil(len(dl_train) * args.epochs / args.grad_accum)
warmup_steps = max(1, int(args.warmup_pct * total_steps))
lr_sched = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(1.0, step / warmup_steps) * 0.5 * (1 + math.cos(math.pi * step / total_steps))
)

focal = FocalLoss(gamma=args.focal_gamma)

# ────────────────────────────────────────────────────────────────────────────────
# Training / validation helpers
# ────────────────────────────────────────────────────────────────────────────────
def evaluate(model_eval: nn.Module):
    model_eval.eval()
    correct, total = 0., 0.
    with torch.no_grad():
        for x, y in dl_val:
            x = x.to(device, dtype=bf16)
            y = y.to(device, dtype=bf16)
            with torch.cuda.amp.autocast(dtype=bf16):
                logits = model_eval(x)[0] if isinstance(model_eval(x), tuple) else model_eval(x)
                preds  = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds.eq(y).float()).sum().item()
            total   += torch.numel(y)
    model_eval.train()
    return correct / total if total else 0.0

# ────────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────────
step, global_step = 0, 0
for epoch in range(math.ceil(args.epochs)):
    for x, y in dl_train:
        x = x.to(engine.local_rank, dtype=bf16)
        y = y.to(engine.local_rank, dtype=bf16)

        with torch.cuda.amp.autocast(dtype=bf16):
            logits = engine(x)[0] if isinstance(engine(x), tuple) else engine(x)

            # Focal loss (weighted)
            w = torch.ones_like(y)
            w[:, old_mask] = args.legacy_downweight
            lfocal = focal(logits, y, w)

            # Elastic Weight Consolidation
            lewc = 0.
            for n, p in engine.module.named_parameters():
                if n in cls_param_names:
                    if n in fisher:
                        diff = p - init_params[n].to(p.device)
                        lewc += (fisher[n].to(p.dtype).to(p.device) * diff.pow(2)).sum()
            lewc = lewc * args.ewc_lambda / len(fisher)

            loss = lfocal + lewc

        engine.backward(loss)
        engine.step()
        lr_sched.step()

        global_step += 1
        if engine.is_gradient_accumulation_boundary() and engine.local_rank == 0:
            if global_step % 100 == 0:
                print(f"{time.strftime('%H:%M:%S')}  step {global_step:>6}  "
                      f"loss {loss.item():.4f}  lfocal {lfocal.item():.4f}  lewc {lewc.item():.4f}")

            if global_step % 1000 == 0:
                val_acc = evaluate(engine.module)
                print(f"[Val] step {global_step}  exact‑match accuracy: {val_acc:.4%}")

# ────────────────────────────────────────────────────────────────────────────────
# Save checkpoint
# ────────────────────────────────────────────────────────────────────────────────
if engine.local_rank == 0:
    torch.save({
        'model_state_dict': engine.module.state_dict(),
        'tag2idx': tag2idx
    }, args.output_ckpt)
    print(f"[Info] Saved fine‑tuned model to {args.output_ckpt}")
