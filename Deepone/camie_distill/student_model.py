import torch.nn as nn
import timm                    # pip install timm==0.9.12
from flash_stub import install_flash_stub; install_flash_stub()

class StudentTagger(nn.Module):
    def __init__(self, total_tags: int, tag_dim: int = 384,
                 num_heads: int = 8, top_k: int = 75):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_s_wp",
                                          pretrained=True, num_classes=0)
        emb_dim = self.backbone.num_features

        # Initial classifier
        self.init_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim//2),
            nn.GELU(),
            nn.Linear(emb_dim//2, total_tags)
        )

        # Tag embedding + cross attention refinement
        self.tag_embed = nn.Embedding(total_tags, tag_dim)
        self.attn = nn.MultiheadAttention(tag_dim, num_heads,
                                          batch_first=True)
        self.ref_head = nn.Sequential(
            nn.LayerNorm(tag_dim),
            nn.Linear(tag_dim, total_tags)
        )
        self.top_k = top_k

    def forward(self, x):
        feats = self.backbone(x)               # (B, F)

        logits_init = self.init_head(feats)    # (B, T)
        # Get top‑K tags as indices
        topk = logits_init.topk(self.top_k, dim=-1).indices          # (B, K)
        tag_tokens = self.tag_embed(topk)                             # (B, K, D)

        # Attend tags to themselves (context) then cross‑attend to image feat
        ctx, _ = self.attn(tag_tokens, tag_tokens, tag_tokens)
        fused  = ctx.mean(1)                                           # (B, D)
        logits_ref = self.ref_head(fused)

        return logits_init, logits_ref
