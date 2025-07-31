import torch, torch.nn as nn, torch.nn.functional as F

class UnifiedFocalLoss(nn.Module):
    """
    Multi‑label focal loss with two‑stage weighting, mirroring the teacher’s
    recipe (γ = 2.0, α = 0.25, λ_initial = 0.4).:contentReference[oaicite:1]{index=1}
    """
    def __init__(self, gamma=2.0, alpha=0.25, lambda_initial=0.4):
        super().__init__()
        self.gamma, self.alpha, self.lambda_initial = gamma, alpha, lambda_initial

    def _focal(self, logits, targets):
        prob = torch.sigmoid(logits)
        ce   = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        p_t  = prob*targets + (1-prob)*(1-targets)
        mod  = (1 - p_t) ** self.gamma
        alpha_t = self.alpha*targets + (1-self.alpha)*(1-targets)
        return (alpha_t * mod * ce).mean()

    def forward(self, logits_init, logits_ref, targets):
        l_init = self._focal(logits_init, targets)
        l_ref  = self._focal(logits_ref,  targets)
        return self.lambda_initial * l_init + (1-self.lambda_initial) * l_ref
