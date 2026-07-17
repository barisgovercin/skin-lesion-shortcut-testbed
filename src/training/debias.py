"""Debiasing components for leading-language shortcut mitigation."""
import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """Identity forward; negates (scaled) gradient on backward."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class AdversarialTextHead(nn.Module):
    """Predicts the label from the text embedding through a gradient-reversal
    layer, discouraging the shared representation from encoding the
    leading-language shortcut."""

    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, text_feat, lambd=1.0):
        return self.net(GradientReversal.apply(text_feat, lambd)).squeeze(-1)


def consistency_loss(logit_concordant, logit_flipped):
    """Penalise prediction change when leading language is flipped (invariance)."""
    p_c = torch.sigmoid(logit_concordant)
    p_f = torch.sigmoid(logit_flipped)
    return ((p_c - p_f) ** 2).mean()
