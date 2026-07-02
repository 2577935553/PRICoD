"""Uncertainty-Rectified Consistency (URC).

Disagreement-aware cross-network consistency on the (refined) logits. Pixels
where the two networks agree (high cosine similarity) are up-weighted via
``exp(cos - 1) = exp(-U)``; the ``(1 - cos) = U`` term penalises global
disagreement.
"""
import torch
import torch.nn.functional as F


def urc_loss(inputs, targets):
    """
    Uncertainty rectified pseudo supervised loss
    """
    # detach from the computational graph
    pseudo_label = torch.argmax(F.softmax(targets, dim=1).detach(), dim=1)
    vanilla_loss = F.cross_entropy(inputs, pseudo_label.long(), reduction='none')
    # uncertainty rectification
    cos_similarity = F.cosine_similarity(inputs, targets.detach(), dim=1)

    loss = (torch.exp(cos_similarity - 1) * vanilla_loss).mean() + (1 - cos_similarity).mean()
    return loss

# Backward-compatible alias.
uncertainty_loss = urc_loss
