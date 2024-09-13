# utils/loss_functions.py

import torch
import torch.nn as nn

class PrototypeLoss(nn.Module):
    """Combined loss function for prototype learning."""

    def __init__(self, alpha=0.1):
        """Initializes the PrototypeLoss.

        Args:
            alpha (float): Weighting factor for prototype loss.
        """
        super(PrototypeLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, logits, features, labels, prototypes):
        """Computes the combined loss.

        Args:
            logits (torch.Tensor): Output logits.
            features (torch.Tensor): Feature embeddings.
            labels (torch.Tensor): Ground truth labels.
            prototypes (torch.Tensor): Class prototypes.

        Returns:
            torch.Tensor: Combined loss.
        """
        ce_loss = self.cross_entropy(logits, labels)
        proto_loss = torch.mean((features - prototypes[labels]) ** 2)
        total_loss = ce_loss + self.alpha * proto_loss
        return total_loss
