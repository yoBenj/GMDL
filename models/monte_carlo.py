# models/prototypes.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeNet(nn.Module):
    """Neural network model for prototype learning."""

    def __init__(self, num_classes=10, feature_dim=128, backbone=None):
        """Initializes the PrototypeNet model.

        Args:
            num_classes (int): Number of classes.
            feature_dim (int): Dimension of the feature embedding.
            backbone (nn.Module): Backbone network for feature extraction.
        """
        super(PrototypeNet, self).__init__()
        if backbone is None:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                # Add more layers as needed
            )
        else:
            self.backbone = backbone

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(64 * 16 * 16, feature_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
            torch.Tensor: Feature embeddings.
        """
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        logits = -torch.cdist(features, self.prototypes)
        return logits, features
