import torch


class PrototypeNet:
    def __init__(self, num_classes=10, feature_dim=128):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.prototypes = torch.randn(num_classes, feature_dim)

    def __call__(self, x):
        return torch.randn(x.size(0), self.num_classes), torch.randn(x.size(0), self.feature_dim)