# utils/evaluation.py

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np

def calculate_accuracy(outputs, labels):
    """Calculates accuracy.

    Args:
        outputs (torch.Tensor): Model outputs.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: Accuracy percentage.
    """
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return 100 * correct / total

def calculate_auroc(outputs, labels):
    """Calculates AUROC for binary classification.

    Args:
        outputs (torch.Tensor): Model outputs.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: AUROC score.
    """
    probs = F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return roc_auc_score(labels, probs)

def get_confusion_matrix(outputs, labels):
    """Generates confusion matrix.

    Args:
        outputs (torch.Tensor): Model outputs.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        numpy.ndarray: Confusion matrix.
    """
    _, preds = torch.max(outputs, 1)
    return confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
