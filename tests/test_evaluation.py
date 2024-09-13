# tests/test_evaluation.py

import pytest
import torch
from utils.evaluation import calculate_accuracy, calculate_auroc, get_confusion_matrix

def test_accuracy():
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    labels = torch.tensor([1, 0])
    acc = calculate_accuracy(outputs, labels)
    assert acc == 100.0

def test_auroc():
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    labels = torch.tensor([1, 0])
    auroc = calculate_auroc(outputs, labels)
    assert 0.0 <= auroc <= 1.0

def test_confusion_matrix():
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    labels = torch.tensor([1, 0])
    cm = get_confusion_matrix(outputs, labels)
    assert cm.shape == (2, 2)
