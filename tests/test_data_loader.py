# tests/test_data_loader.py

import pytest
from utils.data_loader import get_data_loaders
import torch

def test_data_loader_initialization():
    loaders = get_data_loaders(batch_size=32, augmentation=True)
    assert 'train' in loaders and 'test' in loaders

def test_data_shapes():
    loaders = get_data_loaders(batch_size=16)
    data_iter = iter(loaders['train'])
    images, labels = next(data_iter)
    assert images.shape == (16, 3, 32, 32)  # Batch size x Channels x Height x Width
    assert labels.shape == (16,)

def test_data_normalization():
    loaders = get_data_loaders(batch_size=1)
    data_iter = iter(loaders['train'])
    images, _ = next(data_iter)
    mean = images.mean().item()
    std = images.std().item()
    # Assuming normalization to mean=0, std=1
    assert abs(mean) < 0.1
    assert abs(std - 1.0) < 0.1
