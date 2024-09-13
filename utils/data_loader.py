# utils/data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_data_loaders(batch_size, num_workers=4, augmentation=False):
    """Returns data loaders for training and testing datasets.

    Args:
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of worker threads for loading data.
        augmentation (bool): Whether to apply data augmentation.

    Returns:
        dict: Dictionary containing 'train' and 'test' data loaders.
    """
    transform_list = []
    if augmentation:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)
        ])
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.CIFAR10(root='data/raw', train=True,
                                     download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='data/raw', train=False,
                                    download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return {'train': train_loader, 'test': test_loader}
