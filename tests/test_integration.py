# tests/test_integration.py

import pytest
import torch
import torch.optim as optim
from models.prototypes import PrototypeNet
from utils.loss_functions import PrototypeLoss
from utils.data_loader import get_data_loaders
import os


def test_end_to_end_training(tmp_path):
    # Configuration
    batch_size = 4
    epochs = 1
    model = PrototypeNet(num_classes=10, feature_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = PrototypeLoss()
    data_loaders = get_data_loaders(batch_size=batch_size)

    # Training Loop
    model.train()
    for epoch in range(epochs):
        for inputs, labels in data_loaders['train']:
            logits, features = model(inputs)
            loss = criterion(logits, features, labels, model.prototypes)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break  # Run only one batch for the test
        break  # Run only one epoch for the test


def test_model_save_load(tmp_path):
    model = PrototypeNet(num_classes=10, feature_dim=128)
    save_path = tmp_path / "prototype_net.pth"
    torch.save(model.state_dict(), save_path)
    assert os.path.exists(save_path)
    # Load the model
    loaded_model = PrototypeNet(num_classes=10, feature_dim=128)
    loaded_model.load_state_dict(torch.load(save_path))
    # Check that parameters are the same
    for param1, param2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(param1, param2)
