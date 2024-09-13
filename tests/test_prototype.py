# tests/test_prototype.py

import pytest
import torch
import torch.nn as nn
from models.prototypes import PrototypeNet
from utils.loss_functions import PrototypeLoss


@pytest.fixture
def prototype_net():
    """Fixture to initialize the PrototypeNet model."""
    model = PrototypeNet(num_classes=10, feature_dim=128)
    return model


def test_prototype_net_initialization(prototype_net):
    """Test if the PrototypeNet model initializes correctly."""
    assert isinstance(prototype_net, nn.Module)
    assert prototype_net.num_classes == 10
    assert prototype_net.feature_dim == 128
    assert prototype_net.prototypes.shape == (10, 128)


def test_prototype_net_forward(prototype_net):
    """Test the forward pass of the PrototypeNet model."""
    inputs = torch.randn(8, 3, 32, 32)  # Batch size of 8
    logits, features = prototype_net(inputs)
    assert logits.shape == (8, 10), "Logits shape mismatch"
    assert features.shape == (8, 128), "Features shape mismatch"


def test_prototype_net_backward(prototype_net):
    """Test the backward pass of the PrototypeNet model."""
    inputs = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    logits, features = prototype_net(inputs)
    criterion = PrototypeLoss()
    loss = criterion(logits, features, labels, prototype_net.prototypes)
    loss.backward()
    for name, param in prototype_net.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"


def test_prototype_net_prototype_update(prototype_net):
    """Test if prototypes are updated correctly during training."""
    # Before training
    initial_prototypes = prototype_net.prototypes.clone()
    optimizer = torch.optim.Adam(prototype_net.parameters(), lr=0.001)
    inputs = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    criterion = PrototypeLoss()

    # Perform one training step
    optimizer.zero_grad()
    logits, features = prototype_net(inputs)
    loss = criterion(logits, features, labels, prototype_net.prototypes)
    loss.backward()
    optimizer.step()

    # After training
    updated_prototypes = prototype_net.prototypes
    # Check that prototypes have been updated
    assert not torch.equal(initial_prototypes, updated_prototypes), "Prototypes did not update"


def test_prototype_net_inference(prototype_net):
    """Test inference mode of the PrototypeNet model."""
    prototype_net.eval()
    inputs = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        logits, features = prototype_net(inputs)
    assert logits.shape == (8, 10), "Logits shape mismatch during inference"
    assert features.shape == (8, 128), "Features shape mismatch during inference"


def test_prototype_net_save_load(tmp_path, prototype_net):
    """Test saving and loading the PrototypeNet model."""
    save_path = tmp_path / "prototype_net.pth"
    torch.save(prototype_net.state_dict(), save_path)
    assert save_path.exists(), "Model save failed"

    # Load the model
    loaded_model = PrototypeNet(num_classes=10, feature_dim=128)
    loaded_model.load_state_dict(torch.load(save_path))

    # Verify that parameters are the same
    for param1, param2 in zip(prototype_net.parameters(), loaded_model.parameters()):
        assert torch.allclose(param1, param2), "Parameters mismatch after loading"
