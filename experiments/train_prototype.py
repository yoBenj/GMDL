# experiments/train_prototype.py

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from models.prototypes import PrototypeNet
from utils.data_loader import get_data_loaders
from utils.loss_functions import PrototypeLoss
from utils.log_utils import set_logger
from utils.evaluation import calculate_accuracy
import logging

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set up logging
    set_logger(config['logging']['log_file'])

    # Get data loaders
    data_loaders = get_data_loaders(batch_size=config['training']['batch_size'],
                                    augmentation=config['dataset']['augmentation'])

    # Initialize model
    model = PrototypeNet(num_classes=config['model']['num_classes'],
                         feature_dim=config['model']['feature_dim'])
    model = model.to(config['device'])

    # Define loss function and optimizer
    criterion = PrototypeLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in data_loaders['train']:
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])

            optimizer.zero_grad()
            outputs, features = model(inputs)
            loss = criterion(outputs, features, labels, model.prototypes)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loaders['train'].dataset)
        logging.info(f"Epoch {epoch+1}/{config['training']['epochs']}, Loss: {epoch_loss:.4f}")

        # Evaluation
        if (epoch + 1) % config['logging']['eval_interval'] == 0:
            acc = evaluate(model, data_loaders['test'], config)
            logging.info(f"Test Accuracy: {acc:.2f}%")

    # Save the model
    torch.save(model.state_dict(), config['logging']['checkpoint_path'])

def evaluate(model, test_loader, config):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    main('experiments/config/prototype_config.yaml')
