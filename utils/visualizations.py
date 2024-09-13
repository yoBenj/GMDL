# utils/visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def plot_training_curves(train_losses, val_losses, save_path):
    """Plots training and validation loss curves.

    Args:
        train_losses (list): Training losses.
        val_losses (list): Validation losses.
        save_path (str): Path to save the plot.
    """
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)

def plot_confusion_matrix(conf_matrix, class_names, save_path):
    """Plots a confusion matrix.

    Args:
        conf_matrix (numpy.ndarray): Confusion matrix.
        class_names (list): List of class names.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)

def plot_roc_curve(labels, scores, save_path):
    """Plots ROC curve.

    Args:
        labels (numpy.ndarray): True binary labels.
        scores (numpy.ndarray): Predicted scores.
        save_path (str): Path to save the plot.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(save_path)
