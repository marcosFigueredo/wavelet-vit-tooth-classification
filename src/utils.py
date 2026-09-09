import os
import random
import json
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def set_seed(seed=42):
    """Set random seed for full reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """Return available compute device (cuda if available else cpu)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def compute_class_weights(counts_dict, num_classes=9):
    """
    Compute balanced class weights: w_k = N / (K * N_k).
    Returns torch.FloatTensor of size [num_classes].
    """
    total = sum(counts_dict.values())
    weights = []
    for c in range(num_classes):
        cnt = counts_dict.get(c, 1)
        w = total / (num_classes * cnt)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)

def calculate_metrics(y_true, y_pred, num_classes=9):
    """
    Calculate standard evaluation metrics for multi-class classification.
    Returns dictionary with overall and per-class metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    return {
        'accuracy': float(acc),
        'macro_precision': float(prec_macro),
        'macro_recall': float(rec_macro),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
        'per_class': {
            c: {
                'precision': float(prec_cls[c]),
                'recall': float(rec_cls[c]),
                'f1': float(f1_cls[c]),
                'support': int(support[c])
            } for c in range(num_classes)
        },
        'confusion_matrix': cm.tolist()
    }

def plot_confusion_matrix(cm, class_names=None, save_path=None, title='Confusion Matrix', dpi=300):
    """Generate and save a publication-quality 300 DPI confusion matrix."""
    cm_arr = np.array(cm)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=dpi)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm_arr))]
    
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Sample Count'}, ax=ax, linewidths=0.5)
    
    ax.set_xlabel('Predicted Tooth Class', labelpad=10, fontweight='bold')
    ax.set_ylabel('True Tooth Class', labelpad=10, fontweight='bold')
    ax.set_title(title, pad=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
    plt.close()

def plot_training_history(history, save_path=None, dpi=300):
    """Plot loss and macro-F1 learning curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=dpi)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    axes[0].plot(epochs, history['val_loss'], 'r--', label='Val Loss', linewidth=1.5)
    axes[0].set_title('Cross-Entropy Loss Across Epochs', fontweight='bold')
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Macro F1
    axes[1].plot(epochs, history['train_f1'], 'b-', label='Train Macro-F1', linewidth=1.5)
    axes[1].plot(epochs, history['val_f1'], 'r--', label='Val Macro-F1', linewidth=1.5)
    axes[1].set_title('Macro-F1 Score Progression', fontweight='bold')
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Macro-F1', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
    plt.close()
