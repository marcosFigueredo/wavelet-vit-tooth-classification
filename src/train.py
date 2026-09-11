import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import set_seed, prevent_sleep, get_device, load_config, compute_class_weights, calculate_metrics, plot_training_history
from src.wavelet import WaveletTransform2D
from src.dataset import ToothCropDataset
from src.models import WaveletViT

def train_model(config_path):
    prevent_sleep()
    if torch.cuda.is_available() is False:
        num_cores = os.cpu_count() or 8
        torch.set_num_threads(num_cores)
        print(f"Configured PyTorch CPU threads: {num_cores}")
        
    cfg = load_config(config_path)
    set_seed(cfg.get('seed', 42))
    device = get_device()
    print(f"Using compute device: {device}")
    
    # Wavelet transform if configured
    wavelet_cfg = cfg.get('wavelet', None)
    wavelet_trans = None
    in_channels = 3
    
    if wavelet_cfg and wavelet_cfg.get('enabled', False):
        wavelet_name = wavelet_cfg.get('name', 'haar')
        mode = wavelet_cfg.get('mode', 'all')
        wavelet_trans = WaveletTransform2D(wavelet=wavelet_name, mode=mode)
        in_channels = wavelet_trans.out_channels
        print(f"Initialized WaveletTransform2D ({wavelet_name}, mode={mode}) -> {in_channels} input channels")
    else:
        print("Training Baseline ViT with standard RGB input (3 channels)")
        
    # Datasets and Loaders
    data_dir = cfg.get('data_dir', 'data/crops')
    train_dataset = ToothCropDataset(data_dir, split='train', wavelet_transform=wavelet_trans)
    val_dataset = ToothCropDataset(data_dir, split='val', wavelet_transform=wavelet_trans)
    
    batch_size = cfg.get('batch_size', 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model_name = cfg.get('model_name', 'vit_b_16')
    model = WaveletViT(model_name=model_name, in_channels=in_channels, num_classes=9, pretrained=cfg.get('pretrained', True))
    model.to(device)
    
    # Loss with class weighting
    if cfg.get('weighted_loss', True):
        class_weights = compute_class_weights(train_dataset.class_counts, num_classes=9).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using Class-Weighted Cross-Entropy Loss.")
    else:
        criterion = nn.CrossEntropyLoss()
        
    # Optimizer & Scheduler
    lr = float(cfg.get('lr', 1e-4))
    weight_decay = float(cfg.get('weight_decay', 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    epochs = cfg.get('epochs', 25)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # Tracking
    best_val_f1 = 0.0
    patience = cfg.get('patience', 8)
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    exp_name = cfg.get('experiment_name', 'experiment')
    save_dir = os.path.join('results/checkpoints', exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nStarting training for {exp_name} ({epochs} epochs with AMP={device.type == 'cuda'})...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        y_train_true, y_train_pred = [], []
        
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            y_train_true.extend(labels.cpu().numpy())
            y_train_pred.extend(preds.cpu().numpy())
            
        scheduler.step()
        train_loss /= len(train_dataset)
        train_metrics = calculate_metrics(y_train_true, y_train_pred)
        
        # Validation
        model.eval()
        val_loss = 0.0
        y_val_true, y_val_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                y_val_true.extend(labels.cpu().numpy())
                y_val_pred.extend(preds.cpu().numpy())
                
        val_loss /= len(val_dataset)
        val_metrics = calculate_metrics(y_val_true, y_val_pred)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['macro_f1'])
        history['val_f1'].append(val_metrics['macro_f1'])
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} - Macro-F1: {train_metrics['macro_f1']:.4f} | Val Loss: {val_loss:.4f} - Macro-F1: {val_metrics['macro_f1']:.4f} (Acc: {val_metrics['accuracy']:.4f})")
        
        # Checkpoint
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': cfg
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  --> Saved new best model with Val Macro-F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    # Save history and curves
    with open(os.path.join(save_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    plot_training_history(history, save_path=os.path.join('results/figures', f'{exp_name}_learning_curves.png'))
    print(f"\nTraining complete. Best Validation Macro-F1: {best_val_f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    train_model(args.config)
