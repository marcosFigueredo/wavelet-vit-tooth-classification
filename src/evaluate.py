import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
import pandas as pd

from src.utils import get_device, calculate_metrics, plot_confusion_matrix
from src.wavelet import WaveletTransform2D
from src.dataset import ToothCropDataset
from src.models import WaveletViT

def evaluate_model(checkpoint_path, data_dir='data/crops', split='test', output_dir='results'):
    device = get_device()
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint['config']
    
    exp_name = cfg.get('experiment_name', 'evaluation')
    wavelet_cfg = cfg.get('wavelet', None)
    wavelet_trans = None
    in_channels = 3
    
    if wavelet_cfg and wavelet_cfg.get('enabled', False):
        wavelet_name = wavelet_cfg.get('name', 'haar')
        mode = wavelet_cfg.get('mode', 'all')
        wavelet_trans = WaveletTransform2D(wavelet=wavelet_name, mode=mode)
        in_channels = wavelet_trans.out_channels
        
    dataset = ToothCropDataset(data_dir, split=split, wavelet_transform=wavelet_trans)
    loader = DataLoader(dataset, batch_size=cfg.get('batch_size', 16), shuffle=False, num_workers=0)
    
    model = WaveletViT(model_name=cfg.get('model_name', 'vit_b_16'), in_channels=in_channels, num_classes=9, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    metrics = calculate_metrics(y_true, y_pred, num_classes=9)
    print(f"\n=== Test Evaluation Results for {exp_name} ===")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:  {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:     {metrics['weighted_f1']:.4f}")
    
    # Save metrics JSON
    metrics_path = os.path.join(output_dir, 'tables', f'{exp_name}_test_metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
        
    # Save per-class metrics CSV
    per_class_df = pd.DataFrame.from_dict(metrics['per_class'], orient='index')
    per_class_df.index.name = 'class_id'
    per_class_csv = os.path.join(output_dir, 'tables', f'{exp_name}_per_class_metrics.csv')
    per_class_df.to_csv(per_class_csv)
    
    # Plot and save 300 DPI Confusion Matrix
    cm_path = os.path.join(output_dir, 'figures', f'{exp_name}_confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path, title=f'Confusion Matrix - {exp_name}')
    print(f"Saved test metrics to {metrics_path} and confusion matrix to {cm_path}")
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint .pth')
    parser.add_argument('--split', type=str, default='test', help='Split to evaluate on')
    args = parser.parse_args()
    evaluate_model(args.checkpoint, split=args.split)
