import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import prevent_sleep
from src.train import train_model
from src.evaluate import evaluate_model

def run_all():
    prevent_sleep()
    print("=" * 60)
    print("STARTING FULL WAVELET-VIT EXPERIMENTAL PIPELINE")
    print("=" * 60)
    
    experiments = [
        ('configs/haar_vit.yaml', 'haar_vit'),
        ('configs/db2_vit.yaml', 'db2_vit'),
        ('configs/db4_vit.yaml', 'db4_vit'),
        ('configs/ablation_haar_ll.yaml', 'ablation_haar_ll'),
        ('configs/ablation_haar_hf.yaml', 'ablation_haar_hf')
    ]
    
    for config_path, exp_name in experiments:
        print(f"\n>>> [STARTING EXPERIMENT] {exp_name} ({config_path})")
        try:
            train_model(config_path)
            
            # Evaluate best checkpoint on test set
            chk_path = os.path.join('results/checkpoints', exp_name, 'best_model.pth')
            if os.path.exists(chk_path):
                print(f"\n>>> [EVALUATING TEST SET] {exp_name}")
                evaluate_model(chk_path, split='test')
            else:
                print(f"Warning: Checkpoint not found at {chk_path}")
        except Exception as e:
            print(f"Error during experiment {exp_name}: {e}")
            
    # Consolidate all test metrics into master comparison table
    print("\n" + "=" * 60)
    print("CONSOLIDATING FINAL TEST COMPARISON BENCHMARK")
    print("=" * 60)
    
    all_models = [
        ('Baseline ViT (RGB)', 'vit_rgb_baseline'),
        ('Haar-ViT (Wavelet)', 'haar_vit'),
        ('db2-ViT (Wavelet)', 'db2_vit'),
        ('db4-ViT (Wavelet)', 'db4_vit'),
        ('Ablation: LL Only', 'ablation_haar_ll'),
        ('Ablation: HF Only', 'ablation_haar_hf')
    ]
    
    rows = []
    for model_label, exp_name in all_models:
        json_path = os.path.join('results/tables', f'{exp_name}_test_metrics.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            rows.append({
                'Model': model_label,
                'Accuracy': data['accuracy'],
                'Macro_Precision': data['macro_precision'],
                'Macro_Recall': data['macro_recall'],
                'Macro_F1': data['macro_f1'],
                'Weighted_F1': data['weighted_f1']
            })
            
    if rows:
        df_summary = pd.DataFrame(rows)
        summary_csv = 'results/tables/table_all_models_comparison.csv'
        df_summary.to_csv(summary_csv, index=False)
        print("\n=== CONSOLIDATED TEST RESULTS TABLE ===")
        print(df_summary.to_string(index=False))
        
        # Plot 300 DPI Comparison Bar Chart (Figure 5)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 11
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        
        palette = sns.color_palette('tab10', len(df_summary))
        bars = ax.bar(df_summary['Model'], df_summary['Macro_F1'] * 100, color=palette, edgecolor='black', alpha=0.85)
        ax.set_ylabel('Test Macro-F1 Score (%)', fontweight='bold', labelpad=8)
        ax.set_title('Comparative Performance: Baseline ViT vs. Wavelet-ViT Models', fontweight='bold', pad=12)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=15, ha='right', fontweight='bold')
        
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold', fontsize=9.5)
                        
        plt.tight_layout()
        fig_path = 'results/figures/fig5_models_macro_f1_comparison.png'
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"\nGenerated 300 DPI comparative chart: {fig_path}")

if __name__ == '__main__':
    run_all()
