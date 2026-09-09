import os
import glob
import json
import random
from PIL import Image
import numpy as np

def prepare_splits_and_crops(
    image_dir='code/images',
    label_dir='code/labels',
    output_dir='data/crops',
    split_info_dir='data/splits',
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    margin_ratio=0.08,
    target_size=(224, 224),
    seed=42
):
    """
    Executes Etapas 3, 4, and 5:
    1. Discovers unique base stems to avoid data leakage.
    2. Splits stems into train (70%), val (15%), test (15%).
    3. Extracts bounding box tooth crops with 8% margin.
    4. Resizes to 224x224 and organizes by class into data/crops/{train,val,test}/class_{0..8}/.
    5. Saves split manifest and statistics.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(split_info_dir, exist_ok=True)
    
    # Discover unique stems
    lbl_files = glob.glob(os.path.join(label_dir, '*.txt'))
    stem_to_files = {}
    
    for lbf in lbl_files:
        fname = os.path.basename(lbf)
        stem = fname.split('.rf.')[0] if '.rf.' in fname else fname.split('.')[0]
        if stem not in stem_to_files:
            stem_to_files[stem] = []
        stem_to_files[stem].append(fname)
        
    unique_stems = sorted(list(stem_to_files.keys()))
    random.shuffle(unique_stems)
    
    n_total = len(unique_stems)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_stems = set(unique_stems[:n_train])
    val_stems = set(unique_stems[n_train:n_train + n_val])
    test_stems = set(unique_stems[n_train + n_val:])
    
    print(f"Total unique capture stems: {n_total}")
    print(f"Train stems: {len(train_stems)}, Val stems: {len(val_stems)}, Test stems: {len(test_stems)}")
    
    # Create class directories for each split
    splits = ['train', 'val', 'test']
    for sp in splits:
        for c in range(9):
            os.makedirs(os.path.join(output_dir, sp, f'class_{c}'), exist_ok=True)
            
    stats = {sp: {c: 0 for c in range(9)} for sp in splits}
    crop_manifest = []
    
    for lbf in lbl_files:
        lbl_fname = os.path.basename(lbf)
        stem = lbl_fname.split('.rf.')[0] if '.rf.' in lbl_fname else lbl_fname.split('.')[0]
        
        if stem in train_stems:
            current_split = 'train'
        elif stem in val_stems:
            current_split = 'val'
        else:
            current_split = 'test'
            
        # Match image file
        img_name = lbl_fname.replace('.txt', '.jpg')
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            img_name_alt = lbl_fname.replace('.txt', '.png')
            img_path = os.path.join(image_dir, img_name_alt)
            if not os.path.exists(img_path):
                continue
                
        image = Image.open(img_path).convert('RGB')
        img_w, img_h = image.size
        
        with open(lbf, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
            
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            
            # Pixel bounding box
            center_x = xc * img_w
            center_y = yc * img_h
            box_w = w * img_w
            box_h = h * img_h
            
            # Apply margin (8%)
            margin_w = box_w * margin_ratio
            margin_h = box_h * margin_ratio
            
            xmin = max(0, int(center_x - (box_w / 2) - margin_w))
            xmax = min(img_w, int(center_x + (box_w / 2) + margin_w))
            ymin = max(0, int(center_y - (box_h / 2) - margin_h))
            ymax = min(img_h, int(center_y + (box_h / 2) + margin_h))
            
            if xmax <= xmin or ymax <= ymin:
                continue
                
            crop = image.crop((xmin, ymin, xmax, ymax))
            crop = crop.resize(target_size, Image.Resampling.LANCZOS)
            
            crop_filename = f"{lbl_fname.replace('.txt', '')}_tooth_{i}_cls{cls_id}.jpg"
            crop_save_path = os.path.join(output_dir, current_split, f'class_{cls_id}', crop_filename)
            crop.save(crop_save_path, quality=95)
            
            stats[current_split][cls_id] += 1
            crop_manifest.append({
                'crop_filename': crop_filename,
                'source_stem': stem,
                'split': current_split,
                'class_id': cls_id,
                'path': crop_save_path
            })
            
    # Save split manifest and statistics
    with open(os.path.join(split_info_dir, 'split_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'train_stems': sorted(list(train_stems)),
            'val_stems': sorted(list(val_stems)),
            'test_stems': sorted(list(test_stems)),
            'stats': stats,
            'total_crops': len(crop_manifest)
        }, f, indent=2)
        
    print("\n=== Dataset Splitting & Tooth Extraction Complete ===")
    print(f"Total tooth crops generated: {len(crop_manifest)}")
    for sp in splits:
        total_sp = sum(stats[sp].values())
        print(f"[{sp.upper()}] Total: {total_sp} crops | Distribution: {stats[sp]}")
        
    return stats

if __name__ == '__main__':
    prepare_splits_and_crops()
