import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class ToothCropDataset(Dataset):
    """
    PyTorch Dataset for individual tooth crop images organized in class subdirectories.
    Structure:
        data/crops/{split}/class_{0..8}/*.jpg
    """
    def __init__(self, root_dir, split='train', transform=None, wavelet_transform=None, preload_memory=True):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.wavelet_transform = wavelet_transform
        self.preload_memory = preload_memory
        
        self.samples = []
        self.class_counts = {c: 0 for c in range(9)}
        
        for cls_id in range(9):
            cls_dir = os.path.join(self.root_dir, f'class_{cls_id}')
            if os.path.exists(cls_dir):
                files = glob.glob(os.path.join(cls_dir, '*.jpg')) + glob.glob(os.path.join(cls_dir, '*.png'))
                for f in files:
                    self.samples.append((f, cls_id))
                    self.class_counts[cls_id] += 1
        
        # Default transforms if none provided
        if transform is None:
            if split == 'train':
                self.transform = T.Compose([
                    T.Resize((224, 224)),
                    T.RandomRotation(degrees=10),
                    T.ColorJitter(brightness=0.15, contrast=0.15),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        # Precompute and cache all tensors in RAM for ultra-fast GPU throughput
        self.cached_tensors = []
        if self.preload_memory:
            from tqdm import tqdm
            for img_path, _ in tqdm(self.samples, desc=f"Loading & Wavelet precompute ({split})", leave=False):
                with Image.open(img_path) as img:
                    img_rgb = img.convert('RGB')
                    t = self.transform(img_rgb)
                    if self.wavelet_transform is not None:
                        with torch.no_grad():
                            t = self.wavelet_transform(t.unsqueeze(0)).squeeze(0)
                    self.cached_tensors.append(t)
            print(f"Precomputation for {split} complete! ({len(self.cached_tensors)} tensors ready in RAM)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preload_memory:
            return self.cached_tensors[idx], self.samples[idx][1]
        
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(image) # [3, 224, 224]
        
        if self.wavelet_transform is not None:
            with torch.no_grad():
                img_tensor = self.wavelet_transform(img_tensor.unsqueeze(0)).squeeze(0)
                
        return img_tensor, label
