import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

class WaveletTransform2D(nn.Module):
    """
    2D Discrete Wavelet Transform module using PyWavelets.
    Decomposes an RGB or Grayscale image into multiscale frequency bands:
    [LL, LH, HL, HH].
    
    Supports:
    - Wavelet families: 'haar', 'db2', 'db4', etc.
    - Ablation modes:
        - 'all': Concatenates [LL, LH, HL, HH] -> 12 channels (RGB)
        - 'll': Only approximation sub-band [LL] -> 3 channels (RGB)
        - 'hf': Only high-frequency sub-bands [LH, HL, HH] -> 9 channels (RGB)
    """
    def __init__(self, wavelet='haar', mode='all', target_size=(224, 224)):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode.lower()
        self.target_size = target_size
        
        # Determine number of output channels
        if self.mode == 'all':
            self.out_channels = 12 # 3 RGB channels * 4 sub-bands
        elif self.mode == 'll':
            self.out_channels = 3  # 3 RGB channels * 1 sub-band (LL)
        elif self.mode == 'hf':
            self.out_channels = 9  # 3 RGB channels * 3 sub-bands (LH, HL, HH)
        else:
            raise ValueError(f"Unsupported wavelet mode: {mode}. Choose from ['all', 'll', 'hf']")

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, C, H, W] with values in [0, 1] or normalized.
        Returns:
            Tensor of shape [B, out_channels, target_size[0], target_size[1]]
        """
        device = x.device
        x_np = x.detach().cpu().numpy() # [B, C, H, W]
        
        batch_transformed = []
        for b in range(x_np.shape[0]):
            img_channels = []
            for c in range(x_np.shape[1]):
                channel_data = x_np[b, c]
                # 2D DWT level 1
                coeffs2 = pywt.dwt2(channel_data, self.wavelet)
                LL, (LH, HL, HH) = coeffs2
                
                # Normalize each sub-band individually to stable [0, 1] range
                def norm_band(b_data):
                    b_min, b_max = b_data.min(), b_data.max()
                    if b_max - b_min > 1e-6:
                        return (b_data - b_min) / (b_max - b_min)
                    return b_data - b_min

                LL_n = norm_band(LL)
                LH_n = norm_band(LH)
                HL_n = norm_band(HL)
                HH_n = norm_band(HH)
                
                if self.mode == 'all':
                    img_channels.extend([LL_n, LH_n, HL_n, HH_n])
                elif self.mode == 'll':
                    img_channels.append(LL_n)
                elif self.mode == 'hf':
                    img_channels.extend([LH_n, HL_n, HH_n])
            
            # Stack sub-bands along channel dimension -> [out_channels, H_dwt, W_dwt]
            stacked = np.stack(img_channels, axis=0)
            batch_transformed.append(stacked)
        
        # Tensor [B, out_channels, H_dwt, W_dwt]
        out_tensor = torch.tensor(np.array(batch_transformed), dtype=torch.float32, device=device)
        
        # Upsample back to target resolution (224x224) to maintain compatibility with ViT patch grids
        if (out_tensor.shape[2], out_tensor.shape[3]) != self.target_size:
            out_tensor = F.interpolate(out_tensor, size=self.target_size, mode='bilinear', align_corners=False)
            
        return out_tensor
