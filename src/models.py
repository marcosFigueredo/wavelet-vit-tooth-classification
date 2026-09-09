import torch
import torch.nn as nn
import torchvision.models as models

class WaveletViT(nn.Module):
    """
    Vision Transformer model adapted for Wavelet Multiscale Representations.
    
    Can operate as:
    1. Baseline ViT (RGB, in_channels=3)
    2. Wavelet-ViT (Haar, db2, db4 with in_channels=12, 9, or 3)
    
    Weights for new input channels are smartly initialized from the pre-trained
    ImageNet convolutional patch projection to ensure smooth transfer learning.
    """
    def __init__(self, model_name='vit_b_16', in_channels=3, num_classes=9, pretrained=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Load backbone
        if model_name == 'vit_b_16':
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.vit = models.vit_b_16(weights=weights)
            embed_dim = self.vit.heads.head.in_features
            patch_size = 16
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")
        
        # Adapt patch projection layer for arbitrary input channels
        if in_channels != 3:
            orig_proj = self.vit.conv_proj
            new_proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=orig_proj.out_channels,
                kernel_size=orig_proj.kernel_size,
                stride=orig_proj.stride,
                padding=orig_proj.padding,
                bias=(orig_proj.bias is not None)
            )
            
            if pretrained:
                # Initialize new filters by averaging and repeating pre-trained RGB weights
                with torch.no_grad():
                    orig_weight = orig_proj.weight # [embed_dim, 3, 16, 16]
                    # Average over 3 RGB channels -> [embed_dim, 1, 16, 16]
                    avg_weight = orig_weight.mean(dim=1, keepdim=True)
                    # Repeat for in_channels and scale appropriately
                    new_weight = avg_weight.repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
                    new_proj.weight.copy_(new_weight)
                    if orig_proj.bias is not None:
                        new_proj.bias.copy_(orig_proj.bias)
                        
            self.vit.conv_proj = new_proj
            
        # Replace classification head
        self.vit.heads.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.vit(x)
