"""
U-Net Model for Land Cover Segmentation
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class UNetLandCover(nn.Module):
    """U-Net model for land cover classification"""
    
    def __init__(self, in_channels=5, num_classes=4, encoder_name='resnet18'):
        super(UNetLandCover, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        
        # Fix the first conv layer to accept 5 channels
        if in_channels != 3:
            # Get the original first conv layer
            original_conv = self.model.encoder.conv1
            
            # Create new conv layer with correct input channels
            self.model.encoder.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Initialize new weights (copy from original for first 3 channels, random for rest)
            with torch.no_grad():
                # Copy RGB weights
                self.model.encoder.conv1.weight[:, :3, :, :] = original_conv.weight[:, :3, :, :]
                # Initialize extra channels randomly
                nn.init.kaiming_normal_(self.model.encoder.conv1.weight[:, 3:, :, :])
    
    def forward(self, x):
        return self.model(x)

def get_model(config):
    """Create model from config"""
    
    # Device
    if torch.cuda.is_available() and config.get('device') == 'cuda':
        device = torch.device('cuda')
        print("✅ Using GPU")
    else:
        device = torch.device('cpu')
        print("✅ Using CPU")
    
    # Create model
    model = UNetLandCover(
        in_channels=config['input_channels'],
        num_classes=config['num_classes'],
        encoder_name=config['encoder']
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    return model, device