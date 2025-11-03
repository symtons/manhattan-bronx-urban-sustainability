"""
PyTorch Dataset for Land Cover Segmentation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import TILES_DIR

class LandCoverDataset(Dataset):
    """
    Dataset for land cover segmentation
    """
    
    def __init__(self, split='train', augment=True):
        """
        Initialize dataset
        
        Args:
            split: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
        """
        self.split = split
        self.augment = augment and (split == 'train')
        
        # Load split info
        split_file = TILES_DIR / 'split_info.json'
        with open(split_file, 'r') as f:
            split_info = json.load(f)
        
        self.tile_names = split_info[split]
        
        self.images_dir = TILES_DIR / 'images'
        self.masks_dir = TILES_DIR / 'masks'
        
        # Setup augmentation
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
            ])
        else:
            self.transform = None
        
        print(f"✅ {split.capitalize()} dataset initialized: {len(self.tile_names)} tiles")
        if self.augment:
            print("   With data augmentation enabled")
    
    def __len__(self):
        return len(self.tile_names)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            image: Tensor of shape (5, H, W) - RGB + NIR + NDVI
            mask: Tensor of shape (H, W) - class labels
        """
        tile_name = self.tile_names[idx]
        
        # Load image and mask
        image_path = self.images_dir / tile_name
        mask_path = self.masks_dir / tile_name
        
        image = np.load(image_path)  # Shape: (5, H, W)
        mask = np.load(mask_path)    # Shape: (H, W)
        
        # Transpose image to (H, W, C) for albumentations
        image = np.transpose(image, (1, 2, 0))  # (H, W, 5)
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Normalize image (optional - helps training)
        # Each channel normalized independently
        for i in range(image.shape[2]):
            channel = image[:, :, i]
            if channel.std() > 0:
                image[:, :, i] = (channel - channel.mean()) / (channel.std() + 1e-8)
        
        # Transpose back to (C, H, W) for PyTorch
        image = np.transpose(image, (2, 0, 1))
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask

def get_dataloaders(batch_size=8, num_workers=0):
    """
    Create train, validation, and test dataloaders
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = LandCoverDataset(split='train', augment=True)
    val_dataset = LandCoverDataset(split='val', augment=False)
    test_dataset = LandCoverDataset(split='test', augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n✅ DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    """Test dataset loading"""
    
    print("Testing dataset...")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=2)
    
    # Test loading a batch
    for images, masks in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Masks: {masks.shape}")
        print(f"  Image range: [{images.min():.2f}, {images.max():.2f}]")
        print(f"  Mask unique values: {torch.unique(masks).tolist()}")
        break
    
    print("\n✅ Dataset test successful!")