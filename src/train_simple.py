"""
Simple self-contained training script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm
import json
import albumentations as A

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODEL_CONFIG, MODELS_DIR, TILES_DIR

# ============================================================
# MODEL
# ============================================================

def create_model(in_channels=5, num_classes=4):
    """Create U-Net model"""
    model = smp.Unet(
        encoder_name='resnet18',
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
        activation=None
    )
    return model

# ============================================================
# DATASET
# ============================================================

class LandCoverDataset(Dataset):
    """Dataset for land cover"""
    
    def __init__(self, split='train', augment=True):
        self.split = split
        self.augment = augment and (split == 'train')
        
        # Load split
        with open(TILES_DIR / 'split_info.json', 'r') as f:
            split_info = json.load(f)
        self.tile_names = split_info[split]
        
        self.images_dir = TILES_DIR / 'images'
        self.masks_dir = TILES_DIR / 'masks'
        
        # Augmentation
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)
            ])
        else:
            self.transform = None
        
        print(f"✅ {split} dataset: {len(self.tile_names)} tiles")
    
    def __len__(self):
        return len(self.tile_names)
    
    def __getitem__(self, idx):
        tile_name = self.tile_names[idx]
        
        image = np.load(self.images_dir / tile_name)
        mask = np.load(self.masks_dir / tile_name)
        
        # Transpose for augmentation
        image = np.transpose(image, (1, 2, 0))
        
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        
        # Normalize
        for i in range(image.shape[2]):
            ch = image[:, :, i]
            if ch.std() > 0:
                image[:, :, i] = (ch - ch.mean()) / (ch.std() + 1e-8)
        
        # Back to CHW
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

# ============================================================
# TRAINER
# ============================================================

def train_model():
    """Main training function"""
    
    print("="*60)
    print("NYC URBAN SUSTAINABILITY - MODEL TRAINING")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = create_model(in_channels=5, num_classes=4)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    train_dataset = LandCoverDataset('train', augment=True)
    val_dataset = LandCoverDataset('val', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 30
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("\n" + "="*60)
    print(f"STARTING TRAINING - {num_epochs} EPOCHS")
    print("="*60 + "\n")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        # TRAIN
        model.train()
        train_loss = 0
        train_acc = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy
            _, pred = torch.max(outputs, 1)
            acc = (pred == masks).float().mean().item() * 100
            train_acc += acc
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.1f}%'})
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # VALIDATE
        model.eval()
        val_loss = 0
        val_acc = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                _, pred = torch.max(outputs, 1)
                val_acc += (pred == masks).float().mean().item() * 100
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, MODELS_DIR / 'best_model.pth')
            print(f"  💾 Saved best model!")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best val loss: {best_val_loss:.4f}")
    
    # Save history
    with open(MODELS_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history

if __name__ == "__main__":
    train_model()