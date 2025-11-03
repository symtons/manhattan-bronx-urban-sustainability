"""
Balanced training with class weights to fix imbalance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import segmentation_models_pytorch as smp
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm
import json
import albumentations as A
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODEL_CONFIG, MODELS_DIR, TILES_DIR

# ============================================================
# FOCAL LOSS (handles class imbalance better than CrossEntropy)
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses on hard examples
    """
    def __init__(self, alpha=None, gamma=2, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, 
            weight=self.alpha, 
            ignore_index=self.ignore_index,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# ============================================================
# DATASET WITH HEAVY AUGMENTATION
# ============================================================

class LandCoverDataset(Dataset):
    """Dataset with heavy augmentation"""
    
    def __init__(self, split='train', augment=True):
        self.split = split
        self.augment = augment and (split == 'train')
        
        with open(TILES_DIR / 'split_info.json', 'r') as f:
            split_info = json.load(f)
        self.tile_names = split_info[split]
        
        self.images_dir = TILES_DIR / 'images'
        self.masks_dir = TILES_DIR / 'masks'
        
        # HEAVY augmentation for training
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=1),
                    A.MedianBlur(blur_limit=3, p=1),
                ], p=0.3),
            ])
        else:
            self.transform = None
        
        # Calculate class distribution for weighting
        if split == 'train':
            self.class_counts = self._calculate_class_distribution()
            print(f"\n📊 Training data class distribution:")
            for cls, count in self.class_counts.items():
                print(f"   Class {cls}: {count:,} pixels ({count/sum(self.class_counts.values())*100:.1f}%)")
        
        print(f"✅ {split} dataset: {len(self.tile_names)} tiles")
    
    def _calculate_class_distribution(self):
        """Calculate class distribution across all training tiles"""
        class_counts = Counter()
        
        print("📊 Analyzing class distribution...")
        for tile_name in tqdm(self.tile_names, desc="Scanning tiles"):
            mask = np.load(self.masks_dir / tile_name)
            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                class_counts[cls] += count
        
        return class_counts
    
    def __len__(self):
        return len(self.tile_names)
    
    def __getitem__(self, idx):
        tile_name = self.tile_names[idx]
        
        image = np.load(self.images_dir / tile_name)
        mask = np.load(self.masks_dir / tile_name)
        
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
        
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

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
# TRAINING
# ============================================================

def train_balanced_model():
    """Main training with class balancing"""
    
    print("="*60)
    print("NYC URBAN SUSTAINABILITY - BALANCED MODEL TRAINING")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create datasets
    train_dataset = LandCoverDataset('train', augment=True)
    val_dataset = LandCoverDataset('val', augment=False)
    
    # Calculate class weights (inverse frequency)
    class_counts = train_dataset.class_counts
    total_pixels = sum(class_counts.values())
    
    # Inverse frequency weights
    class_weights = {}
    for cls in range(4):
        if cls in class_counts:
            # Weight = total / (num_classes * class_count)
            class_weights[cls] = total_pixels / (4 * class_counts[cls])
        else:
            class_weights[cls] = 1.0
    
    # Normalize weights
    weight_sum = sum(class_weights.values())
    class_weights = {k: v/weight_sum * 4 for k, v in class_weights.items()}
    
    print(f"\n⚖️  Class weights (to balance training):")
    class_names = {0: 'Vegetation', 1: 'Water', 2: 'Built-up', 3: 'Bare/Open'}
    for cls, weight in class_weights.items():
        print(f"   {class_names[cls]:12s}: {weight:.3f}")
    
    # Convert to tensor
    weight_tensor = torch.FloatTensor([class_weights[i] for i in range(4)]).to(device)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    model = create_model(in_channels=5, num_classes=4)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss with class weights
    criterion = FocalLoss(alpha=weight_tensor, gamma=2)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=5, 
        factor=0.5
    )
    
    # Training loop
    num_epochs = 40  # More epochs since we're balancing
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'class_acc': []  # Track per-class accuracy
    }
    
    print("\n" + "="*60)
    print(f"STARTING BALANCED TRAINING - {num_epochs} EPOCHS")
    print("="*60 + "\n")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        # TRAIN
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            _, pred = torch.max(outputs, 1)
            train_correct += (pred == masks).sum().item()
            train_total += masks.numel()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.1f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # VALIDATE
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        # Per-class accuracy
        class_correct = {i: 0 for i in range(4)}
        class_total = {i: 0 for i in range(4)}
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                _, pred = torch.max(outputs, 1)
                val_correct += (pred == masks).sum().item()
                val_total += masks.numel()
                
                # Per-class stats
                for cls in range(4):
                    cls_mask = masks == cls
                    if cls_mask.sum() > 0:
                        class_correct[cls] += ((pred == cls) & cls_mask).sum().item()
                        class_total[cls] += cls_mask.sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Per-class accuracy
        class_acc = {}
        for cls in range(4):
            if class_total[cls] > 0:
                class_acc[cls] = 100 * class_correct[cls] / class_total[cls]
            else:
                class_acc[cls] = 0
        history['class_acc'].append(class_acc)
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        print(f"  Per-class Val Accuracy:")
        for cls, name in class_names.items():
            print(f"    {name:12s}: {class_acc[cls]:.1f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'class_weights': class_weights
            }, MODELS_DIR / 'best_model_balanced.pth')
            print(f"  💾 Saved best model!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping after {epoch} epochs")
            break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best val loss: {best_val_loss:.4f}")
    
    # Save history
    with open(MODELS_DIR / 'training_history_balanced.json', 'w') as f:
        # Convert class_acc to serializable format
        history_serializable = history.copy()
        history_serializable['class_acc'] = [
            {str(k): v for k, v in epoch_acc.items()} 
            for epoch_acc in history['class_acc']
        ]
        json.dump(history_serializable, f, indent=2)
    
    return model, history

if __name__ == "__main__":
    train_balanced_model()