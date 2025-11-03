"""
Training script for U-Net Land Cover Segmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODEL_CONFIG, MODELS_DIR
# Try to import from current directory first
try:
    from model import get_model
    from dataset import get_dataloaders
except ImportError:
    # If that fails, try importing from src
    from src.model import get_model
    from src.dataset import get_dataloaders

class Trainer:
    """
    Handles model training
    """
    
    def __init__(self, config):
        """Initialize trainer"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model, _ = get_model(config)
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer,
        mode='min',
        factor=0.5,
        patience=5
)
        
        # Get dataloaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            batch_size=config['batch_size'],
            num_workers=0
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        print(f"\n✅ Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Training samples: {len(self.train_loader.dataset)}")
        print(f"   Validation samples: {len(self.val_loader.dataset)}")
    
    def calculate_accuracy(self, outputs, masks):
        """Calculate pixel accuracy"""
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == masks).sum().item()
        total = masks.numel()
        return 100 * correct / total
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_acc = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            acc = self.calculate_accuracy(outputs, masks)
            running_acc += acc
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = running_acc / len(self.train_loader)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        running_acc = 0.0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                running_loss += loss.item()
                running_acc += self.calculate_accuracy(outputs, masks)
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = running_acc / len(self.val_loader)
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = MODELS_DIR / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = MODELS_DIR / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"   💾 Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def train(self, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print("="*60 + "\n")
        
        start_time = datetime.now()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            # Save checkpoint
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.epochs_no_improve >= self.config['early_stopping_patience']:
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                break
        
        # Training complete
        elapsed = datetime.now() - start_time
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Total time: {elapsed}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        
        # Save history
        history_path = MODELS_DIR / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n📊 Training history saved to: {history_path}")
        
        return self.history

def main():
    """Main training function"""
    
    # Load config
    config = MODEL_CONFIG.copy()
    
    # Adjust for dataset
    config['num_epochs'] = 30
    config['batch_size'] = 4
    
    print("="*60)
    print("NYC URBAN SUSTAINABILITY - MODEL TRAINING")
    print("="*60)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train model
    history = trainer.train()
    
    print("\n🎉 Training pipeline complete!")

if __name__ == "__main__":
    main()