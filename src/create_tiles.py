"""
Create training tiles from processed rasters
Generates image/mask pairs for U-Net training
"""

import rasterio
from rasterio.windows import Window
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR, TILES_DIR, TILE_CONFIG, WORLDCOVER_PARAMS

class TileGenerator:
    """
    Generate training tiles from processed rasters
    """
    
    def __init__(self):
        """Initialize tile generator"""
        self.tile_size = TILE_CONFIG['size']
        self.overlap = TILE_CONFIG['overlap']
        self.min_valid = TILE_CONFIG['min_valid_pixels']
        
        # Setup directories
        self.images_dir = TILES_DIR / 'images'
        self.masks_dir = TILES_DIR / 'masks'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Tile generator initialized")
        print(f"   Tile size: {self.tile_size}×{self.tile_size}")
        print(f"   Overlap: {self.overlap} pixels")
    
    def reclassify_worldcover(self, worldcover_data):
        """
        Reclassify WorldCover 11 classes into 4 simplified classes
        
        WorldCover classes → Our classes:
        10 (tree_cover), 20 (shrubland), 30 (grassland) → 0 (vegetation)
        80 (water), 90 (wetland), 95 (mangroves) → 1 (water)
        50 (built_up) → 2 (built_up)
        40 (cropland), 60 (bare), 70 (snow), 100 (moss) → 3 (bare/open)
        """
        
        # Create output array
        output = np.zeros_like(worldcover_data, dtype=np.uint8)
        
        # Vegetation (0)
        vegetation_classes = [10, 20, 30]
        for cls in vegetation_classes:
            output[worldcover_data == cls] = 0
        
        # Water (1)
        water_classes = [80, 90, 95]
        for cls in water_classes:
            output[worldcover_data == cls] = 1
        
        # Built-up (2)
        output[worldcover_data == 50] = 2
        
        # Bare/Open (3)
        bare_classes = [40, 60, 70, 100]
        for cls in bare_classes:
            output[worldcover_data == cls] = 3
        
        return output
    
    def is_valid_tile(self, tile_data):
        """
        Check if tile has enough valid pixels
        
        Args:
            tile_data: numpy array
            
        Returns:
            bool: True if tile is valid
        """
        # Check for nodata values (-9999)
        valid_pixels = np.sum(tile_data != -9999)
        total_pixels = tile_data.size
        
        valid_ratio = valid_pixels / total_pixels
        
        return valid_ratio >= self.min_valid
    
    def create_tiles(self):
        """
        Create tiles from all aligned rasters
        """
        print("\n" + "="*60)
        print("GENERATING TRAINING TILES")
        print("="*60)
        
        # File paths - UPDATED TO USE FIXED FILES
        landsat_path = PROCESSED_DATA_DIR / 'landsat_merged.tif'
        ndvi_path = PROCESSED_DATA_DIR / 'ndvi_fixed.tif'
        lst_path = PROCESSED_DATA_DIR / 'lst_fixed.tif'
        worldcover_path = PROCESSED_DATA_DIR / 'worldcover_fixed.tif'
        
        # Check if files exist
        for path in [landsat_path, ndvi_path, lst_path, worldcover_path]:
            if not path.exists():
                print(f"❌ File not found: {path}")
                print("   Please run 'python src/fix_alignment.py' first")
                return 0
        
        # Open all rasters
        print("\nOpening rasters...")
        landsat_src = rasterio.open(landsat_path)
        ndvi_src = rasterio.open(ndvi_path)
        lst_src = rasterio.open(lst_path)
        worldcover_src = rasterio.open(worldcover_path)
        
        print(f"Landsat shape: {landsat_src.shape}")
        print(f"NDVI shape: {ndvi_src.shape}")
        print(f"LST shape: {lst_src.shape}")
        print(f"WorldCover shape: {worldcover_src.shape}")
        
        # Check if dimensions match
        if not (landsat_src.shape == ndvi_src.shape == lst_src.shape == worldcover_src.shape):
            print("\n❌ ERROR: Raster dimensions don't match!")
            print("   Please run 'python src/fix_alignment.py' first")
            return 0
        
        # Use NDVI dimensions as reference
        height, width = ndvi_src.shape
        
        # Calculate number of tiles
        stride = self.tile_size - self.overlap
        n_tiles_y = (height - self.tile_size) // stride + 1
        n_tiles_x = (width - self.tile_size) // stride + 1
        total_possible = n_tiles_y * n_tiles_x
        
        print(f"\nGenerating tiles:")
        print(f"  Image dimensions: {height} × {width}")
        print(f"  Tile size: {self.tile_size}")
        print(f"  Stride: {stride}")
        print(f"  Possible tiles: {total_possible}")
        
        tile_count = 0
        valid_tiles = 0
        
        # Generate tiles
        for i in tqdm(range(n_tiles_y), desc="Rows"):
            for j in range(n_tiles_x):
                # Calculate window
                row_start = i * stride
                col_start = j * stride
                
                # Make sure we don't exceed bounds
                if row_start + self.tile_size > height:
                    row_start = height - self.tile_size
                if col_start + self.tile_size > width:
                    col_start = width - self.tile_size
                
                window = Window(col_start, row_start, self.tile_size, self.tile_size)
                
                # Read NDVI tile to check validity
                ndvi_tile = ndvi_src.read(1, window=window)
                
                # Check if tile is valid
                if not self.is_valid_tile(ndvi_tile):
                    tile_count += 1
                    continue
                
                try:
                    # Read all bands for this tile
                    # Landsat: B2, B3, B4, B5 (Blue, Green, Red, NIR)
                    landsat_tile = landsat_src.read(window=window)
                    
                    # Get RGB and NIR (skip NDVI band from Landsat, we use calculated one)
                    blue = landsat_tile[0]
                    green = landsat_tile[1]
                    red = landsat_tile[2]
                    nir = landsat_tile[3]
                    
                    # Stack: RGB + NIR + NDVI = 5 channels
                    image_tile = np.stack([red, green, blue, nir, ndvi_tile], axis=0)
                    
                    # Read WorldCover label
                    worldcover_tile = worldcover_src.read(1, window=window)
                    
                    # Reclassify to 4 classes
                    mask_tile = self.reclassify_worldcover(worldcover_tile)
                    
                    # Save tiles
                    tile_name = f"tile_{i:04d}_{j:04d}.npy"
                    
                    np.save(self.images_dir / tile_name, image_tile.astype(np.float32))
                    np.save(self.masks_dir / tile_name, mask_tile.astype(np.uint8))
                    
                    valid_tiles += 1
                    
                except Exception as e:
                    print(f"\n  ⚠️ Error processing tile ({i}, {j}): {e}")
                
                tile_count += 1
        
        # Close all rasters
        landsat_src.close()
        ndvi_src.close()
        lst_src.close()
        worldcover_src.close()
        
        print(f"\n✅ Tile generation complete!")
        print(f"   Total tiles processed: {tile_count}")
        print(f"   Valid tiles saved: {valid_tiles}")
        print(f"   Invalid tiles skipped: {tile_count - valid_tiles}")
        print(f"   Images saved to: {self.images_dir}")
        print(f"   Masks saved to: {self.masks_dir}")
        
        return valid_tiles
    
    def split_train_val_test(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split tiles into train/validation/test sets
        """
        print("\n" + "="*60)
        print("SPLITTING INTO TRAIN/VAL/TEST")
        print("="*60)
        
        # Get all tile files
        image_files = sorted(list(self.images_dir.glob("tile_*.npy")))
        
        if len(image_files) == 0:
            print("❌ No tiles found!")
            return
        
        if len(image_files) < 10:
            print(f"⚠️  Only {len(image_files)} tiles found - this is very few for training")
            print("   Consider:")
            print("   1. Reducing tile size further in config/settings.py")
            print("   2. Reducing overlap")
            print("   3. Using data augmentation during training")
        
        # Shuffle
        random.seed(42)
        random.shuffle(image_files)
        
        # Calculate splits
        n_tiles = len(image_files)
        n_train = int(n_tiles * train_ratio)
        n_val = int(n_tiles * val_ratio)
        
        # Ensure at least 1 sample in each split if possible
        if n_tiles >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        print(f"Total tiles: {n_tiles}")
        print(f"  Training: {len(train_files)} ({len(train_files)/n_tiles*100:.1f}%)")
        print(f"  Validation: {len(val_files)} ({len(val_files)/n_tiles*100:.1f}%)")
        print(f"  Testing: {len(test_files)} ({len(test_files)/n_tiles*100:.1f}%)")
        
        # Save split info
        split_info = {
            'train': [f.name for f in train_files],
            'val': [f.name for f in val_files],
            'test': [f.name for f in test_files]
        }
        
        import json
        split_file = TILES_DIR / 'split_info.json'
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"\n✅ Split information saved to: {split_file}")
        
        return split_info

def main():
    """Main function"""
    print("="*60)
    print("NYC URBAN SUSTAINABILITY - TILE GENERATION")
    print("="*60)
    
    generator = TileGenerator()
    
    # Generate tiles
    n_tiles = generator.create_tiles()
    
    if n_tiles > 0:
        # Split into train/val/test
        generator.split_train_val_test()
        
        print("\n" + "="*60)
        print("🎉 TILE GENERATION COMPLETE!")
        print("="*60)
        
        if n_tiles >= 50:
            print("✅ Good number of tiles for training!")
            print("Next step: Train the U-Net model")
        elif n_tiles >= 20:
            print("⚠️  Moderate number of tiles - training will work but might benefit from augmentation")
            print("Next step: Train with data augmentation")
        else:
            print("⚠️  Few tiles - will use heavy data augmentation during training")
            print("Next step: Train with strong augmentation")
        
        print("="*60)
    else:
        print("\n❌ No valid tiles generated!")
        print("Please check raster alignment and dimensions.")

if __name__ == "__main__":
    main()