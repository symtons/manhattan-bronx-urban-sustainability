"""
Generate predictions using trained model
"""

import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR, OUTPUTS_DIR, MODELS_DIR, LANDCOVER_CLASSES
import segmentation_models_pytorch as smp

def load_model(model_path):
    """Load trained model"""
    
    print("Loading trained model...")
    
    # Create model
    model = smp.Unet(
        encoder_name='resnet18',
        encoder_weights=None,
        in_channels=5,
        classes=4,
        activation=None
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    
    return model

def predict_full_image():
    """Generate predictions for full study area"""
    
    print("\n" + "="*60)
    print("GENERATING LAND COVER PREDICTIONS")
    print("="*60)
    
    # Load model
    model_path = MODELS_DIR / 'best_model.pth'
    model = load_model(model_path)
    device = torch.device('cpu')
    model = model.to(device)
    
    # Load input rasters
    landsat_path = PROCESSED_DATA_DIR / 'landsat_merged.tif'
    ndvi_path = PROCESSED_DATA_DIR / 'ndvi_fixed.tif'
    
    print(f"\nLoading rasters...")
    print(f"  Landsat: {landsat_path}")
    print(f"  NDVI: {ndvi_path}")
    
    with rasterio.open(landsat_path) as landsat_src, \
         rasterio.open(ndvi_path) as ndvi_src:
        
        # Get dimensions
        height, width = landsat_src.shape
        print(f"\nImage dimensions: {width} × {height}")
        
        # Create output array
        prediction = np.zeros((height, width), dtype=np.uint8)
        
        # Tile parameters
        tile_size = 128
        stride = 128  # No overlap for prediction
        
        n_tiles_y = (height - tile_size) // stride + 1
        n_tiles_x = (width - tile_size) // stride + 1
        total_tiles = n_tiles_y * n_tiles_x
        
        print(f"Predicting {total_tiles} tiles...")
        
        # Predict tile by tile
        with torch.no_grad():
            for i in tqdm(range(n_tiles_y), desc="Rows"):
                for j in range(n_tiles_x):
                    # Calculate window
                    row_start = i * stride
                    col_start = j * stride
                    
                    # Adjust if at edge
                    if row_start + tile_size > height:
                        row_start = height - tile_size
                    if col_start + tile_size > width:
                        col_start = width - tile_size
                    
                    window = Window(col_start, row_start, tile_size, tile_size)
                    
                    # Read data
                    landsat_tile = landsat_src.read(window=window)
                    ndvi_tile = ndvi_src.read(1, window=window)
                    
                    # Stack channels (R, G, B, NIR, NDVI)
                    image_tile = np.vstack([
                        landsat_tile[2:3],  # Red
                        landsat_tile[1:2],  # Green
                        landsat_tile[0:1],  # Blue
                        landsat_tile[3:4],  # NIR
                        ndvi_tile[np.newaxis, :]  # NDVI
                    ])
                    
                    # Normalize
                    for c in range(image_tile.shape[0]):
                        ch = image_tile[c]
                        if ch.std() > 0:
                            image_tile[c] = (ch - ch.mean()) / (ch.std() + 1e-8)
                    
                    # Convert to tensor
                    image_tensor = torch.from_numpy(image_tile).float().unsqueeze(0)
                    image_tensor = image_tensor.to(device)
                    
                    # Predict
                    output = model(image_tensor)
                    pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                    
                    # Store prediction
                    prediction[row_start:row_start+tile_size, 
                              col_start:col_start+tile_size] = pred
        
        # Save prediction
        output_path = OUTPUTS_DIR / 'maps' / 'land_cover_prediction.tif'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy metadata
        out_meta = landsat_src.meta.copy()
        out_meta.update({
            'count': 1,
            'dtype': 'uint8',
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(prediction, 1)
        
        print(f"\n✅ Prediction saved: {output_path}")
        
        # Generate visualization
        visualize_prediction(prediction, output_path.parent / 'land_cover_prediction.png')
        
        # Calculate statistics
        calculate_stats(prediction)
        
        return prediction

def visualize_prediction(prediction, output_path):
    """Create visualization of prediction"""
    
    print("\nCreating visualization...")
    
    # Define colors for each class
    colors = {
        0: [0.13, 0.55, 0.13],  # Vegetation - Forest Green
        1: [0.00, 0.44, 0.75],  # Water - Blue
        2: [0.50, 0.50, 0.50],  # Built-up - Gray
        3: [0.87, 0.72, 0.53]   # Bare/Open - Tan
    }
    
    # Create RGB image
    rgb = np.zeros((prediction.shape[0], prediction.shape[1], 3))
    for class_id, color in colors.items():
        mask = prediction == class_id
        rgb[mask] = color
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    ax.imshow(rgb)
    ax.set_title('Land Cover Prediction - Manhattan & Brooklyn', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Legend
    patches = [
        mpatches.Patch(color=colors[0], label='Vegetation'),
        mpatches.Patch(color=colors[1], label='Water'),
        mpatches.Patch(color=colors[2], label='Built-up'),
        mpatches.Patch(color=colors[3], label='Bare/Open')
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved: {output_path}")

def calculate_stats(prediction):
    """Calculate prediction statistics"""
    
    print("\n" + "="*60)
    print("LAND COVER STATISTICS")
    print("="*60)
    
    total_pixels = prediction.size
    
    for class_id, class_name in LANDCOVER_CLASSES.items():
        count = np.sum(prediction == class_id)
        percentage = (count / total_pixels) * 100
        print(f"{class_name:12s}: {percentage:6.2f}% ({count:,} pixels)")
    
    print("="*60)

if __name__ == "__main__":
    predict_full_image()