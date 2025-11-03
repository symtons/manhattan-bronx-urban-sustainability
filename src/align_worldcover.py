"""
Align WorldCover to match Landsat/NDVI dimensions and projection
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR

def align_worldcover():
    """Align WorldCover to match NDVI"""
    
    print("="*60)
    print("ALIGNING WORLDCOVER TO MATCH NDVI/LST")
    print("="*60)
    
    # Reference raster (NDVI - already correct)
    reference_path = PROCESSED_DATA_DIR / 'ndvi_fixed.tif'
    worldcover_path = PROCESSED_DATA_DIR / 'worldcover_clipped.tif'
    output_path = PROCESSED_DATA_DIR / 'worldcover_fixed.tif'
    
    print(f"\nReference: {reference_path}")
    print(f"Input: {worldcover_path}")
    print(f"Output: {output_path}")
    
    # Open reference to get target specs
    with rasterio.open(reference_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_bounds = ref.bounds
        
        print(f"\nReference specs:")
        print(f"  CRS: {ref_crs}")
        print(f"  Dimensions: {ref_width} × {ref_height}")
        print(f"  Pixel size: {ref_transform[0]:.2f}m")
        print(f"  Bounds: {ref_bounds}")
    
    # Reproject WorldCover to match
    with rasterio.open(worldcover_path) as src:
        print(f"\nWorldCover input:")
        print(f"  CRS: {src.crs}")
        print(f"  Dimensions: {src.width} × {src.height}")
        
        # Read data
        data = src.read(1)
        
        # Create output
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': ref_crs,
            'transform': ref_transform,
            'width': ref_width,
            'height': ref_height,
            'compress': 'lzw'
        })
        
        print(f"\nReprojecting...")
        
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            reproject(
                source=data,
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest  # Use nearest for categorical data
            )
    
    print(f"\n✅ WorldCover aligned and saved")
    
    # Verify
    with rasterio.open(output_path) as dst:
        print(f"\nOutput verification:")
        print(f"  CRS: {dst.crs}")
        print(f"  Dimensions: {dst.width} × {dst.height}")
        print(f"  Pixel size: {dst.transform[0]:.2f}m")
        print(f"  Bounds: {dst.bounds}")
        
        # Check if matches reference
        if (dst.width == ref_width and dst.height == ref_height and 
            dst.crs == ref_crs):
            print(f"\n✅ Perfect alignment with reference!")
        else:
            print(f"\n⚠️  Dimensions don't match perfectly")

if __name__ == "__main__":
    align_worldcover()