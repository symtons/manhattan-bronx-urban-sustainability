"""
Force all rasters to UTM Zone 18N (EPSG:32618) with 30m resolution
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR

def reproject_to_utm(input_path, output_path, resolution=30):
    """
    Reproject any raster to UTM Zone 18N with specified resolution
    """
    
    target_crs = 'EPSG:32618'  # UTM Zone 18N
    
    print(f"\nProcessing: {input_path.name}")
    
    with rasterio.open(input_path) as src:
        print(f"  Input CRS: {src.crs}")
        print(f"  Input size: {src.width} × {src.height}")
        
        # Calculate transform for target CRS
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=resolution
        )
        
        print(f"  Output CRS: {target_crs}")
        print(f"  Output size: {width} × {height}")
        print(f"  Pixel size: {resolution}m")
        
        # Create output
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Reproject
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear if 'landsat' in str(input_path).lower() or 'ndvi' in str(input_path).lower() or 'lst' in str(input_path).lower() else Resampling.nearest
                )
        
        print(f"  ✅ Saved: {output_path.name}")

def main():
    """Reproject all key rasters to UTM"""
    
    print("="*60)
    print("FORCE UTM PROJECTION - 30m Resolution")
    print("="*60)
    
    # Files to reproject
    files_to_process = [
        ('landsat_merged.tif', 'landsat_utm.tif'),
        ('ndvi.tif', 'ndvi_utm.tif'),
        ('lst.tif', 'lst_utm.tif'),
        ('worldcover_clipped.tif', 'worldcover_utm.tif')
    ]
    
    for input_name, output_name in files_to_process:
        input_path = PROCESSED_DATA_DIR / input_name
        output_path = PROCESSED_DATA_DIR / output_name
        
        if input_path.exists():
            try:
                reproject_to_utm(input_path, output_path, resolution=30)
            except Exception as e:
                print(f"  ❌ Error: {e}")
        else:
            print(f"\n⚠️  Not found: {input_name}")
    
    print("\n" + "="*60)
    print("REPROJECTION COMPLETE")
    print("="*60)
    
    # Verify all have same dimensions
    print("\nVerification:")
    utm_files = [
        'landsat_utm.tif',
        'ndvi_utm.tif', 
        'lst_utm.tif',
        'worldcover_utm.tif'
    ]
    
    for fname in utm_files:
        fpath = PROCESSED_DATA_DIR / fname
        if fpath.exists():
            with rasterio.open(fpath) as src:
                print(f"  {fname:20s}: {src.width:4d} × {src.height:4d} | {src.crs} | {src.transform[0]:.1f}m")
        else:
            print(f"  {fname:20s}: NOT FOUND")

if __name__ == "__main__":
    main()