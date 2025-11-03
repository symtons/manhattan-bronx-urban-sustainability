"""
Fix raster alignment issues - ensure all rasters have same dimensions
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR, STUDY_AREA

def align_to_reference():
    """
    Align all rasters to match NDVI dimensions and extent
    """
    
    print("="*60)
    print("FIXING RASTER ALIGNMENT")
    print("="*60)
    
    # Use merged Landsat as reference (it has proper alignment)
    reference_path = PROCESSED_DATA_DIR / 'landsat_merged.tif'
    
    files_to_align = [
        (PROCESSED_DATA_DIR / 'ndvi.tif', PROCESSED_DATA_DIR / 'ndvi_fixed.tif'),
        (PROCESSED_DATA_DIR / 'lst_clipped.tif', PROCESSED_DATA_DIR / 'lst_fixed.tif'),
        (PROCESSED_DATA_DIR / 'worldcover_clipped.tif', PROCESSED_DATA_DIR / 'worldcover_fixed.tif')
    ]
    
    with rasterio.open(reference_path) as ref:
        ref_profile = ref.profile.copy()
        ref_bounds = ref.bounds
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)
        
        print(f"\nReference raster (Landsat):")
        print(f"  Dimensions: {ref.width} × {ref.height}")
        print(f"  CRS: {ref_crs}")
        print(f"  Bounds: {ref_bounds}")
        
        for input_path, output_path in files_to_align:
            if not input_path.exists():
                print(f"\n⚠️  Skipping {input_path.name} (not found)")
                continue
            
            print(f"\nAligning {input_path.name}...")
            
            with rasterio.open(input_path) as src:
                print(f"  Original: {src.width} × {src.height}")
                
                # Update profile to match reference
                out_profile = src.profile.copy()
                out_profile.update({
                    'crs': ref_crs,
                    'transform': ref_transform,
                    'width': ref.width,
                    'height': ref.height,
                    'compress': 'lzw'
                })
                
                # Create output array
                out_data = np.zeros((src.count, ref.height, ref.width), dtype=src.dtypes[0])
                
                # Reproject each band
                for band in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band),
                        destination=out_data[band-1],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=ResamplingEnum.bilinear
                    )
                
                # Write aligned raster
                with rasterio.open(output_path, 'w', **out_profile) as dst:
                    dst.write(out_data)
                
                print(f"  ✅ Aligned to: {ref.width} × {ref.height}")
                print(f"  Saved: {output_path.name}")
    
    print("\n" + "="*60)
    print("✅ ALIGNMENT COMPLETE")
    print("="*60)
    print("\nAll rasters now have matching dimensions!")
    print("Ready to generate tiles.")

if __name__ == "__main__":
    align_to_reference()