"""
Final alignment - match WorldCover exactly to NDVI dimensions
"""

import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR

def align_to_reference():
    """Align all rasters to match NDVI (our reference)"""
    
    print("="*60)
    print("FINAL ALIGNMENT - Match Everything to NDVI")
    print("="*60)
    
    # Reference (NDVI)
    reference_path = PROCESSED_DATA_DIR / 'ndvi_utm.tif'
    
    with rasterio.open(reference_path) as ref:
        ref_profile = ref.profile
        ref_data = ref.read(1)
        
        print(f"\nReference (NDVI):")
        print(f"  Dimensions: {ref.width} × {ref.height}")
        print(f"  CRS: {ref.crs}")
        print(f"  Bounds: {ref.bounds}")
    
    # Files to align
    files_to_align = [
        ('worldcover_utm.tif', 'worldcover_final.tif', Resampling.nearest),
        ('lst_utm.tif', 'lst_final.tif', Resampling.bilinear)
    ]
    
    for input_name, output_name, resampling_method in files_to_align:
        input_path = PROCESSED_DATA_DIR / input_name
        output_path = PROCESSED_DATA_DIR / output_name
        
        if not input_path.exists():
            print(f"\n⚠️  Skipping {input_name} (not found)")
            continue
        
        print(f"\nAligning: {input_name}")
        
        with rasterio.open(input_path) as src:
            print(f"  Input: {src.width} × {src.height}")
            
            # Create output array matching reference
            output_data = np.zeros((ref.height, ref.width), dtype=src.dtypes[0])
            
            # Reproject to match reference exactly
            reproject(
                source=rasterio.band(src, 1),
                destination=output_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_profile['transform'],
                dst_crs=ref_profile['crs'],
                resampling=resampling_method
            )
            
            # Save with reference profile (but fix nodata)
            output_profile = ref_profile.copy()
            output_profile.update({
                'dtype': src.dtypes[0],
                'count': 1,
                'nodata': src.nodata if src.nodata is not None else None
            })
            
            with rasterio.open(output_path, 'w', **output_profile) as dst:
                dst.write(output_data, 1)
            
            print(f"  Output: {ref.width} × {ref.height}")
            print(f"  ✅ Saved: {output_name}")
    
    # Also copy NDVI as final
    import shutil
    shutil.copy(reference_path, PROCESSED_DATA_DIR / 'ndvi_final.tif')
    print(f"\n✅ Copied NDVI as ndvi_final.tif")
    
    print("\n" + "="*60)
    print("ALIGNMENT COMPLETE - All rasters now match!")
    print("="*60)
    
    # Verify
    print("\nVerification:")
    final_files = ['ndvi_final.tif', 'worldcover_final.tif', 'lst_final.tif']
    
    for fname in final_files:
        fpath = PROCESSED_DATA_DIR / fname
        if fpath.exists():
            with rasterio.open(fpath) as src:
                print(f"  {fname:20s}: {src.width:4d} × {src.height:4d} | {src.crs}")
        else:
            print(f"  {fname:20s}: NOT FOUND")

if __name__ == "__main__":
    align_to_reference()