"""
Merge Manhattan and Brooklyn LST files
"""

import rasterio
from rasterio.merge import merge
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

def merge_lst_files():
    """Merge LST files"""
    
    print("="*60)
    print("MERGING LST FILES")
    print("="*60)
    
    manhattan_lst = RAW_DATA_DIR / 'landsat' / 'manhattan_lst.tif'
    brooklyn_lst = RAW_DATA_DIR / 'landsat' / 'brooklyn_lst.tif'
    output_path = PROCESSED_DATA_DIR / 'lst.tif'
    
    # Check files exist
    if not manhattan_lst.exists():
        print(f"❌ Manhattan LST not found: {manhattan_lst}")
        return False
    
    if not brooklyn_lst.exists():
        print(f"❌ Brooklyn LST not found: {brooklyn_lst}")
        return False
    
    print(f"Manhattan LST: {manhattan_lst.exists()}")
    print(f"Brooklyn LST: {brooklyn_lst.exists()}")
    
    # Merge
    src_man = rasterio.open(manhattan_lst)
    src_bk = rasterio.open(brooklyn_lst)
    
    print(f"\nMerging...")
    mosaic, out_trans = merge([src_man, src_bk])
    
    out_meta = src_man.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw"
    })
    
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    src_man.close()
    src_bk.close()
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ LST merged: {size_mb:.2f} MB")
    print(f"   Output: {output_path}")
    
    return True

if __name__ == "__main__":
    merge_lst_files()