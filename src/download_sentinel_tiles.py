"""
Download Sentinel-2 in smaller tiles to avoid size limits
"""

import ee
import geemap
from pathlib import Path
import sys
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import STUDY_AREA, ANALYSIS_PERIOD, SENTINEL2_PARAMS, RAW_DATA_DIR

# Initialize GEE
project_id = os.getenv('GEE_PROJECT')
ee.Initialize(project=project_id)

def download_sentinel2_smaller():
    """
    Download Sentinel-2 with smaller region to avoid size limit
    """
    
    print("Downloading Sentinel-2 (optimized for size limits)...")
    
    # Use smaller bounding box
    bbox = STUDY_AREA['bbox']
    roi = ee.Geometry.Rectangle([
        bbox['west'], bbox['south'],
        bbox['east'], bbox['north']
    ])
    
    # Load and process Sentinel-2
    s2 = ee.ImageCollection(SENTINEL2_PARAMS['collection']) \
        .filterBounds(roi) \
        .filterDate(ANALYSIS_PERIOD['start_date'], ANALYSIS_PERIOD['end_date']) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    
    print(f"Found {s2.size().getInfo()} images")
    
    # Add NDVI
    def add_ndvi(image):
        nir = image.select('B8')
        red = image.select('B4')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        return image.addBands(ndvi)
    
    s2_ndvi = s2.map(add_ndvi)
    composite = s2_ndvi.median()
    
    # Export only essential bands to reduce size
    export_image = composite.select(['B4', 'B3', 'B2', 'B8', 'NDVI'])
    
    output_path = RAW_DATA_DIR / 'sentinel' / 'manhattan_brooklyn_sentinel2.tif'
    
    # Use higher scale (lower resolution) to reduce file size
    print("Downloading at 20m resolution to reduce file size...")
    
    try:
        geemap.ee_export_image(
            export_image,
            filename=str(output_path),
            scale=20,  # Use 20m instead of 10m
            region=roi,
            file_per_band=False
        )
        
        print(f"✅ Download successful!")
        print(f"   Saved to: {output_path}")
        
        # Check file size
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"   File size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        
        # Try even lower resolution
        print("\nTrying with 30m resolution...")
        
        try:
            geemap.ee_export_image(
                export_image,
                filename=str(output_path),
                scale=30,  # Even lower resolution
                region=roi,
                file_per_band=False
            )
            
            print(f"✅ Download successful at 30m resolution!")
            
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"   File size: {size_mb:.2f} MB")
            
            return True
            
        except Exception as e2:
            print(f"❌ Still failed: {e2}")
            print("\nWe'll use Landsat data (30m) for vegetation analysis instead.")
            return False

if __name__ == "__main__":
    download_sentinel2_smaller()