"""
Download Landsat 9 in separate tiles for Manhattan and Brooklyn
"""

import ee
import geemap
from pathlib import Path
import sys
from dotenv import load_dotenv
import os
import rasterio
from rasterio.merge import merge
import numpy as np

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import ANALYSIS_PERIOD, RAW_DATA_DIR

# Initialize GEE
project_id = os.getenv('GEE_PROJECT')
ee.Initialize(project=project_id)

def download_landsat_by_borough(borough_name, bbox):
    """
    Download Landsat for a single borough
    
    Args:
        borough_name: 'Manhattan' or 'Brooklyn'
        bbox: dict with 'west', 'south', 'east', 'north'
    """
    
    print(f"\n{'='*60}")
    print(f"DOWNLOADING {borough_name.upper()}")
    print(f"{'='*60}")
    
    roi = ee.Geometry.Rectangle([
        bbox['west'], bbox['south'],
        bbox['east'], bbox['north']
    ])
    
    # Load Landsat 9
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
        .filterBounds(roi) \
        .filterDate(ANALYSIS_PERIOD['start_date'], ANALYSIS_PERIOD['end_date']) \
        .filter(ee.Filter.lt('CLOUD_COVER', 30))
    
    count = l9.size().getInfo()
    print(f"Found {count} images")
    
    # Apply scaling
    def apply_scale_factors(image):
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        return optical_bands
    
    l9_scaled = l9.map(apply_scale_factors)
    composite = l9_scaled.median()
    
    # Select bands
    export_bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5']  # Blue, Green, Red, NIR
    export_image = composite.select(export_bands)
    
    # Calculate NDVI
    nir = composite.select('SR_B5')
    red = composite.select('SR_B4')
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    export_image = export_image.addBands(ndvi)
    
    # Output path
    output_path = RAW_DATA_DIR / 'landsat' / f'{borough_name.lower()}_landsat.tif'
    
    print(f"📥 Downloading {borough_name}...")
    
    try:
        geemap.ee_export_image(
            export_image,
            filename=str(output_path),
            scale=30,
            region=roi,
            file_per_band=False
        )
        
        # Check if file exists and has content
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ {borough_name} downloaded: {size_mb:.2f} MB")
            
            if size_mb > 1:
                return output_path
            else:
                print(f"⚠️ File size too small, may have failed")
                return None
        else:
            print(f"❌ File not created")
            return None
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def main():
    """Download both boroughs separately"""
    
    print("="*60)
    print("DOWNLOADING LANDSAT 9 - SPLIT BY BOROUGH")
    print("="*60)
    
    # Define bounding boxes
    manhattan_bbox = {
        'west': -74.02,
        'south': 40.70,
        'east': -73.91,
        'north': 40.88
    }
    
    brooklyn_bbox = {
        'west': -74.05,
        'south': 40.57,
        'east': -73.85,
        'north': 40.74
    }
    
    # Download both
    manhattan_file = download_landsat_by_borough('Manhattan', manhattan_bbox)
    brooklyn_file = download_landsat_by_borough('Brooklyn', brooklyn_bbox)
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Manhattan: {'✅ Success' if manhattan_file else '❌ Failed'}")
    print(f"Brooklyn: {'✅ Success' if brooklyn_file else '❌ Failed'}")
    
    if manhattan_file and brooklyn_file:
        print("\n✅ Both boroughs downloaded successfully!")
        print("\nFiles created:")
        print(f"  - {manhattan_file}")
        print(f"  - {brooklyn_file}")
        print("\n🎉 Ready for preprocessing!")
    elif manhattan_file or brooklyn_file:
        print("\n⚠️ Partial success - we can work with what we have")
    else:
        print("\n❌ Both downloads failed")
        print("\nAlternative: We can proceed with just the LST data we already have")
        print("and use WorldCover for vegetation analysis.")
    
    print("="*60)

if __name__ == "__main__":
    main()