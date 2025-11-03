"""
Download WorldCover for expanded study area
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

from config.settings import STUDY_AREA, RAW_DATA_DIR, WORLDCOVER_PARAMS

# Initialize GEE
project_id = os.getenv('GEE_PROJECT')
ee.Initialize(project=project_id)

def download_worldcover():
    """Download ESA WorldCover for expanded area"""
    
    print("="*60)
    print("DOWNLOADING ESA WORLDCOVER (EXPANDED AREA)")
    print("="*60)
    
    # Get expanded study area
    bbox = STUDY_AREA['bbox']
    roi = ee.Geometry.Rectangle([
        bbox['west'], bbox['south'],
        bbox['east'], bbox['north']
    ])
    
    print(f"Bounding box: {bbox}")
    
    # Load WorldCover
    worldcover = ee.ImageCollection(WORLDCOVER_PARAMS['collection']).first()
    
    # Export
    output_path = RAW_DATA_DIR / 'landcover' / 'manhattan_brooklyn_worldcover.tif'
    
    print(f"📥 Downloading WorldCover...")
    print(f"   This may take 3-5 minutes...")
    
    try:
        geemap.ee_export_image(
            worldcover,
            filename=str(output_path),
            scale=10,
            region=roi,
            file_per_band=False
        )
        
        print(f"✅ WorldCover downloaded")
        print(f"   Saved to: {output_path}")
        
        # Check file size
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"   File size: {size_mb:.2f} MB")
            return True
        return False
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    download_worldcover()