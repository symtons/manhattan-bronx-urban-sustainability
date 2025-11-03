"""
Complete redownload with expanded bounding box
Downloads: Landsat multispectral, LST, and WorldCover
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

from config.settings import STUDY_AREA, ANALYSIS_PERIOD, RAW_DATA_DIR

# Initialize GEE
project_id = os.getenv('GEE_PROJECT')
ee.Initialize(project=project_id)

def download_by_borough(borough_name, bbox):
    """Download all data for one borough"""
    
    print(f"\n{'='*60}")
    print(f"DOWNLOADING {borough_name.upper()}")
    print(f"{'='*60}")
    print(f"Bounds: {bbox}")
    
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
    print(f"Found {count} Landsat images")
    
    # Process optical bands
    def apply_scale_factors(image):
        optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
        return optical.addBands(thermal)
    
    l9_processed = l9.map(apply_scale_factors)
    composite = l9_processed.median()
    
    # Export multispectral (RGB + NIR + NDVI)
    nir = composite.select('SR_B5')
    red = composite.select('SR_B4')
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    
    multispectral = composite.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5']).addBands(ndvi)
    
    output_ms = RAW_DATA_DIR / 'landsat' / f'{borough_name.lower()}_landsat.tif'
    
    print(f"\n📥 Downloading multispectral data...")
    try:
        geemap.ee_export_image(
            multispectral,
            filename=str(output_ms),
            scale=30,
            region=roi,
            file_per_band=False
        )
        
        if output_ms.exists():
            size = output_ms.stat().st_size / (1024 * 1024)
            print(f"✅ Multispectral: {size:.2f} MB")
    except Exception as e:
        print(f"❌ Multispectral failed: {e}")
    
    # Export LST separately
    lst = composite.select('LST')
    output_lst = RAW_DATA_DIR / 'landsat' / f'{borough_name.lower()}_lst.tif'
    
    print(f"📥 Downloading LST...")
    try:
        geemap.ee_export_image(
            lst,
            filename=str(output_lst),
            scale=30,
            region=roi,
            file_per_band=False
        )
        
        if output_lst.exists():
            size = output_lst.stat().st_size / (1024 * 1024)
            print(f"✅ LST: {size:.2f} MB")
    except Exception as e:
        print(f"❌ LST failed: {e}")

def download_worldcover():
    """Download WorldCover"""
    
    print(f"\n{'='*60}")
    print("DOWNLOADING ESA WORLDCOVER")
    print(f"{'='*60}")
    
    bbox = STUDY_AREA['bbox']
    roi = ee.Geometry.Rectangle([
        bbox['west'], bbox['south'],
        bbox['east'], bbox['north']
    ])
    
    worldcover = ee.ImageCollection('ESA/WorldCover/v200').first()
    output_path = RAW_DATA_DIR / 'landcover' / 'manhattan_brooklyn_worldcover.tif'
    
    print("📥 Downloading...")
    try:
        geemap.ee_export_image(
            worldcover,
            filename=str(output_path),
            scale=10,
            region=roi,
            file_per_band=False
        )
        
        if output_path.exists():
            size = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ WorldCover: {size:.2f} MB")
    except Exception as e:
        print(f"❌ WorldCover failed: {e}")

def main():
    """Download everything"""
    
    print("="*60)
    print("COMPLETE DATA REDOWNLOAD - EXPANDED AREA")
    print("="*60)
    
    # Manhattan bbox
    manhattan_bbox = {
        'west': -74.05,
        'south': 40.68,
        'east': -73.90,
        'north': 40.93
    }
    
    # Brooklyn bbox
    brooklyn_bbox = {
        'west': -74.10,
        'south': 40.55,
        'east': -73.70,
        'north': 40.75
    }
    
    # Download both boroughs
    download_by_borough('Manhattan', manhattan_bbox)
    download_by_borough('Brooklyn', brooklyn_bbox)
    
    # Download WorldCover
    download_worldcover()
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python src/preprocessing.py")
    print("2. Run: python src/fix_alignment.py")
    print("3. Run: python src/create_tiles.py")

if __name__ == "__main__":
    main()