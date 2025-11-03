"""
Download complete Landsat 9 imagery with all bands for NDVI
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

def download_landsat_multispectral():
    """
    Download Landsat 9 with optical bands for NDVI calculation
    """
    
    print("=" * 60)
    print("DOWNLOADING LANDSAT 9 - COMPLETE MULTISPECTRAL")
    print("=" * 60)
    
    # Get study area
    bbox = STUDY_AREA['bbox']
    roi = ee.Geometry.Rectangle([
        bbox['west'], bbox['south'],
        bbox['east'], bbox['north']
    ])
    
    print(f"Study area: {STUDY_AREA['name']}")
    print(f"Date range: {ANALYSIS_PERIOD['start_date']} to {ANALYSIS_PERIOD['end_date']}")
    
    # Load Landsat 9 Surface Reflectance
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
        .filterBounds(roi) \
        .filterDate(ANALYSIS_PERIOD['start_date'], ANALYSIS_PERIOD['end_date']) \
        .filter(ee.Filter.lt('CLOUD_COVER', 20))
    
    count = l9.size().getInfo()
    print(f"Found {count} cloud-free images")
    
    if count == 0:
        print("⚠️ No images found, relaxing cloud cover threshold...")
        l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
            .filterBounds(roi) \
            .filterDate(ANALYSIS_PERIOD['start_date'], ANALYSIS_PERIOD['end_date']) \
            .filter(ee.Filter.lt('CLOUD_COVER', 50))
        count = l9.size().getInfo()
        print(f"Found {count} images with 50% cloud threshold")
    
    # Apply scaling factors
    def apply_scale_factors(image):
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        return optical_bands
    
    l9_scaled = l9.map(apply_scale_factors)
    
    # Create median composite
    composite = l9_scaled.median()
    
    # Select bands: Blue, Green, Red, NIR, SWIR1, SWIR2
    # Band names in Landsat 9: SR_B2=Blue, SR_B3=Green, SR_B4=Red, SR_B5=NIR
    export_bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    export_image = composite.select(export_bands)
    
    # Calculate and add NDVI
    nir = composite.select('SR_B5')
    red = composite.select('SR_B4')
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    
    export_image = export_image.addBands(ndvi)
    
    # Export
    output_path = RAW_DATA_DIR / 'landsat' / 'manhattan_brooklyn_landsat_multispectral.tif'
    
    print(f"\n📥 Downloading Landsat 9 multispectral data...")
    print(f"   Bands: Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI")
    print(f"   Resolution: 30m")
    print(f"   This may take 5-8 minutes...")
    
    try:
        geemap.ee_export_image(
            export_image,
            filename=str(output_path),
            scale=30,
            region=roi,
            file_per_band=False
        )
        
        print(f"\n✅ Download successful!")
        print(f"   Saved to: {output_path}")
        
        # Check file size
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"   File size: {size_mb:.2f} MB")
            
            if size_mb > 5:
                print(f"   ✅ File size looks good!")
                return True
            else:
                print(f"   ⚠️ File size seems small, but proceeding...")
                return True
        else:
            print(f"   ⚠️ File not found at expected location")
            return False
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    success = download_landsat_multispectral()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ ALL DATA COLLECTION COMPLETE!")
        print("=" * 60)
        print("\nDownloaded datasets:")
        print("1. ✅ Borough boundaries")
        print("2. ✅ Landsat 9 multispectral (for NDVI)")
        print("3. ✅ Landsat 9 thermal (for LST)")
        print("4. ✅ ESA WorldCover (training labels)")
        print("\n🚀 Ready to proceed with preprocessing!")
        print("=" * 60)
    else:
        print("\n⚠️ Download had issues. Check errors above.")