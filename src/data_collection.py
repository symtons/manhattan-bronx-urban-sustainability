"""
Data Collection Script for NYC Urban Sustainability Project
Downloads satellite imagery and vector data for Manhattan & Brooklyn
"""

import ee
import geemap
import geopandas as gpd
import requests
from pathlib import Path
import sys
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import (
    STUDY_AREA, ANALYSIS_PERIOD, SENTINEL2_PARAMS, LANDSAT9_PARAMS,
    WORLDCOVER_PARAMS, NYC_DATA_URLS, RAW_DATA_DIR
)

class NYCDataCollector:
    """
    Handles data collection from Google Earth Engine and NYC Open Data
    """
    
    def __init__(self):
        """Initialize the data collector"""
        self.initialize_gee()
        self.setup_directories()
        
    def initialize_gee(self):
        """Initialize Google Earth Engine"""
        try:
            project_id = os.getenv('GEE_PROJECT')
            if project_id:
                ee.Initialize(project=project_id)
                print(f"✅ GEE initialized with project: {project_id}")
            else:
                ee.Initialize()
                print("✅ GEE initialized with default credentials")
        except Exception as e:
            print(f"❌ Failed to initialize GEE: {e}")
            print("   Run 'python src/gee_auth.py' first to authenticate")
            sys.exit(1)
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            RAW_DATA_DIR / 'boundaries',
            RAW_DATA_DIR / 'sentinel',
            RAW_DATA_DIR / 'landsat',
            RAW_DATA_DIR / 'landcover'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ Directories created")
    
    def get_study_area_geometry(self):
        """
        Create Earth Engine geometry for study area
        
        Returns:
            ee.Geometry: Bounding box for Manhattan & Brooklyn
        """
        bbox = STUDY_AREA['bbox']
        geometry = ee.Geometry.Rectangle([
            bbox['west'], bbox['south'],
            bbox['east'], bbox['north']
        ])
        
        print(f"📍 Study area: {STUDY_AREA['name']}")
        print(f"   Bounds: {bbox}")
        
        return geometry
    
    def download_nyc_boundaries(self):
        """
        Download NYC borough boundaries from NYC Open Data
        """
        print("\n" + "="*60)
        print("DOWNLOADING NYC BOROUGH BOUNDARIES")
        print("="*60)
        
        try:
            # Download GeoJSON
            url = NYC_DATA_URLS['borough_boundaries']
            print(f"Fetching from: {url}")
            
            gdf = gpd.read_file(url)
            
            # Filter for Manhattan and Brooklyn only
            gdf = gdf[gdf['boro_name'].isin(['Manhattan', 'Brooklyn'])]
            
            # Save to file
            output_path = RAW_DATA_DIR / 'boundaries' / 'manhattan_brooklyn_boundaries.geojson'
            gdf.to_file(output_path, driver='GeoJSON')
            
            print(f"✅ Downloaded boundaries for {len(gdf)} boroughs")
            print(f"   Saved to: {output_path}")
            print(f"   Boroughs: {', '.join(gdf['boro_name'].tolist())}")
            
            return gdf
            
        except Exception as e:
            print(f"❌ Failed to download boundaries: {e}")
            return None
    
    def download_sentinel2_ndvi(self):
        """
        Download Sentinel-2 imagery and calculate NDVI composite
        """
        print("\n" + "="*60)
        print("DOWNLOADING SENTINEL-2 IMAGERY (NDVI)")
        print("="*60)
        
        try:
            # Get study area
            roi = self.get_study_area_geometry()
            
            # Get date range
            start_date = ANALYSIS_PERIOD['start_date']
            end_date = ANALYSIS_PERIOD['end_date']
            print(f"Date range: {start_date} to {end_date}")
            
            # Load Sentinel-2 collection
            s2 = ee.ImageCollection(SENTINEL2_PARAMS['collection']) \
                .filterBounds(roi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 
                                     SENTINEL2_PARAMS['cloud_threshold']))
            
            # Count available images
            count = s2.size().getInfo()
            print(f"Found {count} cloud-free images")
            
            if count == 0:
                print("⚠️  No images found. Trying with higher cloud threshold...")
                s2 = ee.ImageCollection(SENTINEL2_PARAMS['collection']) \
                    .filterBounds(roi) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                count = s2.size().getInfo()
                print(f"Found {count} images with 50% cloud threshold")
            
            # Select bands and calculate NDVI
            def add_ndvi(image):
                nir = image.select('B8')
                red = image.select('B4')
                ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
                return image.addBands(ndvi)
            
            s2_ndvi = s2.map(add_ndvi)
            
            # Create median composite
            composite = s2_ndvi.median()
            
            # Select bands for export (RGB + NIR + NDVI)
            export_bands = ['B4', 'B3', 'B2', 'B8', 'NDVI']
            export_image = composite.select(export_bands)
            
            # Export to Drive (will need to manually download)
            output_path = RAW_DATA_DIR / 'sentinel' / 'manhattan_brooklyn_sentinel2.tif'
            
            print(f"📥 Downloading Sentinel-2 composite...")
            print(f"   This may take 5-10 minutes...")
            
            # Download using geemap
            geemap.ee_export_image(
                export_image,
                filename=str(output_path),
                scale=10,
                region=roi,
                file_per_band=False
            )
            
            print(f"✅ Sentinel-2 data downloaded")
            print(f"   Saved to: {output_path}")
            print(f"   Bands: {', '.join(export_bands)}")
            
            return export_image
            
        except Exception as e:
            print(f"❌ Failed to download Sentinel-2: {e}")
            print(f"   Error details: {type(e).__name__}")
            return None
    
    def download_landsat9_lst(self):
        """
        Download Landsat 9 Land Surface Temperature data
        """
        print("\n" + "="*60)
        print("DOWNLOADING LANDSAT 9 (Land Surface Temperature)")
        print("="*60)
        
        try:
            # Get study area
            roi = self.get_study_area_geometry()
            
            # Get date range
            start_date = ANALYSIS_PERIOD['start_date']
            end_date = ANALYSIS_PERIOD['end_date']
            print(f"Date range: {start_date} to {end_date}")
            
            # Load Landsat 9 collection
            l9 = ee.ImageCollection(LANDSAT9_PARAMS['collection']) \
                .filterBounds(roi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUD_COVER', LANDSAT9_PARAMS['cloud_threshold']))
            
            # Count available images
            count = l9.size().getInfo()
            print(f"Found {count} cloud-free images")
            
            if count == 0:
                print("⚠️  No images found. Trying with higher cloud threshold...")
                l9 = ee.ImageCollection(LANDSAT9_PARAMS['collection']) \
                    .filterBounds(roi) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.lt('CLOUD_COVER', 50))
                count = l9.size().getInfo()
                print(f"Found {count} images with 50% cloud threshold")
            
            # Apply scale and offset to convert to Celsius
            def apply_scale_factors(image):
                thermal = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
                return thermal.rename('LST')
            
            l9_lst = l9.map(apply_scale_factors)
            
            # Create median composite
            composite = l9_lst.median()
            
            # Export
            output_path = RAW_DATA_DIR / 'landsat' / 'manhattan_brooklyn_lst.tif'
            
            print(f"📥 Downloading Landsat 9 LST...")
            print(f"   This may take 5-10 minutes...")
            
            # Download using geemap
            geemap.ee_export_image(
                composite,
                filename=str(output_path),
                scale=30,
                region=roi,
                file_per_band=False
            )
            
            print(f"✅ Landsat 9 LST downloaded")
            print(f"   Saved to: {output_path}")
            
            return composite
            
        except Exception as e:
            print(f"❌ Failed to download Landsat 9: {e}")
            print(f"   Error details: {type(e).__name__}")
            return None
    
    def download_worldcover_labels(self):
        """
        Download ESA WorldCover land cover classification
        (Used as training labels for our model)
        """
        print("\n" + "="*60)
        print("DOWNLOADING ESA WORLDCOVER (Training Labels)")
        print("="*60)
        
        try:
            # Get study area
            roi = self.get_study_area_geometry()
            
            # Load WorldCover
            worldcover = ee.ImageCollection(WORLDCOVER_PARAMS['collection']).first()
            
            # Export
            output_path = RAW_DATA_DIR / 'landcover' / 'manhattan_brooklyn_worldcover.tif'
            
            print(f"📥 Downloading ESA WorldCover...")
            print(f"   This may take 3-5 minutes...")
            
            # Download using geemap
            geemap.ee_export_image(
                worldcover,
                filename=str(output_path),
                scale=10,
                region=roi,
                file_per_band=False
            )
            
            print(f"✅ WorldCover downloaded")
            print(f"   Saved to: {output_path}")
            print(f"   Classes: {len(WORLDCOVER_PARAMS['classes'])} land cover types")
            
            return worldcover
            
        except Exception as e:
            print(f"❌ Failed to download WorldCover: {e}")
            print(f"   Error details: {type(e).__name__}")
            return None
    
    def run_full_download(self):
        """
        Execute complete data download workflow
        """
        print("\n" + "="*60)
        print("NYC URBAN SUSTAINABILITY DATA COLLECTION")
        print("Manhattan & Brooklyn")
        print("="*60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Download boundaries
        boundaries = self.download_nyc_boundaries()
        
        # Step 2: Download Sentinel-2
        sentinel2 = self.download_sentinel2_ndvi()
        
        # Step 3: Download Landsat 9
        landsat9 = self.download_landsat9_lst()
        
        # Step 4: Download WorldCover
        worldcover = self.download_worldcover_labels()
        
        # Summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"✅ Boundaries: {'Success' if boundaries is not None else 'Failed'}")
        print(f"✅ Sentinel-2: {'Success' if sentinel2 is not None else 'Failed'}")
        print(f"✅ Landsat 9: {'Success' if landsat9 is not None else 'Failed'}")
        print(f"✅ WorldCover: {'Success' if worldcover is not None else 'Failed'}")
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

def main():
    """
    Main function to run data collection
    """
    collector = NYCDataCollector()
    collector.run_full_download()

if __name__ == "__main__":
    main()