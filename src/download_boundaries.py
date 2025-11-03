"""
Quick script to download NYC boundaries with SSL fix
"""

import geopandas as gpd
import urllib3
import warnings
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import RAW_DATA_DIR

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

def download_boundaries():
    """Download NYC borough boundaries"""
    
    print("Downloading NYC Borough Boundaries...")
    
    # Alternative: Use direct download URL
    url = "https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=GeoJSON"
    
    try:
        # Try with SSL verification disabled
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Download
        gdf = gpd.read_file(url)
        
        # Filter for Manhattan and Brooklyn
        gdf = gdf[gdf['boro_name'].isin(['Manhattan', 'Brooklyn'])]
        
        # Ensure directory exists
        output_dir = RAW_DATA_DIR / 'boundaries'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save
        output_path = output_dir / 'manhattan_brooklyn_boundaries.geojson'
        gdf.to_file(output_path, driver='GeoJSON')
        
        print(f"✅ Downloaded boundaries for {len(gdf)} boroughs")
        print(f"   Saved to: {output_path}")
        print(f"   Boroughs: {', '.join(gdf['boro_name'].tolist())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("\nAlternative: Download manually from:")
        print("https://data.cityofnewyork.us/City-Government/Borough-Boundaries/tqmj-j8zm")
        print("Export as GeoJSON and save to data/raw/boundaries/")
        return False

if __name__ == "__main__":
    download_boundaries()