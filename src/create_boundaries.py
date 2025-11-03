"""
Create Manhattan & Brooklyn boundaries from coordinates
"""

import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import RAW_DATA_DIR, STUDY_AREA

def create_simple_boundaries():
    """Create simple rectangular boundaries for study area"""
    
    bbox = STUDY_AREA['bbox']
    
    # Create Manhattan boundary (simplified)
    manhattan_coords = [
        (-74.02, 40.70),  # SW corner
        (-73.91, 40.70),  # SE corner
        (-73.91, 40.88),  # NE corner
        (-74.02, 40.88),  # NW corner
        (-74.02, 40.70)   # Close polygon
    ]
    
    # Create Brooklyn boundary (simplified)
    brooklyn_coords = [
        (-74.05, 40.57),  # SW corner
        (-73.85, 40.57),  # SE corner
        (-73.85, 40.74),  # NE corner
        (-74.05, 40.74),  # NW corner
        (-74.05, 40.57)   # Close polygon
    ]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'boro_name': ['Manhattan', 'Brooklyn'],
        'geometry': [
            Polygon(manhattan_coords),
            Polygon(brooklyn_coords)
        ]
    }, crs='EPSG:4326')
    
    # Save
    output_dir = RAW_DATA_DIR / 'boundaries'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'manhattan_brooklyn_boundaries.geojson'
    
    gdf.to_file(output_path, driver='GeoJSON')
    
    print("✅ Created simplified boundaries")
    print(f"   Saved to: {output_path}")
    print(f"   Boroughs: Manhattan, Brooklyn")
    
    return True

if __name__ == "__main__":
    create_simple_boundaries()