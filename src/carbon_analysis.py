"""
Carbon Sequestration Analysis using WorldCover + NDVI
"""

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR, OUTPUTS_DIR
from config.cost_constants import CARBON_SEQUESTRATION

class CarbonAnalyzer:
    """
    Analyze carbon sequestration potential
    """
    
    def __init__(self):
        """Initialize analyzer"""
        self.landcover_path = PROCESSED_DATA_DIR / 'worldcover_utm.tif'
        self.ndvi_path = PROCESSED_DATA_DIR / 'ndvi_utm.tif'
        self.lst_path = PROCESSED_DATA_DIR / 'lst_utm.tif'
        
        
        self.boundaries_path = RAW_DATA_DIR / 'boundaries' / 'manhattan_brooklyn_boundaries.geojson'
        
        print("✅ Carbon Analyzer initialized")
    
    def reclassify_worldcover_to_simple(self, worldcover_data):
        """
        Reclassify WorldCover to simplified classes
        
        WorldCover classes:
        10: Tree cover → Vegetation
        20: Shrubland → Vegetation
        30: Grassland → Vegetation
        40: Cropland → Vegetation
        50: Built-up → Built-up
        60: Bare/sparse → Bare
        80: Water → Water
        """
        
        output = np.zeros_like(worldcover_data, dtype=np.uint8)
        
        # Vegetation (0)
        vegetation_classes = [10, 20, 30, 40]
        for cls in vegetation_classes:
            output[worldcover_data == cls] = 0
        
        # Water (1)
        output[worldcover_data == 80] = 1
        
        # Built-up (2)
        output[worldcover_data == 50] = 2
        
        # Bare/Open (3)
        output[worldcover_data == 60] = 3
        
        return output
    
    def calculate_vegetation_carbon(self):
        """
        Calculate carbon sequestration from vegetation
        """
        
        print("\n" + "="*60)
        print("CARBON SEQUESTRATION ANALYSIS")
        print("="*60)
        
        # Load rasters
        print("\nLoading rasters...")
        with rasterio.open(self.landcover_path) as lc_src, \
             rasterio.open(self.ndvi_path) as ndvi_src:
            
            landcover = lc_src.read(1)
            ndvi = ndvi_src.read(1)
            
            # Get pixel size (in meters)
            pixel_width = lc_src.transform[0]
            pixel_height = abs(lc_src.transform[4])
            pixel_area_m2 = pixel_width * pixel_height
            pixel_area_ha = pixel_area_m2 / 10000  # Convert to hectares
            
            print(f"Pixel size: {pixel_width}m × {pixel_height}m")
            print(f"Pixel area: {pixel_area_ha:.6f} hectares")
        
        # Reclassify
        landcover_simple = self.reclassify_worldcover_to_simple(landcover)
        
        # Calculate land cover areas
        print("\n" + "="*60)
        print("LAND COVER DISTRIBUTION")
        print("="*60)
        
        total_pixels = landcover_simple.size
        
        stats = {}
        class_names = {0: 'Vegetation', 1: 'Water', 2: 'Built-up', 3: 'Bare/Open'}
        
        for class_id, class_name in class_names.items():
            mask = landcover_simple == class_id
            count = np.sum(mask)
            percentage = (count / total_pixels) * 100
            area_ha = count * pixel_area_ha
            
            stats[class_name] = {
                'pixels': count,
                'percentage': percentage,
                'area_ha': area_ha,
                'area_km2': area_ha / 100
            }
            
            print(f"{class_name:12s}: {percentage:6.2f}% | {area_ha:10,.1f} ha | {area_ha/100:7.2f} km²")
        
        # Calculate carbon sequestration for vegetation
        print("\n" + "="*60)
        print("CARBON SEQUESTRATION POTENTIAL")
        print("="*60)
        
        # Vegetation mask
        veg_mask = landcover_simple == 0
        
        # Calculate carbon based on NDVI density
        carbon_map = np.zeros_like(ndvi, dtype=np.float32)
        
        # High density vegetation (NDVI > 0.6) - Dense trees
        high_veg_mask = veg_mask & (ndvi > 0.6)
        carbon_rate_high = 20  # tons CO2/ha/year
        carbon_map[high_veg_mask] = carbon_rate_high * pixel_area_ha
        
        # Medium density (0.4 < NDVI <= 0.6) - Mixed vegetation
        med_veg_mask = veg_mask & (ndvi > 0.4) & (ndvi <= 0.6)
        carbon_rate_med = 10  # tons CO2/ha/year
        carbon_map[med_veg_mask] = carbon_rate_med * pixel_area_ha
        
        # Low density (0.2 < NDVI <= 0.4) - Sparse vegetation/grass
        low_veg_mask = veg_mask & (ndvi > 0.2) & (ndvi <= 0.4)
        carbon_rate_low = 5  # tons CO2/ha/year
        carbon_map[low_veg_mask] = carbon_rate_low * pixel_area_ha
        
        # Very sparse (NDVI <= 0.2) - Minimal vegetation
        sparse_veg_mask = veg_mask & (ndvi <= 0.2)
        carbon_rate_sparse = 2  # tons CO2/ha/year
        carbon_map[sparse_veg_mask] = carbon_rate_sparse * pixel_area_ha
        
        # Total carbon sequestration
        total_carbon = np.sum(carbon_map)
        
        print(f"\nVegetation Breakdown:")
        print(f"  High density (NDVI > 0.6):     {np.sum(high_veg_mask):8,} pixels | {np.sum(carbon_map[high_veg_mask]):10,.0f} tCO₂/year")
        print(f"  Medium density (0.4-0.6):      {np.sum(med_veg_mask):8,} pixels | {np.sum(carbon_map[med_veg_mask]):10,.0f} tCO₂/year")
        print(f"  Low density (0.2-0.4):         {np.sum(low_veg_mask):8,} pixels | {np.sum(carbon_map[low_veg_mask]):10,.0f} tCO₂/year")
        print(f"  Very sparse (< 0.2):           {np.sum(sparse_veg_mask):8,} pixels | {np.sum(carbon_map[sparse_veg_mask]):10,.0f} tCO₂/year")
        
        print(f"\n{'='*60}")
        print(f"TOTAL CARBON SEQUESTRATION: {total_carbon:,.0f} tons CO₂/year")
        print(f"{'='*60}")
        
        # Save carbon map
        output_dir = OUTPUTS_DIR / 'maps'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(self.landcover_path) as src:
            meta = src.meta.copy()
            meta.update({'dtype': 'float32', 'count': 1})
            
            with rasterio.open(output_dir / 'carbon_sequestration.tif', 'w', **meta) as dst:
                dst.write(carbon_map, 1)
        
        print(f"\n✅ Carbon map saved: {output_dir / 'carbon_sequestration.tif'}")
        
        return stats, total_carbon, carbon_map
    
    def analyze_by_borough(self):
        """
        Calculate statistics per borough
        """
        
        print("\n" + "="*60)
        print("BOROUGH-LEVEL ANALYSIS")
        print("="*60)
        
        # Load boundaries
        gdf = gpd.read_file(self.boundaries_path)
        
        # Load rasters
        with rasterio.open(self.landcover_path) as lc_src, \
             rasterio.open(self.ndvi_path) as ndvi_src, \
             rasterio.open(self.lst_path) as lst_src:
            
            landcover = lc_src.read(1)
            ndvi = ndvi_src.read(1)
            lst = lst_src.read(1)
            
            pixel_area_ha = (lc_src.transform[0] * abs(lc_src.transform[4])) / 10000
            
            landcover_simple = self.reclassify_worldcover_to_simple(landcover)
            
            borough_stats = []
            
            for idx, row in gdf.iterrows():
                borough_name = row['boro_name']
                geometry = row.geometry
                
                # Create mask for this borough
                from rasterio.mask import mask as rio_mask
                
                try:
                    masked_lc, _ = rio_mask(lc_src, [geometry], crop=False, all_touched=True)
                    borough_mask = masked_lc[0] != lc_src.nodata
                    
                    # Calculate statistics for this borough
                    borough_landcover = landcover_simple[borough_mask]
                    borough_ndvi = ndvi[borough_mask]
                    borough_lst = lst[borough_mask]
                    
                    # Land cover percentages
                    veg_pct = np.sum(borough_landcover == 0) / len(borough_landcover) * 100
                    built_pct = np.sum(borough_landcover == 2) / len(borough_landcover) * 100
                    
                    # NDVI stats
                    ndvi_mean = np.mean(borough_ndvi[borough_ndvi != -9999])
                    
                    # LST stats (filter out nodata)
                    valid_lst = borough_lst[(borough_lst != 0) & (borough_lst < 100)]
                    lst_mean = np.mean(valid_lst) if len(valid_lst) > 0 else 0
                    
                    # Carbon calculation
                    veg_mask = borough_landcover == 0
                    veg_ndvi = borough_ndvi[veg_mask]
                    
                    carbon = 0
                    carbon += np.sum(veg_ndvi > 0.6) * pixel_area_ha * 20
                    carbon += np.sum((veg_ndvi > 0.4) & (veg_ndvi <= 0.6)) * pixel_area_ha * 10
                    carbon += np.sum((veg_ndvi > 0.2) & (veg_ndvi <= 0.4)) * pixel_area_ha * 5
                    carbon += np.sum(veg_ndvi <= 0.2) * pixel_area_ha * 2
                    
                    borough_stats.append({
                        'Borough': borough_name,
                        'Vegetation (%)': veg_pct,
                        'Built-up (%)': built_pct,
                        'Avg NDVI': ndvi_mean,
                        'Avg LST (°C)': lst_mean,
                        'Carbon (tCO₂/yr)': carbon
                    })
                    
                    print(f"\n{borough_name}:")
                    print(f"  Vegetation:  {veg_pct:5.1f}%")
                    print(f"  Built-up:    {built_pct:5.1f}%")
                    print(f"  Avg NDVI:    {ndvi_mean:5.3f}")
                    print(f"  Avg LST:     {lst_mean:5.1f}°C")
                    print(f"  Carbon:      {carbon:,.0f} tCO₂/year")
                    
                except Exception as e:
                    print(f"  Error processing {borough_name}: {e}")
        
        # Save to CSV
        df = pd.DataFrame(borough_stats)
        output_path = OUTPUTS_DIR / 'tables' / 'borough_statistics.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Borough statistics saved: {output_path}")
        
        return df
    
    def run_analysis(self):
        """
        Run complete carbon analysis
        """
        
        print("="*60)
        print("NYC URBAN CARBON SEQUESTRATION ANALYSIS")
        print("Manhattan & Brooklyn")
        print("="*60)
        
        # Calculate overall carbon
        stats, total_carbon, carbon_map = self.calculate_vegetation_carbon()
        
        # Borough-level analysis
        borough_df = self.analyze_by_borough()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\n📊 Total Carbon Sequestration: {total_carbon:,.0f} tons CO₂/year")
        print(f"🌳 This is equivalent to:")
        print(f"   - Removing {total_carbon/4.6:,.0f} cars from the road for a year")
        print(f"   - {total_carbon/0.039:,.0f} tree-years of carbon absorption")
        print("="*60)
        
        return stats, total_carbon, borough_df

def main():
    """Main function"""
    analyzer = CarbonAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()