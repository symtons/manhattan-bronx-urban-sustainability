"""
Identify Priority Zones for Urban Greening Interventions
"""

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from scipy import ndimage
from rasterio.features import shapes

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR, OUTPUTS_DIR
from config.cost_constants import INTERVENTION_COSTS, CARBON_SEQUESTRATION

class PriorityAnalyzer:
    """
    Identify priority zones for urban greening
    """
    
    def __init__(self):
        """Initialize analyzer"""
        
        self.landcover_path = PROCESSED_DATA_DIR / 'worldcover_final.tif'
        self.ndvi_path = PROCESSED_DATA_DIR / 'ndvi_final.tif'
        self.lst_path = PROCESSED_DATA_DIR / 'lst_final.tif'
        self.carbon_path = OUTPUTS_DIR / 'maps' / 'carbon_sequestration.tif'
        
        print("✅ Priority Analyzer initialized")
    
    def calculate_priority_score(self):
        """
        Calculate priority score for each pixel
        
        Score components:
        - Vegetation deficit (40%): Low NDVI = high priority
        - Heat stress (30%): High LST = high priority  
        - Carbon potential (20%): Low current carbon = high opportunity
        - Built-up proximity (10%): Near development = easier to implement
        """
        
        print("\n" + "="*60)
        print("CALCULATING PRIORITY SCORES")
        print("="*60)
        
        # Load all rasters
        with rasterio.open(self.landcover_path) as lc_src, \
             rasterio.open(self.ndvi_path) as ndvi_src, \
             rasterio.open(self.lst_path) as lst_src, \
             rasterio.open(self.carbon_path) as carbon_src:
            
            landcover = lc_src.read(1)
            ndvi = ndvi_src.read(1)
            lst = lst_src.read(1)
            carbon = carbon_src.read(1)
            
            meta = ndvi_src.meta.copy()
        
        print("\nNormalizing factors...")
        
        # Initialize score map
        priority_score = np.zeros_like(ndvi, dtype=np.float32)
        
        # Factor 1: Vegetation Deficit (40%)
        # Lower NDVI = higher priority
        # Normalize NDVI to 0-1 range, then invert
        ndvi_normalized = np.clip((ndvi - ndvi.min()) / (ndvi.max() - ndvi.min()), 0, 1)
        vegetation_deficit = 1 - ndvi_normalized  # Invert: low NDVI = high score
        
        # Factor 2: Heat Stress (30%)
        # Higher LST = higher priority
        # Filter valid LST values
        valid_lst_mask = (lst > 0) & (lst < 60)
        lst_clean = np.where(valid_lst_mask, lst, np.nan)
        lst_normalized = np.where(
            valid_lst_mask,
            (lst - np.nanmin(lst_clean)) / (np.nanmax(lst_clean) - np.nanmin(lst_clean)),
            0
        )
        
        # Factor 3: Carbon Potential (20%)
        # Lower current carbon = higher opportunity
        carbon_normalized = np.clip((carbon - carbon.min()) / (carbon.max() - carbon.min() + 1e-6), 0, 1)
        carbon_opportunity = 1 - carbon_normalized
        
        # Factor 4: Built-up Proximity (10%)
        # Near built-up areas = easier implementation
        built_up_mask = (landcover == 50).astype(float)
        # Distance from built-up (closer = higher priority)
        distance_from_buildup = ndimage.distance_transform_edt(1 - built_up_mask)
        # Normalize and invert (closer = higher score)
        proximity_normalized = 1 - np.clip(distance_from_buildup / distance_from_buildup.max(), 0, 1)
        
        # Combine factors with weights
        priority_score = (
            0.40 * vegetation_deficit +
            0.30 * lst_normalized +
            0.20 * carbon_opportunity +
            0.10 * proximity_normalized
        )
        
        # Mask out water and existing dense vegetation
        water_mask = (landcover == 80)
        dense_veg_mask = (ndvi > 0.7)
        priority_score[water_mask | dense_veg_mask] = 0
        
        # Normalize to 0-100 scale
        priority_score = priority_score * 100
        
        print(f"\nPriority score range: {priority_score.min():.1f} - {priority_score.max():.1f}")
        
        # Save priority map
        output_path = OUTPUTS_DIR / 'maps' / 'priority_zones.tif'
        meta.update({'dtype': 'float32'})
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(priority_score, 1)
        
        print(f"✅ Priority map saved: {output_path}")
        
        return priority_score, meta
    
    def identify_top_zones(self, priority_score, meta, n_zones=10):
        """
        Identify top N priority zones
        """
        
        print("\n" + "="*60)
        print(f"IDENTIFYING TOP {n_zones} PRIORITY ZONES")
        print("="*60)
        
        # Define priority threshold (top 10% of scores)
        threshold = np.percentile(priority_score[priority_score > 0], 90)
        
        high_priority = priority_score >= threshold
        
        # Label connected regions
        labeled_array, num_features = ndimage.label(high_priority)
        
        print(f"\nFound {num_features} high-priority zones")
        
        # Calculate statistics for each zone
        zones = []
        
        pixel_area_ha = (meta['transform'][0] * abs(meta['transform'][4])) / 10000
        
        for zone_id in range(1, min(num_features + 1, n_zones * 3)):  # Check 3x zones to get best N
            zone_mask = labeled_array == zone_id
            zone_size = np.sum(zone_mask)
            
            # Skip very small zones
            if zone_size < 50:  # Less than 50 pixels
                continue
            
            avg_score = np.mean(priority_score[zone_mask])
            area_ha = zone_size * pixel_area_ha
            
            # Get zone centroid
            y_coords, x_coords = np.where(zone_mask)
            centroid_y = int(np.mean(y_coords))
            centroid_x = int(np.mean(x_coords))
            
            # Convert to geographic coordinates
            transform = meta['transform']
            lon = transform[2] + centroid_x * transform[0]
            lat = transform[5] + centroid_y * transform[4]
            
            zones.append({
                'zone_id': zone_id,
                'priority_score': avg_score,
                'area_ha': area_ha,
                'area_acres': area_ha * 2.47,
                'centroid_lat': lat,
                'centroid_lon': lon,
                'pixels': zone_size
            })
        
        # Sort by priority score and take top N
        zones_df = pd.DataFrame(zones)
        zones_df = zones_df.nlargest(n_zones, 'priority_score').reset_index(drop=True)
        zones_df['rank'] = range(1, len(zones_df) + 1)
        
        print(f"\nTop {len(zones_df)} Priority Zones:")
        print("-" * 60)
        
        for _, zone in zones_df.iterrows():
            print(f"\nZone {zone['rank']}:")
            print(f"  Priority Score: {zone['priority_score']:.1f}/100")
            print(f"  Area: {zone['area_ha']:.1f} ha ({zone['area_acres']:.1f} acres)")
            print(f"  Location: ({zone['centroid_lat']:.4f}, {zone['centroid_lon']:.4f})")
        
        return zones_df
    
    def generate_recommendations(self, zones_df):
        """
        Generate specific recommendations for each zone
        """
        
        print("\n" + "="*60)
        print("GENERATING RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        for _, zone in zones_df.iterrows():
            area_ha = zone['area_ha']
            area_m2 = area_ha * 10000
            
            # Determine primary intervention based on zone size
            if area_ha < 0.5:  # Small zones (< 0.5 ha)
                intervention = 'street_trees'
                n_units = int(area_ha * 100)  # ~100 trees per hectare
                cost_per_unit = INTERVENTION_COSTS['street_tree']['cost_per_unit']
                total_cost = n_units * cost_per_unit
                carbon_impact = n_units * CARBON_SEQUESTRATION['urban_tree']['rate_per_year']
                unit_desc = f"{n_units} street trees"
                
            elif area_ha < 2:  # Medium zones (0.5-2 ha)
                intervention = 'pocket_park'
                n_units = max(1, int(area_ha / 0.16))  # Parks ~0.16 ha each
                cost_per_unit = INTERVENTION_COSTS['pocket_park']['cost_per_site']
                total_cost = n_units * cost_per_unit
                carbon_impact = area_m2 * CARBON_SEQUESTRATION['mixed_vegetation']['rate_per_m2']
                unit_desc = f"{n_units} pocket park(s)"
                
            else:  # Large zones (> 2 ha)
                intervention = 'green_roof'
                suitable_area_m2 = area_m2 * 0.3  # Assume 30% suitable for green roofs
                cost_per_m2 = INTERVENTION_COSTS['green_roof']['cost_per_m2']
                total_cost = suitable_area_m2 * cost_per_m2
                carbon_impact = suitable_area_m2 * CARBON_SEQUESTRATION['green_roof']['rate_per_m2']
                unit_desc = f"{suitable_area_m2:,.0f} m² green roofs"
            
            # Calculate payback and ROI
            carbon_value_per_ton = 50  # USD per ton CO2 (social cost of carbon)
            annual_benefit = carbon_impact * carbon_value_per_ton
            payback_years = total_cost / annual_benefit if annual_benefit > 0 else 999
            
            recommendations.append({
                'Zone Rank': zone['rank'],
                'Priority Score': zone['priority_score'],
                'Area (ha)': area_ha,
                'Intervention': intervention,
                'Description': unit_desc,
                'Estimated Cost (USD)': total_cost,
                'Annual Carbon (tCO₂)': carbon_impact,
                'Annual Benefit (USD)': annual_benefit,
                'Payback (years)': payback_years,
                'Latitude': zone['centroid_lat'],
                'Longitude': zone['centroid_lon']
            })
        
        rec_df = pd.DataFrame(recommendations)
        
        # Save recommendations
        output_path = OUTPUTS_DIR / 'tables' / 'priority_recommendations.csv'
        rec_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Recommendations saved: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("RECOMMENDATIONS SUMMARY")
        print("="*60)
        
        total_cost = rec_df['Estimated Cost (USD)'].sum()
        total_carbon = rec_df['Annual Carbon (tCO₂)'].sum()
        
        print(f"\nTotal Investment: ${total_cost:,.0f}")
        print(f"Total Annual Carbon Impact: {total_carbon:,.0f} tCO₂/year")
        print(f"Average Cost per ton CO₂: ${total_cost/total_carbon:,.0f}")
        
        print(f"\nTop 3 Priority Zones:")
        for idx, row in rec_df.head(3).iterrows():
            print(f"\n  Zone {row['Zone Rank']}: {row['Description']}")
            print(f"    Cost: ${row['Estimated Cost (USD)']:,.0f}")
            print(f"    Impact: {row['Annual Carbon (tCO₂)']:.0f} tCO₂/year")
            print(f"    Payback: {row['Payback (years)']:.1f} years")
        
        return rec_df
    
    def run_analysis(self):
        """Run complete priority analysis"""
        
        print("="*60)
        print("NYC URBAN GREENING PRIORITY ANALYSIS")
        print("="*60)
        
        # Calculate priority scores
        priority_score, meta = self.calculate_priority_score()
        
        # Identify top zones
        zones_df = self.identify_top_zones(priority_score, meta, n_zones=10)
        
        # Generate recommendations
        rec_df = self.generate_recommendations(zones_df)
        
        print("\n" + "="*60)
        print("PRIORITY ANALYSIS COMPLETE!")
        print("="*60)
        
        return priority_score, zones_df, rec_df

def main():
    """Main function"""
    analyzer = PriorityAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()