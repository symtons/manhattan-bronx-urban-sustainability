"""
Preprocessing Script for NYC Urban Sustainability Project
Processes raw satellite data into analysis-ready datasets
"""

import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, STUDY_AREA
)

class DataPreprocessor:
    """
    Handles preprocessing of satellite imagery and vector data
    """
    
    def __init__(self):
        """Initialize preprocessor"""
        self.setup_directories()
        print("✅ Preprocessor initialized")
    
    def setup_directories(self):
        """Create output directories"""
        dirs = [
            PROCESSED_DATA_DIR,
            PROCESSED_DATA_DIR / 'aligned'
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def merge_landsat_boroughs(self):
        """
        Merge Manhattan and Brooklyn Landsat files into one
        """
        print("\n" + "="*60)
        print("STEP 1: MERGING LANDSAT TILES")
        print("="*60)
        
        manhattan_path = RAW_DATA_DIR / 'landsat' / 'manhattan_landsat.tif'
        brooklyn_path = RAW_DATA_DIR / 'landsat' / 'brooklyn_landsat.tif'
        output_path = PROCESSED_DATA_DIR / 'landsat_merged.tif'
        
        try:
            # Open both files
            src_manhattan = rasterio.open(manhattan_path)
            src_brooklyn = rasterio.open(brooklyn_path)
            
            print(f"Manhattan bands: {src_manhattan.count}")
            print(f"Brooklyn bands: {src_brooklyn.count}")
            
            # Merge
            print("Merging tiles...")
            mosaic, out_trans = merge([src_manhattan, src_brooklyn])
            
            # Update metadata
            out_meta = src_manhattan.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": "lzw"
            })
            
            # Write merged file
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            src_manhattan.close()
            src_brooklyn.close()
            
            print(f"✅ Merged Landsat data")
            print(f"   Output: {output_path}")
            print(f"   Shape: {mosaic.shape}")
            print(f"   Bands: {mosaic.shape[0]}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to merge: {e}")
            return None
    
    def calculate_ndvi(self, input_path, output_path):
        """
        Calculate NDVI from merged Landsat
        
        Assumes bands are: B2 (Blue), B3 (Green), B4 (Red), B5 (NIR), NDVI
        We'll recalculate NDVI to ensure consistency
        """
        print("\n" + "="*60)
        print("STEP 2: CALCULATING NDVI")
        print("="*60)
        
        try:
            with rasterio.open(input_path) as src:
                # Read bands
                print(f"Reading from: {input_path}")
                print(f"Total bands: {src.count}")
                
                # Band 4 = Red (SR_B4), Band 5 = NIR (SR_B5) in Landsat 9
                red = src.read(3).astype('float32')  # Band 3 in the file (SR_B4)
                nir = src.read(4).astype('float32')  # Band 4 in the file (SR_B5)
                
                # Calculate NDVI
                print("Calculating NDVI...")
                np.seterr(divide='ignore', invalid='ignore')
                ndvi = (nir - red) / (nir + red)
                
                # Handle invalid values
                ndvi = np.nan_to_num(ndvi, nan=-9999, posinf=-9999, neginf=-9999)
                
                # Clip to valid range
                ndvi = np.clip(ndvi, -1, 1)
                
                print(f"NDVI range: {ndvi.min():.3f} to {ndvi.max():.3f}")
                print(f"NDVI mean: {ndvi[ndvi != -9999].mean():.3f}")
                
                # Save NDVI
                meta = src.meta.copy()
                meta.update({
                    'count': 1,
                    'dtype': 'float32',
                    'nodata': -9999,
                    'compress': 'lzw'
                })
                
                with rasterio.open(output_path, 'w', **meta) as dest:
                    dest.write(ndvi, 1)
                
                print(f"✅ NDVI calculated and saved")
                print(f"   Output: {output_path}")
                
                return output_path
                
        except Exception as e:
            print(f"❌ Failed to calculate NDVI: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def reproject_raster(self, input_path, output_path, target_crs):
        """
        Reproject raster to target CRS
        """
        print(f"\nReprojecting {input_path.name}...")
        
        try:
            with rasterio.open(input_path) as src:
                # Calculate transform for target CRS
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                
                # Update metadata
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'compress': 'lzw'
                })
                
                # Reproject
                with rasterio.open(output_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.bilinear
                        )
                
                print(f"   ✅ Reprojected to {target_crs}")
                return output_path
                
        except Exception as e:
            print(f"   ❌ Reprojection failed: {e}")
            return None
    
    def align_rasters(self):
        """
        Align all rasters to consistent CRS and resolution
        """
        print("\n" + "="*60)
        print("STEP 3: ALIGNING RASTERS")
        print("="*60)
        
        target_crs = STUDY_AREA['crs']
        aligned_dir = PROCESSED_DATA_DIR / 'aligned'
        
        files_to_align = [
            (PROCESSED_DATA_DIR / 'ndvi.tif', aligned_dir / 'ndvi_aligned.tif'),
            (RAW_DATA_DIR / 'landsat' / 'manhattan_brooklyn_lst.tif', 
             aligned_dir / 'lst_aligned.tif'),
            (RAW_DATA_DIR / 'landcover' / 'manhattan_brooklyn_worldcover.tif',
             aligned_dir / 'worldcover_aligned.tif')
        ]
        
        aligned_files = []
        
        for input_path, output_path in files_to_align:
            if input_path.exists():
                result = self.reproject_raster(input_path, output_path, target_crs)
                if result:
                    aligned_files.append(result)
            else:
                print(f"⚠️  File not found: {input_path}")
        
        print(f"\n✅ Aligned {len(aligned_files)} rasters")
        return aligned_files
    
    def clip_to_study_area(self):
        """
        Clip all rasters to study area boundaries
        """
        print("\n" + "="*60)
        print("STEP 4: CLIPPING TO STUDY AREA")
        print("="*60)
        
        # Load boundaries
        boundaries_path = RAW_DATA_DIR / 'boundaries' / 'manhattan_brooklyn_boundaries.geojson'
        
        if not boundaries_path.exists():
            print("⚠️  Boundaries file not found, skipping clipping")
            return
        
        try:
            gdf = gpd.read_file(boundaries_path)
            
            # Reproject boundaries to match rasters
            target_crs = STUDY_AREA['crs']
            gdf = gdf.to_crs(target_crs)
            
            print(f"Loaded boundaries: {len(gdf)} features")
            
            # Get geometry for clipping
            geoms = gdf.geometry.values
            
            # Files to clip
            aligned_dir = PROCESSED_DATA_DIR / 'aligned'
            files_to_clip = [
                'ndvi_aligned.tif',
                'lst_aligned.tif',
                'worldcover_aligned.tif'
            ]
            
            for filename in files_to_clip:
                input_path = aligned_dir / filename
                output_path = PROCESSED_DATA_DIR / filename.replace('_aligned', '_clipped')
                
                if not input_path.exists():
                    print(f"⚠️  Skipping {filename} (not found)")
                    continue
                
                print(f"\nClipping {filename}...")
                
                try:
                    with rasterio.open(input_path) as src:
                        # Clip
                        out_image, out_transform = mask(src, geoms, crop=True)
                        out_meta = src.meta.copy()
                        
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform,
                            "compress": "lzw"
                        })
                        
                        # Save clipped raster
                        with rasterio.open(output_path, "w", **out_meta) as dest:
                            dest.write(out_image)
                        
                        print(f"   ✅ Clipped: {output_path.name}")
                        
                except Exception as e:
                    print(f"   ❌ Failed to clip {filename}: {e}")
            
            print("\n✅ Clipping complete")
            
        except Exception as e:
            print(f"❌ Clipping failed: {e}")
    
    def generate_statistics(self):
        """
        Generate summary statistics for processed data
        """
        print("\n" + "="*60)
        print("GENERATING STATISTICS")
        print("="*60)
        
        files = [
            PROCESSED_DATA_DIR / 'ndvi_clipped.tif',
            PROCESSED_DATA_DIR / 'lst_clipped.tif'
        ]
        
        for file_path in files:
            if file_path.exists():
                try:
                    with rasterio.open(file_path) as src:
                        data = src.read(1)
                        valid_data = data[data != src.nodata]
                        
                        print(f"\n{file_path.name}:")
                        print(f"  Min: {valid_data.min():.3f}")
                        print(f"  Max: {valid_data.max():.3f}")
                        print(f"  Mean: {valid_data.mean():.3f}")
                        print(f"  Std: {valid_data.std():.3f}")
                        
                except Exception as e:
                    print(f"  ❌ Error reading {file_path.name}: {e}")
    
    def run_full_preprocessing(self):
        """
        Execute complete preprocessing workflow
        """
        print("\n" + "="*60)
        print("NYC URBAN SUSTAINABILITY - DATA PREPROCESSING")
        print("="*60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Step 1: Merge Landsat tiles
        merged_path = self.merge_landsat_boroughs()
        
        if not merged_path:
            print("❌ Merging failed, cannot continue")
            return
        
        # Step 2: Calculate NDVI
        ndvi_path = PROCESSED_DATA_DIR / 'ndvi.tif'
        self.calculate_ndvi(merged_path, ndvi_path)
        
        # Step 3: Align all rasters
        self.align_rasters()
        
        # Step 4: Clip to study area
        self.clip_to_study_area()
        
        # Step 5: Generate statistics
        self.generate_statistics()
        
        # Summary
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE!")
        print("="*60)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📁 Processed files saved to:")
        print(f"   {PROCESSED_DATA_DIR}")
        print("\n🎉 Ready for model training!")
        print("="*60)

def main():
    """Main function"""
    preprocessor = DataPreprocessor()
    preprocessor.run_full_preprocessing()

if __name__ == "__main__":
    main()