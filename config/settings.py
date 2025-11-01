"""
Central configuration file for the NYC Urban Sustainability project
"""

import os
from pathlib import Path

# ==============================================================================
# PROJECT PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TILES_DIR = DATA_DIR / "tiles"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ==============================================================================
# STUDY AREA - Manhattan & Brooklyn
# ==============================================================================
STUDY_AREA = {
    'name': 'Manhattan and Brooklyn',
    'boroughs': ['Manhattan', 'Brooklyn'],
    'bbox': {
        'west': -74.05,
        'south': 40.57,
        'east': -73.75,
        'north': 40.92
    },
    'crs': 'EPSG:32618',  # UTM Zone 18N (appropriate for NYC)
    'resolution': 10  # meters
}

# ==============================================================================
# TEMPORAL PARAMETERS
# ==============================================================================
ANALYSIS_PERIOD = {
    'year': 2024,
    'start_date': '2024-06-01',  # Summer period for peak vegetation
    'end_date': '2024-08-31',
    'season': 'Summer'
}

# ==============================================================================
# SATELLITE DATA PARAMETERS
# ==============================================================================

# Sentinel-2 (for NDVI)
SENTINEL2_PARAMS = {
    'collection': 'COPERNICUS/S2_SR_HARMONIZED',
    'cloud_threshold': 20,  # Max cloud coverage percentage
    'bands': {
        'blue': 'B2',
        'green': 'B3',
        'red': 'B4',
        'nir': 'B8',  # Near Infrared for NDVI
        'swir1': 'B11',
        'swir2': 'B12',
        'qa': 'QA60'
    }
}

# Landsat 9 (for LST - Land Surface Temperature)
LANDSAT9_PARAMS = {
    'collection': 'LANDSAT/LC09/C02/T1_L2',
    'cloud_threshold': 20,
    'bands': {
        'thermal': 'ST_B10',  # Thermal band
        'qa': 'QA_PIXEL'
    },
    'scale_factor': 0.00341802,  # Convert to Celsius
    'offset': 149.0
}

# ESA WorldCover (for training labels)
WORLDCOVER_PARAMS = {
    'collection': 'ESA/WorldCover/v200',
    'year': 2021,
    'classes': {
        10: 'tree_cover',
        20: 'shrubland',
        30: 'grassland',
        40: 'cropland',
        50: 'built_up',
        60: 'bare_sparse_vegetation',
        70: 'snow_ice',
        80: 'water',
        90: 'herbaceous_wetland',
        95: 'mangroves',
        100: 'moss_lichen'
    }
}

# ==============================================================================
# LAND COVER CLASSIFICATION
# ==============================================================================
LANDCOVER_CLASSES = {
    0: 'vegetation',
    1: 'water',
    2: 'built_up',
    3: 'bare_open'
}

NUM_CLASSES = len(LANDCOVER_CLASSES)

# ==============================================================================
# MODEL PARAMETERS
# ==============================================================================
MODEL_CONFIG = {
    'architecture': 'unet',
    'encoder': 'resnet18',
    'input_channels': 4,  # RGB + NDVI
    'num_classes': NUM_CLASSES,
    'image_size': 512,  # pixels
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'early_stopping_patience': 10,
    'device': 'cuda'  # Will fallback to 'cpu' if CUDA unavailable
}

# ==============================================================================
# NDVI THRESHOLDS
# ==============================================================================
NDVI_THRESHOLDS = {
    'water': -1.0,
    'bare': 0.0,
    'sparse_veg': 0.2,
    'moderate_veg': 0.4,
    'dense_veg': 0.6,
    'very_dense_veg': 0.8
}

# ==============================================================================
# TEMPERATURE THRESHOLDS (Celsius)
# ==============================================================================
LST_THRESHOLDS = {
    'cool': 25,
    'moderate': 30,
    'warm': 33,
    'hot': 35,
    'extreme': 38
}

# ==============================================================================
# NYC OPEN DATA
# ==============================================================================
NYC_DATA_URLS = {
    'borough_boundaries': 'https://data.cityofnewyork.us/resource/7t3b-ywvw.geojson',
    'parks': 'https://data.cityofnewyork.us/resource/ghu2-eden.geojson'
}

# ==============================================================================
# PROCESSING PARAMETERS
# ==============================================================================
TILE_CONFIG = {
    'size': 512,  # pixels
    'overlap': 128,  # pixels overlap between tiles
    'min_valid_pixels': 0.7  # Minimum 70% valid data per tile
}

# ==============================================================================
# VISUALIZATION
# ==============================================================================
VIZ_CONFIG = {
    'dpi': 300,
    'figsize': (12, 8),
    'cmap_ndvi': 'RdYlGn',
    'cmap_lst': 'hot_r',
    'cmap_landcover': 'tab10'
}

# ==============================================================================
# DASHBOARD
# ==============================================================================
DASHBOARD_CONFIG = {
    'title': 'NYC Urban Sustainability Intelligence System',
    'subtitle': 'Manhattan & Brooklyn Analysis',
    'map_center': [40.7128, -73.9060],  # NYC center
    'map_zoom': 11,
    'theme': 'light'
}