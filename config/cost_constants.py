"""
Cost estimates and carbon sequestration rates for urban greening interventions
All costs in USD, carbon rates in tons CO2/year
Based on literature review and NYC-specific data
"""

# ==============================================================================
# INTERVENTION COSTS
# ==============================================================================

INTERVENTION_COSTS = {
    'street_tree': {
        'cost_per_unit': 900,  # USD per tree (includes planting + 2yr maintenance)
        'unit': 'tree',
        'source': 'NYC Parks Department Street Tree Planting Program, 2024',
        'notes': 'Average cost for medium-sized street tree with initial care'
    },
    
    'green_roof': {
        'cost_per_m2': 100,  # USD per square meter
        'unit': 'm²',
        'source': 'Green Roofs for Healthy Cities Annual Report, 2023',
        'notes': 'Extensive green roof system, excludes structural reinforcement'
    },
    
    'pocket_park': {
        'cost_per_acre': 500000,  # USD per acre
        'cost_per_site': 80000,  # Small park (0.16 acre / 650 m²)
        'unit': 'site',
        'source': 'Trust for Public Land, NYC Park Equity Report, 2024',
        'notes': 'Includes basic amenities, landscaping, and infrastructure'
    },
    
    'bioswale': {
        'cost_per_linear_ft': 150,  # USD per linear foot
        'unit': 'linear_ft',
        'source': 'NYC DEP Green Infrastructure Program',
        'notes': 'Right-of-way bioswale with plantings'
    },
    
    'green_wall': {
        'cost_per_m2': 200,  # USD per square meter
        'unit': 'm²',
        'source': 'Urban Green Council, NYC Green Infrastructure Guide',
        'notes': 'Vertical garden system on building facade'
    }
}

# ==============================================================================
# CARBON SEQUESTRATION RATES
# ==============================================================================

CARBON_SEQUESTRATION = {
    'urban_tree': {
        'rate_per_year': 0.6,  # tons CO2/year per mature tree
        'unit': 'tree',
        'source': 'Nowak et al. (2013), Urban Forestry & Urban Greening',
        'notes': 'Average for temperate urban trees, ages 10-30 years'
    },
    
    'green_roof': {
        'rate_per_m2': 0.015,  # tons CO2/year per m²
        'unit': 'm²',
        'source': 'Getter et al. (2009), Environmental Science & Technology',
        'notes': 'Extensive sedum-based green roof'
    },
    
    'grass_lawn': {
        'rate_per_m2': 0.002,  # tons CO2/year per m²
        'unit': 'm²',
        'source': 'Townsend-Small & Czimczik (2010), Geophysical Research',
        'notes': 'Urban lawn with regular maintenance'
    },
    
    'shrubland': {
        'rate_per_m2': 0.005,  # tons CO2/year per m²
        'unit': 'm²',
        'source': 'Liu et al. (2016), Urban Forestry & Urban Greening',
        'notes': 'Mixed shrub plantings'
    },
    
    'mixed_vegetation': {
        'rate_per_m2': 0.01,  # tons CO2/year per m²
        'unit': 'm²',
        'source': 'McPherson et al. (2011), USDA Forest Service',
        'notes': 'Mixed trees, shrubs, and groundcover'
    }
}

# ==============================================================================
# URBAN HEAT ISLAND MITIGATION
# ==============================================================================

LST_REDUCTION_ESTIMATES = {
    'street_trees': {
        'reduction_per_unit': 0.003,  # °C per tree per km²
        'max_reduction': 1.5,  # Maximum expected reduction in °C
        'source': 'Ziter et al. (2019), PNAS',
        'notes': 'Effect radius ~100m per tree'
    },
    
    'green_roof': {
        'reduction_building': 2.0,  # °C at building surface
        'reduction_ambient': 0.5,  # °C in surrounding area (within 50m)
        'source': 'Santamouris (2014), Energy and Buildings',
        'notes': 'Peak summer afternoon conditions'
    },
    
    'park': {
        'reduction_center': 3.0,  # °C at park center
        'reduction_edge': 1.0,  # °C at 100m from park edge
        'distance_effect': 400,  # meters of cooling influence
        'source': 'Jaganmohan et al. (2016), Urban Forestry & Urban Greening'
    },
    
    'tree_canopy': {
        'reduction_per_10pct_increase': 0.5,  # °C per 10% canopy increase
        'source': 'McDonald et al. (2021), Nature Communications'
    }
}

# ==============================================================================
# CO-BENEFITS VALUATION (Annual benefits in USD)
# ==============================================================================

COBENEFITS_VALUE = {
    'air_quality': {
        'per_tree': 15,  # USD/year (PM2.5, NO2, O3 removal)
        'source': 'i-Tree Eco Model, USDA Forest Service'
    },
    
    'stormwater': {
        'per_tree': 12,  # USD/year (runoff reduction)
        'per_m2_green_roof': 5,  # USD/year
        'source': 'NYC DEP Green Infrastructure Valuation'
    },
    
    'energy_savings': {
        'per_tree_shading': 35,  # USD/year (cooling costs)
        'per_m2_green_roof': 8,  # USD/year
        'source': 'NYC Energy Conservation Code, 2023'
    },
    
    'property_value': {
        'increase_percent': 7,  # % increase with street trees
        'source': 'Donovan & Butry (2010), Landscape and Urban Planning'
    },
    
    'health': {
        'heat_illness_prevention': 50,  # USD/year per capita in cooled area
        'source': 'Nature Conservancy, Cool Cities Report, 2021'
    }
}

# ==============================================================================
# PLANTING DENSITIES
# ==============================================================================

PLANTING_DENSITY = {
    'street_trees': {
        'per_km': 100,  # trees per km of street
        'per_block': 20,  # trees per city block
        'source': 'NYC Urban Forestry Standards'
    },
    
    'park_trees': {
        'per_hectare': 150,  # trees per hectare
        'source': 'Urban Parks Institute Standards'
    },
    
    'green_roof_coverage': {
        'percent_suitable_roofs': 0.65,  # 65% of flat roofs suitable
        'source': 'NYC Green Roof Feasibility Study, 2022'
    }
}

# ==============================================================================
# PRIORITY SCORING WEIGHTS
# ==============================================================================

PRIORITY_WEIGHTS = {
    'heat_index': 0.40,        # LST above average
    'vegetation_gap': 0.30,    # NDVI below target
    'carbon_potential': 0.20,  # Potential sequestration gain
    'population_density': 0.10 # Impact on people
}

# ==============================================================================
# INTERVENTION SELECTION CRITERIA
# ==============================================================================

INTERVENTION_CRITERIA = {
    'street_trees': {
        'trigger_ndvi': 0.25,  # Recommend if NDVI below this
        'min_trees': 50,
        'max_trees': 500,
        'priority': 'HIGH'
    },
    
    'green_roof': {
        'trigger_building_density': 0.60,  # Recommend if >60% built-up
        'min_area_m2': 1000,
        'max_area_m2': 20000,
        'priority': 'MEDIUM'
    },
    
    'pocket_park': {
        'trigger_ndvi': 0.25,
        'require_vacant_land': True,
        'min_sites': 1,
        'max_sites': 5,
        'priority': 'HIGH'
    }
}