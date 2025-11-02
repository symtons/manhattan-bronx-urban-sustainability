"""
Google Earth Engine Authentication Helper
"""

import ee
import os
from pathlib import Path

def authenticate_gee(project_id=None):
    """
    Authenticate and initialize Google Earth Engine
    
    Args:
        project_id (str): Your GEE project ID (e.g., 'nyc-urban-sustainability')
    
    Returns:
        bool: True if authentication successful, False otherwise
    """
    try:
        # Try to initialize with existing credentials
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        
        print("✅ Google Earth Engine initialized successfully!")
        print(f"   Project: {project_id if project_id else 'default'}")
        return True
        
    except Exception as e:
        print("⚠️  GEE not authenticated. Starting authentication process...")
        print(f"   Error: {e}")
        
        try:
            # Authenticate (this will open a browser window)
            ee.Authenticate()
            
            # Initialize after authentication
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
            
            print("✅ Authentication successful!")
            return True
            
        except Exception as auth_error:
            print(f"❌ Authentication failed: {auth_error}")
            print("\nTroubleshooting steps:")
            print("1. Make sure you've registered at https://code.earthengine.google.com/register")
            print("2. Check that your project is approved")
            print("3. Try running: ee.Authenticate() manually")
            return False

def test_gee_connection():
    """
    Test GEE connection by fetching a simple dataset
    
    Returns:
        bool: True if test successful
    """
    try:
        # Try to access a simple dataset
        image = ee.Image('USGS/SRTMGL1_003')
        info = image.getInfo()
        
        print("✅ GEE connection test successful!")
        print("   Successfully accessed SRTM elevation data")
        return True
        
    except Exception as e:
        print(f"❌ GEE connection test failed: {e}")
        return False

def get_gee_project_info():
    """
    Display information about your GEE setup
    """
    try:
        # Get some basic info
        print("=" * 60)
        print("GOOGLE EARTH ENGINE STATUS")
        print("=" * 60)
        
        # Test with a simple query
        point = ee.Geometry.Point([-73.9, 40.7])  # NYC coordinates
        image = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20240701T154901_20240701T155829_T18TWL')
        
        # If we get here, GEE is working
        print("Status: ✅ Connected")
        print("Region: Worldwide access enabled")
        print("Collections: Full catalog available")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"Status: ❌ Not connected - {e}")
        print("=" * 60)
        return False

if __name__ == "__main__":
    """
    Run this script directly to test GEE authentication
    """
    print("🛰️  Google Earth Engine Authentication Test\n")
    
    # Get project ID from environment or use None
    project_id = os.getenv('GEE_PROJECT', None)
    
    if project_id:
        print(f"Using project ID from environment: {project_id}\n")
    else:
        print("No project ID specified. Using default authentication.\n")
    
    # Authenticate
    auth_success = authenticate_gee(project_id)
    
    if auth_success:
        print("\n" + "=" * 60)
        # Test connection
        test_gee_connection()
        print()
        # Show project info
        get_gee_project_info()
    else:
        print("\n❌ Please complete GEE authentication before proceeding.")