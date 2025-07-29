# =============================================================================
#  GEBCO Bathymetry Preprocessing for Wave Downscaling Model
#  ---------------------------------------------------------
#  Description:
#  This script processes a global GEBCO NetCDF file to generate the specific
#  static input files required by the physics-informed ConvLSTM model.
#
#  Workflow:
#  1. Defines the irregular grid points of interest (buoys and reanalysis points).
#  2. Creates a new, regular grid that encompasses these points. This is crucial
#     for correctly calculating spatial derivatives (slope, aspect) and for
#     providing a structured input to the ConvLSTM.
#  3. Loads the specified section from the GEBCO .nc file.
#  4. Interpolates the high-resolution bathymetry onto the new regular grid.
#  5. Creates a binary land/sea mask (1 for land, 0 for sea).
#  6. Saves the final bathymetry grid and land mask as CSV files, ready for
#     use in the main modeling script.
#
#  Required Libraries:
#  - xarray: For handling NetCDF files (pip install xarray)
#  - netCDF4: The engine for reading .nc files (pip install netCDF4)
#  - pandas, numpy, pathlib
# =============================================================================

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# --- Input/Output Paths ---
# PLEASE UPDATE THESE PATHS TO MATCH YOUR SYSTEM
BASE_DIR = Path('/home/artorias/Desktop/Paper 3_FInal/Data')
# Path to your global GEBCO 2024 NetCDF file
GEBCO_NC_PATH = BASE_DIR / 'Bathymetry' / 'gebco_2024_n26.0_s23.5_w56.2_e58.8.nc'
# Directory where the output CSVs will be saved
OUTPUT_DIR = BASE_DIR / 'Bathymetry'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Output File Names ---
# These names align with the inputs required by the main ConvLSTM script
BATHYMETRY_CSV_PATH = OUTPUT_DIR / 'processed_bathymetry.csv'
LAND_MASK_CSV_PATH = OUTPUT_DIR / 'land_sea_mask.csv'

# --- Grid Definition ---
# The 21 specific, irregular points from your master dataset
points_of_interest = [
    (24.0, 57.2), (24.2, 57.2), (24.2, 57.4), (24.4, 56.8), (24.4, 57.0),
    (24.4, 57.2), (24.4, 57.4), (24.6, 56.6), (24.6, 56.8), (24.6, 57.0),
    (24.6, 57.2), (24.8, 56.6), (24.8, 56.8), (24.8, 57.0), (24.8, 57.2),
    (25.0, 56.6), (25.0, 56.8), (25.0, 57.0), (25.2, 56.6), (25.4, 56.6)
]

# Define the resolution for our new regular grid.
# 0.02 degrees is a reasonably high resolution (~2.2 km).
GRID_RESOLUTION = 0.02  # in degrees

# =============================================================================
# 2. PROCESSING SCRIPT
# =============================================================================

def create_regular_grid(points, resolution):
    """
    Creates a regular grid that encompasses all irregular points of interest.
    """
    print("--- 1. Defining new regular grid boundaries ---")
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    # Define boundaries with a small buffer
    lat_min, lat_max = min(lats) - resolution, max(lats) + resolution
    lon_min, lon_max = min(lons) - resolution, max(lons) + resolution

    # Create new coordinate arrays
    new_lats = np.arange(lat_min, lat_max, resolution)
    new_lons = np.arange(lon_min, lon_max, resolution)

    print(f"  -> Grid Extent: Lats from {new_lats.min():.2f} to {new_lats.max():.2f}")
    print(f"  -> Grid Extent: Lons from {new_lons.min():.2f} to {new_lons.max():.2f}")
    print(f"  -> New Grid Shape: ({len(new_lats)}, {len(new_lons)})")
    
    return new_lats, new_lons

def process_gebco_data(gebco_path, lats, lons):
    """
    Loads GEBCO data, slices the area of interest, and interpolates
    it onto the new regular grid using a higher-quality method.
    """
    if not gebco_path.exists():
        raise FileNotFoundError(
            f"GEBCO file not found at: {gebco_path}\n"
            "Please download the GEBCO 2024 grid and update the GEBCO_NC_PATH."
        )

    print("\n--- 2. Loading and interpolating GEBCO data ---")
    with xr.open_dataset(gebco_path) as ds:
        ds_subset = ds.sel(
            lat=slice(lats.min(), lats.max()),
            lon=slice(lons.min(), lons.max())
        )
        print("  -> Sliced GEBCO data to area of interest.")

        # --- EDITED: Use higher-quality cubic interpolation ---
        # This provides a smoother, more physically realistic surface.
        # Requires the 'scipy' library to be installed.
        interpolated_ds = ds_subset.interp(
            lat=lats, lon=lons, method='cubic'
        )
        print("  -> Interpolation to new regular grid complete (using cubic method).")
        
        # --- NEW: Fill any remaining NaN values ---
        # This handles missing values at the edges of the domain by propagating
        # the nearest valid data point into the NaN cells.
        filled_ds = interpolated_ds.interpolate_na(dim='lon', method='nearest', fill_value="extrapolate")
        filled_ds = filled_ds.interpolate_na(dim='lat', method='nearest', fill_value="extrapolate")
        
        print("  -> Filled remaining missing values using nearest neighbor extrapolation.")
        
        elevation_grid = filled_ds['elevation']
        
    return elevation_grid

def generate_and_save_outputs(elevation_grid, bathy_path, mask_path):
    """
    Generates the final bathymetry and land mask files and saves them to CSV.
    """
    print("\n--- 3. Generating and saving output files ---")
    
    # --- Generate Bathymetry CSV ---
    bathy_df = elevation_grid.to_dataframe().reset_index()
    bathy_df['depth'] = -bathy_df['elevation']
    bathy_df = bathy_df[['lat', 'lon', 'depth']]
    
    bathy_df.to_csv(bathy_path, index=False)
    print(f"  -> Saved processed bathymetry to: {bathy_path}")
    print("     This file contains the depth for each point on the new regular grid.")

    # --- Generate Land/Sea Mask CSV ---
    land_mask_grid = xr.where(elevation_grid >= 0, 1, 0)
    mask_df = land_mask_grid.to_dataframe(name='is_land').reset_index()
    
    mask_df.to_csv(mask_path, index=False)
    print(f"  -> Saved land/sea mask to: {mask_path}")
    print("     This file contains a 'is_land' flag (1=land, 0=sea) for each grid point.")
    
if __name__ == '__main__':
    # Step 1: Define the target grid
    regular_lats, regular_lons = create_regular_grid(points_of_interest, GRID_RESOLUTION)
    
    # Step 2: Load and process the GEBCO data
    elevation_data = process_gebco_data(GEBCO_NC_PATH, regular_lats, regular_lons)
    
    # Step 3: Generate and save the final output files
    generate_and_save_outputs(elevation_data, BATHYMETRY_CSV_PATH, LAND_MASK_CSV_PATH)
    
    print("\n--- Preprocessing Complete ---")
    print("The files 'processed_bathymetry.csv' and 'land_sea_mask.csv' have been created.")
    print("You can now proceed with updating and running the main ConvLSTM modeling script.")
