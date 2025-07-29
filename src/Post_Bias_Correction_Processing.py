import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================

# --- Input Paths ---
BASE_DIR = Path('/home/artorias/Desktop/Paper 3_FInal/Data')
QDM_INPUT_DIR = BASE_DIR / 'Bias_Correction' / 'qdm_corrected'
CATBOOST_INPUT_DIR = BASE_DIR / 'Bias_Correction' / 'catboost_corrected'

# Paths to your new, clean data sources
GEBCO_PATH = BASE_DIR / 'Bathymetry' / 'gebco_2024_n26.0_s23.5_w56.2_e58.8.nc'
HYCOM_PATH = BASE_DIR / 'Reanalysis' / 'HYCOM'  / 'HYCOM_SSH_Sohar_2018_2024_Paper3.csv'

# --- Output Paths ---
ENRICHED_DATA_DIR = BASE_DIR / 'modeling_dataset'
QDM_OUTPUT_DIR = ENRICHED_DATA_DIR / 'qdm_enriched'
CATBOOST_OUTPUT_DIR = ENRICHED_DATA_DIR / 'catboost_enriched'

# Create output directories
QDM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CATBOOST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def load_yearly_csvs(data_dir: Path, file_suffix: str) -> pd.DataFrame:
    """Loads and combines yearly bias-corrected CSV files."""
    print(f"Loading yearly data from: {data_dir}")
    all_years_df = []
    for year in range(2018, 2025):
        file_path = data_dir / f"{year}_{file_suffix}.csv"
        if file_path.exists():
            df_year = pd.read_csv(file_path, parse_dates=['time'])
            all_years_df.append(df_year)
    if not all_years_df: raise FileNotFoundError(f"No files with suffix '{file_suffix}' found.")
    
    combined_df = pd.concat(all_years_df, ignore_index=True)
    return combined_df

def process_bathymetry(gebco_path: Path, grid_points: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates GEBCO bathymetry onto the dataset's grid and calculates NaN-free slopes
    using a robust spatial interpolation method to fill gaps in the pivoted grid.
    """
    print("Processing bathymetry from GEBCO file...")
    ds_gebco = xr.open_dataset(gebco_path)
    
    # Prepare GEBCO data for interpolation
    df_gebco = ds_gebco.to_dataframe().reset_index()
    gebco_points = df_gebco[['lon', 'lat']].values
    gebco_values = df_gebco['elevation'].values
    
    # Interpolate onto our specific grid points
    grid_lons = grid_points['lon'].values
    grid_lats = grid_points['lat'].values
    interpolated_elevation = griddata(gebco_points, gebco_values, (grid_lons, grid_lats), method='cubic')
    
    if np.isnan(interpolated_elevation).any():
        print("  -> Filling NaNs from initial interpolation (points likely outside GEBCO convex hull)...")
        interpolated_elevation_linear = griddata(gebco_points, gebco_values, (grid_lons, grid_lats), method='linear')
        interpolated_elevation = np.where(np.isnan(interpolated_elevation), interpolated_elevation_linear, interpolated_elevation)

    bathy_df = grid_points.copy()
    bathy_df['depth_new'] = -interpolated_elevation
    
    # --- START: ROBUST SLOPE CALCULATION FIX ---
    print("  -> Creating complete surface for robust slope calculation...")
    # Pivot the irregular points onto a grid, which will create NaNs
    pivoted_depth = bathy_df.pivot(index='lat', columns='lon', values='depth_new')

    # Create coordinate arrays for the grid
    x_coords = pivoted_depth.columns.values
    y_coords = pivoted_depth.index.values
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Create a mask of the valid, non-NaN data points
    valid_mask = ~np.isnan(pivoted_depth.values)
    
    # Get the coordinates and values of the valid points
    valid_points = np.array([x_grid[valid_mask], y_grid[valid_mask]]).T
    valid_values = pivoted_depth.values[valid_mask]

    # Use griddata to interpolate and fill the NaN values, creating a complete surface
    # Use 'cubic' for a smooth surface, with 'linear' as a fallback for robustness
    interpolated_grid = griddata(valid_points, valid_values, (x_grid, y_grid), method='cubic')
    fallback_grid = griddata(valid_points, valid_values, (x_grid, y_grid), method='linear')
    final_grid_depth = np.where(np.isnan(interpolated_grid), fallback_grid, interpolated_grid)

    # Calculate gradients on the new, complete, NaN-free grid
    grad_ns, grad_ew = np.gradient(final_grid_depth, y_coords, x_coords)
    # --- END: ROBUST SLOPE CALCULATION FIX ---

    # Map gradients back to the original dataframe structure
    lat_to_idx = {lat: i for i, lat in enumerate(y_coords)}
    lon_to_idx = {lon: i for i, lon in enumerate(x_coords)}
    
    bathy_df['slope_ns_new'] = bathy_df.apply(lambda row: grad_ns[lat_to_idx[row['lat']], lon_to_idx[row['lon']]], axis=1)
    bathy_df['slope_ew_new'] = bathy_df.apply(lambda row: grad_ew[lat_to_idx[row['lat']], lon_to_idx[row['lon']]], axis=1)
    
    print("✓ Bathymetry and slope processing complete.")
    return bathy_df

def process_ssh(hycom_path: Path, main_df: pd.DataFrame) -> pd.DataFrame:
    """Efficiently maps HYCOM SSH data to the main dataset grid."""
    print("Processing sea surface height from HYCOM file...")
    df_hycom = pd.read_csv(hycom_path, parse_dates=['date'])
    df_hycom.rename(columns={'date': 'time', 'latitude': 'lat', 'longitude': 'lon', 'surface_elevation': 'ssh_new'}, inplace=True)

    # Find the nearest HYCOM spatial point for each of our grid points (once)
    our_points = main_df[['lat', 'lon']].drop_duplicates()
    hycom_points = df_hycom[['lat', 'lon']].drop_duplicates()
    
    tree = cKDTree(hycom_points)
    _, indices = tree.query(our_points, k=1)
    
    # Create a mapping from our point to the nearest HYCOM point
    nearest_hycom_points = hycom_points.iloc[indices].reset_index(drop=True)
    mapping = pd.concat([our_points.reset_index(drop=True), nearest_hycom_points.add_suffix('_hycom')], axis=1)

    # --- FIX STARTS HERE ---

    # Step 1: Add the mapping to our main dataframe.
    # This gives each row in main_df the coordinates of its nearest HYCOM neighbor.
    main_df_with_mapping = pd.merge(main_df, mapping, on=['lat', 'lon'], how='left')
    
    # Step 2: Merge with the HYCOM data using the mapped coordinates and time.
    # This looks up the ssh_new value for the correct time and nearest neighbor location.
    main_df_with_ssh = pd.merge(
        main_df_with_mapping,
        df_hycom[['time', 'lat', 'lon', 'ssh_new']],
        left_on=['time', 'lat_hycom', 'lon_hycom'],
        right_on=['time', 'lat', 'lon'],
        how='left'
    )
    
    # --- FIX ENDS HERE ---

    print("SSH processing complete.")
    # The result has the same index as the original main_df, so we can return the column directly.
    return main_df_with_ssh['ssh_new']

# =============================================================================
# 3. MAIN WORKFLOW
# =============================================================================

def enrich_dataset(input_dir: Path, output_dir: Path, method: str):
    """Main workflow to clean, enrich, and save a dataset."""
    print("\n" + "="*80)
    print(f"ENRICHING DATASET FOR METHOD: {method.upper()}")
    print("="*80)

    # 1. Load Data
    file_suffix = 'corrected_spatiotemporal' if method == 'catboost' else 'corrected_qdm'
    df_main = load_yearly_csvs(input_dir, file_suffix)

    # 2. Select and Drop Columns
    print("Cleaning and selecting columns...")
    cols_to_drop = [col for col in df_main.columns if 'slope' in col or 'ssh' in col or 'depth' in col]
    # Also drop raw reanalysis columns that have been corrected
    cols_to_drop.extend(['VHM0', 'VTPK', 'VMDR', 'eastward_wind', 'northward_wind', 'uo', 'vo'])
    df_main.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # 3. Integrate New Bathymetry and Slopes
    grid_points = df_main[['lat', 'lon']].drop_duplicates()
    df_bathy = process_bathymetry(GEBCO_PATH, grid_points)
    df_main = df_main.merge(df_bathy, on=['lat', 'lon'], how='left')

    # 4. Integrate New SSH
    df_main['ssh_new'] = process_ssh(HYCOM_PATH, df_main[['time', 'lat', 'lon']])
    
# 5. Final NaN check and fill (safeguard)
    if df_main.isnull().values.any():
        print("Warning: NaNs detected after merge. Applying robust imputation...")
        df_main.set_index('time', inplace=True)
        
        # Group by each unique spatial point
        grouped = df_main.groupby(['lat', 'lon'])
        
        # --- Impute time-varying columns (like ssh) using time interpolation ---
        time_varying_cols = ['ssh_new']
        for col in time_varying_cols:
            if col in df_main.columns:
                df_main[col] = grouped[col].transform(
                    lambda x: x.interpolate(method='time').bfill().ffill()
                )

        # --- Impute static columns (like slope) using forward/backward fill ---
        static_cols = ['slope_ns_new', 'slope_ew_new', 'depth_new']
        for col in static_cols:
            if col in df_main.columns:
                # Use ffill and bfill to propagate the single valid value across the group
                df_main[col] = grouped[col].transform(lambda x: x.ffill().bfill())
        
        df_main.reset_index(inplace=True)
        print("NaN values filled.")

    # 6. Save Enriched Data to Yearly CSVs
    print("Saving enriched data to yearly CSV files...")
    df_main.set_index('time', inplace=True)
    for year in range(2018, 2025):
        yearly_data = df_main[df_main.index.year == year]
        if not yearly_data.empty:
            output_file = output_dir / f'{year}_enriched.csv'
            yearly_data.to_csv(output_file)
            print(f"  -> Saved: {output_file}")


if __name__ == '__main__':
    enrich_dataset(QDM_INPUT_DIR, QDM_OUTPUT_DIR, 'qdm')
    enrich_dataset(CATBOOST_INPUT_DIR, CATBOOST_OUTPUT_DIR, 'catboost')
    print("\nAll datasets have been successfully enriched.")