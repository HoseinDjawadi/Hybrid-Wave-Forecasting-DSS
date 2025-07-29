import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.interpolate import griddata # <-- Add this import

def create_final_yearly_csvs(cmems_dir, hycom_path, gebco_path, output_dir):
    """
    Creates final, clean yearly CSV files using a direct nearest-neighbor
    lookup method and robust slope calculation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- Phase 1: Preparing Data and Lookup Structures ---")
    cmems_files = sorted(list(cmems_dir.glob("sohar_data_*.csv")))
    if not cmems_files:
        raise FileNotFoundError(f"No CMEMS CSV files found in {cmems_dir}")
        
    # FIX: Ensure all expected columns are read, including VMDR for wave direction
    df_cmems_full = pd.concat([pd.read_csv(f, parse_dates=['time']) for f in cmems_files], ignore_index=True)

    df_hycom = pd.read_csv(hycom_path, parse_dates=['date']).rename(columns={'date': 'time', 'longitude' : 'lon', 'latitude' : 'lat', 'surface_elevation': 'ssh'})
    ds_gebco = xr.open_dataset(gebco_path)
    print("✓ All source data loaded.")

    hycom_unique_coords = df_hycom[['lon', 'lat']].drop_duplicates()
    hycom_tree = cKDTree(hycom_unique_coords)
    
    gebco_lon, gebco_lat = np.meshgrid(ds_gebco['lon'], ds_gebco['lat'])
    gebco_grid = np.vstack([gebco_lon.ravel(), gebco_lat.ravel()]).T
    gebco_tree = cKDTree(gebco_grid)
    print("✓ Created efficient lookup structures.")

    print("--- Phase 2: Create Coordinate and Data Maps ---")
    target_points_df = df_cmems_full[['lon', 'lat']].drop_duplicates()
    
    _, hycom_indices = hycom_tree.query(target_points_df, k=1)
    _, gebco_indices = gebco_tree.query(target_points_df, k=1)

    map_df = target_points_df.copy()
    map_df[['hycom_lon', 'hycom_lat']] = hycom_unique_coords.iloc[hycom_indices].values
    map_df['depth'] = -ds_gebco['elevation'].values.ravel()[gebco_indices]
    
    # --- START: ROBUST SLOPE CALCULATION (MOVED FROM Post_Bias_Correction_Processing.py) ---
    print("  -> Creating complete surface for robust slope calculation...")
    pivoted_depth = map_df.pivot(index='lat', columns='lon', values='depth')

    x_coords = pivoted_depth.columns.values
    y_coords = pivoted_depth.index.values
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    valid_mask = ~np.isnan(pivoted_depth.values)
    valid_points = np.array([x_grid[valid_mask], y_grid[valid_mask]]).T
    valid_values = pivoted_depth.values[valid_mask]

    interpolated_grid = griddata(valid_points, valid_values, (x_grid, y_grid), method='cubic')
    fallback_grid = griddata(valid_points, valid_values, (x_grid, y_grid), method='linear')
    final_grid_depth = np.where(np.isnan(interpolated_grid), fallback_grid, interpolated_grid)

    grad_ns, grad_ew = np.gradient(final_grid_depth, y_coords, x_coords)
    
    lat_to_idx = {lat: i for i, lat in enumerate(y_coords)}
    lon_to_idx = {lon: i for i, lon in enumerate(x_coords)}
    
    map_df['slope_ns'] = map_df.apply(lambda row: grad_ns[lat_to_idx[row['lat']], lon_to_idx[row['lon']]], axis=1)
    map_df['slope_ew'] = map_df.apply(lambda row: grad_ew[lat_to_idx[row['lat']], lon_to_idx[row['lon']]], axis=1)
    # --- END: ROBUST SLOPE CALCULATION ---
    
    depth_map = map_df.set_index(['lon', 'lat'])['depth'].to_dict()
    slope_ns_map = map_df.set_index(['lon', 'lat'])['slope_ns'].to_dict()
    slope_ew_map = map_df.set_index(['lon', 'lat'])['slope_ew'].to_dict()
    coord_map = map_df.set_index(['lon', 'lat'])[['hycom_lon', 'hycom_lat']].apply(tuple, axis=1).to_dict()
    print("✓ Built coordinate and data mapping tables.")

    print("--- Phase 3: Process Each Year ---")
    all_years = df_cmems_full['time'].dt.year.unique()
    for year in sorted(all_years):
        print(f"\n--- Processing Year: {year} ---")
        df_year = df_cmems_full[df_cmems_full['time'].dt.year == year].copy()
        
        # Add features using the fast maps
        point_tuples = list(zip(df_year['lon'], df_year['lat']))
        df_year['depth'] = point_tuples
        df_year['slope_ns'] = point_tuples
        df_year['slope_ew'] = point_tuples
        
        df_year['depth'] = df_year['depth'].map(depth_map)
        df_year['slope_ns'] = df_year['slope_ns'].map(slope_ns_map)
        df_year['slope_ew'] = df_year['slope_ew'].map(slope_ew_map)

        df_year['ssh'] = np.nan
        for (lon, lat), group in df_year.groupby(['lon', 'lat']):
            hycom_lon, hycom_lat = coord_map[(lon, lat)]
            hycom_series = df_hycom[(df_hycom['lon'] == hycom_lon) & (df_hycom['lat'] == hycom_lat)].set_index('time')['ssh']
            hycom_series_3h = hycom_series.resample('3h').mean() # Use mean for smoother resampling
            group_ssh = hycom_series_3h.reindex(group['time'], method='nearest')
            df_year.loc[group.index, 'ssh'] = group_ssh.values

        print("✓ Mapped SSH, Depth, and Slope data to CMEMS grid.")

        # Final check for NaNs just in case
        if df_year.isnull().values.any():
            print("  -> Warning: NaNs detected. Applying forward/backward fill.")
            df_year.sort_values(by=['lon', 'lat', 'time'], inplace=True)
            df_year.ffill(inplace=True)
            df_year.bfill(inplace=True)

        output_filename = output_dir / f"{year}.csv"
        # FIX: Standardize column names here. Let's assume wave direction is VMDR from source.
        # This makes subsequent scripts easier to manage.
        if 'VMDR' not in df_year.columns:
            print("  -> WARNING: Wave direction 'VMDR' not found in source CMEMS data!")
        df_year.to_csv(output_filename, index=False, date_format='%Y-%m-%d %H:%M:%S', float_format='%.6f')
        print(f"✓ Saved final data to {output_filename}")

    print("\nSUCCESS: All yearly files have been processed and saved correctly.")

def main():
    """Defines paths and runs the data integration workflow."""
    # The yearly CSVs created by the previous, corrected script
    cmems_yearly_dir = Path("/home/artorias/Desktop/Paper 3_FInal/Data/Reanalysis/CMEMS/Processed")
    
    # Your source HYCOM and GEBCO files
    hycom_csv_path = Path("/home/artorias/Desktop/Paper 3_FInal/Data/Reanalysis/HYCOM/HYCOM_SSH_Sohar_2018_2024_Paper3.csv")
    gebco_file_path = Path("/home/artorias/Desktop/Paper 3_FInal/Data/Bathymetry/gebco_2024_n26.0_s23.5_w56.2_e58.8.nc")
    
    # The new output directory for the final, fully integrated yearly files
    final_output_dir = Path("/home/artorias/Desktop/Paper 3_FInal/Latest_Modifications/Integrated_Files")

    create_final_yearly_csvs(
        cmems_dir=cmems_yearly_dir,
        hycom_path=hycom_csv_path,
        gebco_path=gebco_file_path,
        output_dir=final_output_dir
    )

if __name__ == "__main__":
    main()