import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
from pathlib import Path

def create_yearly_csv_from_sources(wave_files, wind_file, current_file, output_dir):
    """
    Implements a robust, intersection-based regridding workflow. This final
    version resamples all data to a 3-hourly frequency using the maximum
    value and rounds coordinates after converting to a DataFrame.

    Args:
        wave_files (list): List of paths to wave NetCDF files.
        wind_file (Path): Path to the wind NetCDF file.
        current_file (Path): Path to the current NetCDF file.
        output_dir (Path): Directory to save the final yearly CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path("./regridding_weights")
    weights_dir.mkdir(exist_ok=True)

    # --- Phases 1-3: Loading, Masking, Regridding, and Merging ---
    # This logic remains the same and is correct.
    print("--- Phase 1: Loading Data and Defining Target Grid ---")
    ds_wave = xr.open_mfdataset(wave_files, combine='by_coords').rename({'longitude': 'lon', 'latitude': 'lat'})
    ds_wind = xr.open_dataset(wind_file).rename({'longitude': 'lon', 'latitude': 'lat'})
    ds_current = xr.open_dataset(current_file).rename({'longitude': 'lon', 'latitude': 'lat'})
    ds_target_grid = ds_wave
    print("✓ All source datasets loaded. Wave grid selected as the target.")

    print("\n--- Phase 2: Creating a Common Ocean Mask on the Target Grid ---")
    mask_wave_native = ds_wave['VHM0'].notnull().any(dim='time')
    mask_wind_native = ds_wind['eastward_wind'].notnull().any(dim='time')
    mask_current_native = ds_current['uo'].notnull().any(dim='time').squeeze(drop=True)
    regridder_mask_wind = xe.Regridder(ds_wind, ds_target_grid, 'nearest_s2d', reuse_weights=False)
    mask_wind_regridded = regridder_mask_wind(mask_wind_native)
    regridder_mask_current = xe.Regridder(ds_current, ds_target_grid, 'nearest_s2d', reuse_weights=False)
    mask_current_regridded = regridder_mask_current(mask_current_native)
    final_mask = mask_wave_native & mask_wind_regridded.astype(bool) & mask_current_regridded.astype(bool)
    print("✓ Common ocean domain mask created by intersecting all masks.")

    print("\n--- Phase 3: Regridding Data and Merging ---")
    regridder_wind = xe.Regridder(ds_wind, ds_target_grid, 'nearest_s2d', filename=str(weights_dir/'wind_to_wave_nn.nc'))
    ds_wind_regridded = regridder_wind(ds_wind)
    regridder_current = xe.Regridder(ds_current, ds_target_grid, 'nearest_s2d', filename=str(weights_dir/'current_to_wave_nn.nc'))
    ds_current_regridded = regridder_current(ds_current).squeeze(dim='depth', drop=True)
    merged_ds = xr.merge([
        ds_wave[['VHM0', 'VTPK', 'VMDR']],
        ds_wind_regridded[['eastward_wind', 'northward_wind']],
        ds_current_regridded[['uo', 'vo']]
    ])
    final_ds_masked = merged_ds.where(final_mask)
    print("✓ All data regridded, merged, and masked.")

    print("\n--- Phase 4: Final Resampling and Export ---")
    
    # Resample time to 3-hourly frequency, taking the max value in each bin
    final_ds_resampled = final_ds_masked.resample(time='3h').max(skipna=True)
    print("✓ Dataset resampled to 3-hourly frequency.")

    # Convert to pandas DataFrame FIRST
    print("Converting to pandas DataFrame... (this is the memory-intensive step)")
    df = final_ds_resampled.to_dataframe()
    df = df.reset_index()
    print("✓ Conversion to DataFrame complete.")
    
    # Drop rows that are fully empty after resampling
    df = df.dropna(subset=['VHM0', 'eastward_wind', 'uo'], how='all')

    # Now, round the coordinates in the DataFrame
    df['lon'] = df['lon'].round(4)
    df['lat'] = df['lat'].round(4)
    print("✓ Geo-coordinates rounded in the final DataFrame.")

    # Loop through each year and save a separate CSV file
    all_years = df['time'].dt.year.unique()
    for year in all_years:
        print(f"Processing and saving data for year: {year}...")
        df_year = df[df['time'].dt.year == year]
        
        output_filename = output_dir / f"sohar_data_{year}.csv"
        df_year.to_csv(output_filename, index=False, date_format='%Y-%m-%dT%H:%M:%S')
        print(f"  ✓ Saved {output_filename}")
        
    print("\nWorkflow complete.")


def main():
    """Main function to run the entire workflow."""
    # --- DEFINE FILE AND FOLDER PATHS ---
    output_dir = Path("/home/artorias/Desktop/Paper 3_FInal/Data/Reanalysis/CMEMS/Processed")
    
    # Paths to your source CMEMS NetCDF files
    wave_file_1 = Path("/home/artorias/Desktop/Paper 3_FInal/Data/Reanalysis/CMEMS/cmems_mod_glo_wav_my_0.2deg_PT3H-i_1752576423537.nc")
    wave_file_2 = Path("/home/artorias/Desktop/Paper 3_FInal/Data/Reanalysis/CMEMS/cmems_mod_glo_wav_myint_0.2deg_PT3H-i_1752576515181.nc")
    wind_file = Path("/home/artorias/Desktop/Paper 3_FInal/Data/Reanalysis/CMEMS/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_1752577055451.nc")
    current_file = Path("/home/artorias/Desktop/Paper 3_FInal/Data/Reanalysis/CMEMS/cmems_obs-mob_glo_phy-cur_my_0.25deg_PT1H-i_1752577660004.nc")
    
    create_yearly_csv_from_sources(
        wave_files=[wave_file_1, wave_file_2],
        wind_file=wind_file,
        current_file=current_file,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
