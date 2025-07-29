import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================

# --- Provide placeholder paths for your data and output directories ---
# It's recommended to use absolute paths to avoid any ambiguity.
BASE_DIR = Path('/home/artorias/Desktop/Paper 3_FInal/Data')
REANALYSIS_INPUT_DIR = BASE_DIR / 'Reanalysis/Merged_Data'
BUOY_INPUT_DIR = BASE_DIR / 'Sohar_Buoy'
OUTPUT_DIR = BASE_DIR / 'Bias_Correction'
QDM_OUTPUT_DIR = OUTPUT_DIR / 'qdm_corrected'
FIGURES_OUTPUT_DIR = QDM_OUTPUT_DIR / 'evaluation_figures'

# --- Create output directories if they don't exist ---
QDM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Define analysis periods ---
CALIBRATION_START = '2018-01-01'
CALIBRATION_END = '2021-12-31'
APPLICATION_START = '2022-01-01'
APPLICATION_END = '2024-12-31'

# --- Define variables for bias correction and their name mapping ---
# Format: { 'reanalysis_col_name': 'buoy_col_name' }
VARIABLES_TO_CORRECT = {
    'VHM0': 'Hm0',
    'VTPK': 'Tp',
    'VMDR': 'Mdir'
    # NOTE: Wind and Current components are handled separately below
    # due to their vector nature (U/V vs. Speed/Direction).
}

# =============================================================================
# 2. UTILITY FUNCTIONS
# =============================================================================

def load_and_combine_data(data_dir: Path, start_year: int, end_year: int) -> pd.DataFrame:
    """Loads and combines yearly CSV files into a single DataFrame."""
    print(f"Loading reanalysis data from {start_year} to {end_year}...")
    all_years_df = []
    for year in range(start_year, end_year + 1):
        file_path = data_dir / f"{year}.csv"
        if file_path.exists():
            df_year = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
            all_years_df.append(df_year)
        else:
            print(f"Warning: File not found for year {year} at {file_path}")
    
    if not all_years_df:
        raise FileNotFoundError("No reanalysis data files were found. Please check the REANALYSIS_INPUT_DIR path.")
        
    combined_df = pd.concat(all_years_df)
    print("Reanalysis data loaded and combined successfully.")
    return combined_df

def convert_speed_dir_to_uv(df: pd.DataFrame, speed_col: str, dir_col: str) -> tuple[pd.Series, pd.Series]:
    """Converts speed and direction (degrees) to U and V vector components."""
    direction_rad = np.deg2rad(df[dir_col])
    u = df[speed_col] * np.sin(direction_rad)
    v = df[speed_col] * np.cos(direction_rad)
    return u, v

def convert_uv_to_speed_dir(u: pd.Series, v: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Converts U and V vector components to speed and direction (degrees)."""
    speed = np.sqrt(u**2 + v**2)
    direction_deg = np.rad2deg(np.arctan2(u, v))
    direction_deg = (direction_deg + 360) % 360  # Ensure direction is 0-360
    return speed, direction_deg

def quantile_delta_mapping(
    obs_hist: pd.Series, mod_hist: pd.Series, mod_fut: pd.Series
) -> pd.Series:
    """
    Performs Quantile Delta Mapping based on Cannon et al., 2015.
    
    Args:
        obs_hist: Observed series during the historical/calibration period.
        mod_hist: Modeled series during the historical/calibration period.
        mod_fut: Modeled series during the future/application period.
        
    Returns:
        A pandas Series with the bias-corrected future data.
    """
    # Use fine-grained quantiles for precise mapping
    quantiles = np.linspace(0.001, 0.999, 1001)
    
    # Calculate quantile values for historical distributions
    mod_hist_q = mod_hist.quantile(quantiles)
    obs_hist_q = obs_hist.quantile(quantiles)

    # Find the quantile of each future value within the historical modeled distribution
    # np.interp requires x-coordinates (quantiles) to be increasing
    mod_fut_quantiles = np.interp(mod_fut, mod_hist_q, quantiles)

    # Find the value in the historical observed distribution for that same quantile
    obs_corrected = np.interp(mod_fut_quantiles, quantiles, obs_hist_q)

    # Find the value in the historical modeled distribution for that same quantile
    mod_proj_q = np.interp(mod_fut_quantiles, quantiles, mod_hist_q)
    
    # Calculate the delta (the model's projected change)
    delta = mod_fut - mod_proj_q

    # Apply the delta to the quantile-mapped observed value to preserve the trend
    corrected_series = pd.Series(obs_corrected + delta, index=mod_fut.index)
    
    return corrected_series
    
def generate_evaluation_plots(
    raw: pd.Series, corrected: pd.Series, obs: pd.Series, var_name: str, file_path: Path
):
    """Generates and saves a 3-panel evaluation plot."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f"Bias Correction Evaluation for: {var_name}", fontsize=16)

    # Panel 1: Time Series (showing a sample period for clarity)
    sample_period = obs.index[-1000:] # Plot last 1000 points
    axes[0].plot(obs.loc[sample_period], label='Buoy (Observed)', color='k', linestyle='--')
    axes[0].plot(raw.loc[sample_period], label='Reanalysis (Raw)', color='r', alpha=0.7)
    axes[0].plot(corrected.loc[sample_period], label='QDM Corrected', color='g', alpha=0.8)
    axes[0].set_title('Time Series Comparison (Sample)')
    axes[0].set_ylabel(var_name)
    axes[0].legend()
    axes[0].grid(True, linestyle=':')
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')

    # --- FIX STARTS HERE ---
    # Create a temporary, clean DataFrame to ensure alignment and handle NaNs
    df_plot = pd.DataFrame({
        'raw': raw,
        'corrected': corrected,
        'obs': obs
    }).dropna()

    # Panel 2: Quantile-Quantile (QQ) Plot
    quantiles = np.linspace(0.01, 0.99, 100)
    q_obs = df_plot['obs'].quantile(quantiles)
    q_raw = df_plot['raw'].quantile(quantiles)
    q_corr = df_plot['corrected'].quantile(quantiles)
    axes[1].plot(q_obs, q_obs, 'k--', label='1:1 Line')
    axes[1].scatter(q_obs, q_raw, color='r', alpha=0.7, label='Raw vs. Obs')
    axes[1].scatter(q_obs, q_corr, color='g', alpha=0.7, label='Corrected vs. Obs')
    axes[1].set_title('Quantile-Quantile Plot')
    axes[1].set_xlabel('Observed Quantiles')
    axes[1].set_ylabel('Modeled Quantiles')
    axes[1].legend()
    axes[1].grid(True, linestyle=':')
    axes[1].axis('equal')

    # Panel 3: Scatter Plot
    # Sample from the cleaned DataFrame to avoid overplotting and ensure alignment
    df_sample = df_plot.sample(n=min(5000, len(df_plot)), random_state=1)
    axes[2].plot(df_sample['obs'], df_sample['obs'], 'k--', label='1:1 Line')
    axes[2].scatter(df_sample['obs'], df_sample['raw'], color='r', alpha=0.5, label='Raw vs. Obs')
    axes[2].scatter(df_sample['obs'], df_sample['corrected'], color='g', alpha=0.5, label='Corrected vs. Obs')
    axes[2].set_title('Scatter Plot')
    axes[2].set_xlabel('Observed')
    axes[2].set_ylabel('Modeled')
    axes[2].legend()
    axes[2].grid(True, linestyle=':')
    axes[2].axis('equal')
    # --- FIX ENDS HERE ---

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(file_path)
    plt.close()
    print(f"Evaluation plot saved to: {file_path}")

# =============================================================================
# 3. MAIN EXECUTION SCRIPT
# =============================================================================

def main():
    """Main function to run the QDM bias correction workflow."""
    
    # --- Load Data ---
    df_reanalysis = load_and_combine_data(REANALYSIS_INPUT_DIR, 2018, 2024)
    buoy_file = BUOY_INPUT_DIR / 'buoy_072_3hourly.csv' # Assuming Buoy 1 is the reference
    df_buoy = pd.read_csv(buoy_file, parse_dates=['time'], index_col='time')
    
    # --- Temporal Harmonization ---
    # Align reanalysis data to the buoy's timestamps
    print("Performing temporal harmonization...")
    original_reanalysis_count = len(df_reanalysis)
    df_reanalysis = df_reanalysis[df_reanalysis.index.isin(df_buoy.index)]
    df_buoy = df_buoy[df_buoy.index.isin(df_reanalysis.index)]
    print(f"Harmonization complete. {original_reanalysis_count - len(df_reanalysis)} rows dropped from reanalysis.")
    
    # --- Prepare Vector Data (Wind & Currents) ---
    print("Preparing vector data (wind and currents)...")
    # Convert buoy speed/direction to U/V for consistent correction
    df_buoy['u_wind'], df_buoy['v_wind'] = convert_speed_dir_to_uv(df_buoy, 'WindSpeed', 'WindDirection')
    df_buoy['u_curr'], df_buoy['v_curr'] = convert_speed_dir_to_uv(df_buoy, 'CurrSpd', 'CurrDir')
    
    # Add vector variables to the correction dictionary
    VECTOR_VARIABLES = {
        'eastward_wind': 'u_wind',
        'northward_wind': 'v_wind',
        'uo': 'u_curr',
        'vo': 'v_curr'
    }
    ALL_VARIABLES = {**VARIABLES_TO_CORRECT, **VECTOR_VARIABLES}
    
    # --- Perform Bias Correction ---
    print("\nStarting QDM bias correction process...")
    df_corrected = df_reanalysis.copy()
    
    # Define calibration and application data slices
    obs_calib = df_buoy.loc[CALIBRATION_START:CALIBRATION_END]
    mod_calib = df_reanalysis.loc[CALIBRATION_START:CALIBRATION_END]
    mod_app = df_reanalysis.loc[APPLICATION_START:APPLICATION_END]
    
    for mod_col, obs_col in ALL_VARIABLES.items():
        print(f"--- Correcting '{mod_col}' against '{obs_col}' ---")
        
        # Combine calibration and application periods for full correction series
        full_mod_series = df_reanalysis[mod_col]
        
        # Split the full series into calibration and application periods for the QDM function
        mod_hist = full_mod_series.loc[CALIBRATION_START:CALIBRATION_END]
        mod_fut = full_mod_series.loc[APPLICATION_START:APPLICATION_END]
        obs_hist = df_buoy[obs_col].loc[CALIBRATION_START:CALIBRATION_END]
        
        # Apply QDM to the application period
        corrected_fut = quantile_delta_mapping(obs_hist, mod_hist, mod_fut)
        
        # For the calibration period, a simple Quantile Mapping is often used
        # Here, we will just keep the original data for simplicity, as the primary
        # goal is correcting the application/future period.
        # A more advanced implementation might correct the calibration period as well.
        corrected_hist = mod_hist # Or apply a simple QM
        
        # Combine into a single corrected series
        df_corrected[f'{mod_col}_qdm'] = pd.concat([corrected_hist, corrected_fut])

    # --- Recombine Vector Components ---
    print("\nRecombining vector components for wind and currents...")
    df_corrected['WindSpeed_qdm'], df_corrected['WindDirection_qdm'] = convert_uv_to_speed_dir(
        df_corrected['eastward_wind_qdm'], df_corrected['northward_wind_qdm']
    )
    df_corrected['CurrSpd_qdm'], df_corrected['CurrDir_qdm'] = convert_uv_to_speed_dir(
        df_corrected['uo_qdm'], df_corrected['vo_qdm']
    )

    # --- Generate Evaluation and Save Results ---
    print("\nGenerating evaluation metrics and plots...")
    evaluation_metrics = []

    # Evaluate scalar variables
    for mod_col, obs_col in VARIABLES_TO_CORRECT.items():
        raw_series = df_reanalysis[mod_col]
        corr_series = df_corrected[f'{mod_col}_qdm']
        obs_series = df_buoy[obs_col]

        generate_evaluation_plots(
            raw_series, corr_series, obs_series, mod_col,
            FIGURES_OUTPUT_DIR / f'QDM_Evaluation_{mod_col}.png'
        )
        
        # Calculate metrics
        rmse_raw = np.sqrt(np.mean((raw_series - obs_series)**2))
        rmse_corr = np.sqrt(np.mean((corr_series - obs_series)**2))
        mae_raw = np.mean(np.abs(raw_series - obs_series))
        mae_corr = np.mean(np.abs(corr_series - obs_series))
        
        evaluation_metrics.append({
            'Variable': mod_col,
            'RMSE_Raw': rmse_raw, 'RMSE_Corrected': rmse_corr,
            'MAE_Raw': mae_raw, 'MAE_Corrected': mae_corr
        })

    # Save metrics report
    df_metrics = pd.DataFrame(evaluation_metrics)
    metrics_path = QDM_OUTPUT_DIR / 'evaluation_metrics.csv'
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nEvaluation metrics saved to: {metrics_path}")
    print(df_metrics)

    # --- Save Corrected Data to Yearly CSVs ---
    print("\nSaving bias-corrected data to yearly CSV files...")
    df_to_save = df_corrected.copy()
    # Add original buoy data for easy comparison in the final files
    df_to_save = df_to_save.join(df_buoy, rsuffix='_buoy')
    
    for year in range(2018, 2025):
        yearly_data = df_to_save[df_to_save.index.year == year]
        if not yearly_data.empty:
            output_file = QDM_OUTPUT_DIR / f'{year}_corrected_qdm.csv'
            yearly_data.to_csv(output_file)
            print(f"Saved: {output_file}")
            
    print("\nQDM bias correction process complete.")


if __name__ == '__main__':
    main()