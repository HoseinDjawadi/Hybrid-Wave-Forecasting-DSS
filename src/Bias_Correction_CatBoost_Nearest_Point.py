import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import optuna
from sklearn.model_selection import TimeSeriesSplit

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================

BASE_DIR = Path('/home/artorias/Desktop/Paper 3_FInal/Data')
REANALYSIS_INPUT_DIR = BASE_DIR / 'Reanalysis/Merged_Data'
BUOY_INPUT_DIR = BASE_DIR / 'Sohar_Buoy'
OUTPUT_DIR = BASE_DIR / 'Bias_Correction'
CATBOOST_OUTPUT_DIR = OUTPUT_DIR / 'catboost_corrected'
FIGURES_OUTPUT_DIR = CATBOOST_OUTPUT_DIR / 'evaluation_figures'

CATBOOST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Coordinates of the reference buoy (Buoy 1)
# Please update with your actual buoy coordinates
BUOY_LAT = 24.535 
BUOY_LON = 56.629

CALIBRATION_START = '2018-05-25' # Adjusted to reflect new understanding of data start
CALIBRATION_END = '2021-12-31'
OPTUNA_TRIALS = 50
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 2. HELPER & ML FUNCTIONS
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on Earth."""
    R = 6371  # Earth radius in kilometers
    dLat = np.deg2rad(lat2 - lat1)
    dLon = np.deg2rad(lon2 - lon1)
    a = np.sin(dLat / 2)**2 + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dLon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def select_nearest_point_data(df_reanalysis: pd.DataFrame, buoy_lat: float, buoy_lon: float) -> pd.DataFrame:
    """Selects the time series from the single reanalysis grid point closest to the given buoy coordinates."""
    print("Spatio-temporal data detected. Selecting nearest reanalysis point...")
    coords = df_reanalysis[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
    coords['distance_km'] = haversine_distance(buoy_lat, buoy_lon, coords['lat'], coords['lon'])
    nearest_coord = coords.loc[coords['distance_km'].idxmin()]
    
    print(f"Buoy Location: ({buoy_lat}, {buoy_lon})")
    print(f"Nearest Reanalysis Point Found: ({nearest_coord['lat']:.4f}, {nearest_coord['lon']:.4f}) at {nearest_coord['distance_km']:.2f} km.")
    
    df_nearest = df_reanalysis[(df_reanalysis['lat'] == nearest_coord['lat']) & (df_reanalysis['lon'] == nearest_coord['lon'])].copy()
    
    if df_nearest.index.has_duplicates:
        df_nearest = df_nearest[~df_nearest.index.duplicated(keep='first')]
    return df_nearest

def load_and_combine_data(data_dir: Path, start_year: int, end_year: int) -> pd.DataFrame:
    """Loads and combines yearly CSV files."""
    print(f"Loading reanalysis data from {start_year} to {end_year}...")
    all_years_df = []
    for year in range(start_year, end_year + 1):
        file_path = data_dir / f"{year}.csv"
        if file_path.exists():
            df_year = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
            all_years_df.append(df_year)
    if not all_years_df: raise FileNotFoundError("No reanalysis data files were found.")
    return pd.concat(all_years_df)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers features from the reanalysis data for the ML model."""
    print("Engineering features...")
    features = df.copy()
    features['month'] = features.index.month
    features['day_of_year'] = features.index.dayofyear
    features['hour'] = features.index.hour
    
    # Create lagged features
    for col in ['VHM0', 'VTPK', 'eastward_wind', 'northward_wind', 'uo', 'vo']:
        for i in range(1, 3):
            features[f'{col}_lag{i}'] = features[col].shift(i)
            
    # --- FIX: Instead of a simple bfill, use a more robust interpolation strategy ---
    # First, interpolate for gaps in the middle of the series.
    # Then, use bfill/ffill to catch any remaining NaNs at the very start/end.
    print("Interpolating to fill gaps created by feature engineering...")
    features.interpolate(method='time', inplace=True)
    features.bfill(inplace=True)
    features.ffill(inplace=True)
    
    return features

def tune_catboost_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Tunes CatBoost hyperparameters using Optuna."""
    X_train_reset = X_train.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)
    def objective(trial):
        params = {
            'objective': 'RMSE', 'iterations': trial.suggest_int('iterations', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 1e-2, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 7),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
            'verbose': 0, 'thread_count': -1
        }
        tscv = TimeSeriesSplit(n_splits=3)
        rmses = []
        for train_idx, val_idx in tscv.split(X_train_reset):
            X_t, X_v = X_train_reset.iloc[train_idx], X_train_reset.iloc[val_idx]
            y_t, y_v = y_train_reset.iloc[train_idx], y_train_reset.iloc[val_idx]
            model = CatBoostRegressor(**params)
            model.fit(X_t, y_t, eval_set=[(X_v, y_v)], early_stopping_rounds=30, verbose=0)
            rmses.append(np.sqrt(np.mean((model.predict(X_v) - y_v)**2)))
        return np.mean(rmses)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    print(f"  Best trial RMSE from tuning: {study.best_value:.4f}")
    return study.best_params

def convert_speed_dir_to_uv(df: pd.DataFrame, speed_col: str, dir_col: str):
    """Converts speed and direction (degrees) to U and V vector components."""
    direction_rad = np.deg2rad(df[dir_col])
    u = df[speed_col] * np.sin(direction_rad)
    v = df[speed_col] * np.cos(direction_rad)
    return u, v

def convert_uv_to_speed_dir(u: pd.Series, v: pd.Series):
    """Converts U and V vector components to speed and direction (degrees)."""
    speed = np.sqrt(u**2 + v**2)
    direction_deg = np.rad2deg(np.arctan2(u, v))
    return speed, (direction_deg + 360) % 360

def generate_evaluation_plots(raw: pd.Series, corrected: pd.Series, obs: pd.Series, var_name: str, file_path: Path):
    """Generates and saves a 3-panel evaluation plot."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f"Bias Correction Evaluation for: {var_name}", fontsize=16)
    sample_period = obs.index[-1000:]
    axes[0].plot(obs.loc[sample_period], label='Buoy (Observed)', color='k', linestyle='--')
    axes[0].plot(raw.loc[sample_period], label='Reanalysis (Raw)', color='r', alpha=0.7)
    axes[0].plot(corrected.loc[sample_period], label='CatBoost Corrected', color='b', alpha=0.8)
    axes[0].set_title('Time Series Comparison (Sample)')
    axes[0].set_ylabel(var_name)
    axes[0].legend()
    axes[0].grid(True, linestyle=':')
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
    df_plot = pd.DataFrame({'raw': raw, 'corrected': corrected, 'obs': obs}).dropna()
    quantiles = np.linspace(0.01, 0.99, 100)
    q_obs = df_plot['obs'].quantile(quantiles)
    q_raw = df_plot['raw'].quantile(quantiles)
    q_corr = df_plot['corrected'].quantile(quantiles)
    axes[1].plot(q_obs, q_obs, 'k--', label='1:1 Line')
    axes[1].scatter(q_obs, q_raw, color='r', alpha=0.7, label='Raw vs. Obs')
    axes[1].scatter(q_obs, q_corr, color='b', alpha=0.7, label='Corrected vs. Obs')
    axes[1].set_title('Quantile-Quantile Plot')
    axes[1].set_xlabel('Observed Quantiles')
    axes[1].set_ylabel('Modeled Quantiles')
    axes[1].legend()
    axes[1].grid(True, linestyle=':')
    axes[1].axis('equal')
    df_sample = df_plot.sample(n=min(5000, len(df_plot)), random_state=1)
    axes[2].plot(df_sample['obs'], df_sample['obs'], 'k--', label='1:1 Line')
    axes[2].scatter(df_sample['obs'], df_sample['raw'], color='r', alpha=0.5, label='Raw vs. Obs')
    axes[2].scatter(df_sample['obs'], df_sample['corrected'], color='b', alpha=0.5, label='Corrected vs. Obs')
    axes[2].set_title('Scatter Plot')
    axes[2].set_xlabel('Observed')
    axes[2].set_ylabel('Modeled')
    axes[2].legend()
    axes[2].grid(True, linestyle=':')
    axes[2].axis('equal')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(file_path)
    plt.close()
    print(f"Evaluation plot saved for {var_name}.")

# =============================================================================
# 3. MAIN EXECUTION SCRIPT
# =============================================================================

def main():
    """Main function to run the robust CatBoost bias correction workflow."""
    
    df_reanalysis_raw = load_and_combine_data(REANALYSIS_INPUT_DIR, 2018, 2024)
    buoy_file = BUOY_INPUT_DIR / 'buoy_072_3hourly.csv'
    df_buoy = pd.read_csv(buoy_file, parse_dates=['time'], index_col='time')

    df_reanalysis = select_nearest_point_data(df_reanalysis_raw, BUOY_LAT, BUOY_LON)
    
    print("Performing temporal harmonization...")
    df_reanalysis = df_reanalysis[df_reanalysis.index.isin(df_buoy.index)]
    df_buoy = df_buoy[df_buoy.index.isin(df_reanalysis.index)]
    
    X = create_features(df_reanalysis)
    
    df_buoy['u_wind'], df_buoy['v_wind'] = convert_speed_dir_to_uv(df_buoy, 'WindSpeed', 'WindDirection')
    df_buoy['u_curr'], df_buoy['v_curr'] = convert_speed_dir_to_uv(df_buoy, 'CurrSpd', 'CurrDir')
    mdir_rad = np.deg2rad(df_buoy['Mdir'])
    df_buoy['u_mdir'] = np.sin(mdir_rad)
    df_buoy['v_mdir'] = np.cos(mdir_rad)

    target_cols = ['Hm0', 'Tp', 'u_wind', 'v_wind', 'u_curr', 'v_curr', 'u_mdir', 'v_mdir']
    
    # --- FIX: Replace aggressive .dropna() with robust interpolation ---
    combined_df = X.join(df_buoy[target_cols])
    
    # Check for NaNs before filling, for confirmation
    if combined_df.isnull().values.any():
        print(f"Found {combined_df.isnull().values.sum()} total NaN values before final fill. Interpolating...")
        # Use time-based interpolation first, then fill any remaining edge cases
        combined_df.interpolate(method='time', inplace=True)
        combined_df.bfill(inplace=True)
        combined_df.ffill(inplace=True)
        print("NaN values filled.")

    X_aligned = combined_df[X.columns]
    Y_aligned = combined_df[target_cols]
    
    X_train = X_aligned.loc[CALIBRATION_START:CALIBRATION_END]
    Y_train = Y_aligned.loc[CALIBRATION_START:CALIBRATION_END]

    if X_train.empty:
        print("\nCRITICAL ERROR: Training dataset is still empty. Check your CALIBRATION_START date against available data.")
        return
    
    print("\n--- Tuning hyperparameters on primary target (Hm0) ---")
    best_params = tune_catboost_hyperparameters(X_train, Y_train['Hm0'])
    print(f"  Best params found: {best_params}")

    df_corrected = df_reanalysis.copy()
    for target_col in target_cols:
        print(f"\n--- Training final model for '{target_col}' ---")
        model = CatBoostRegressor(**best_params, verbose=100)
        model.fit(X_train, Y_train[target_col])
        
        print(f"  Predicting on full dataset for '{target_col}'...")
        predictions = model.predict(X_aligned)
        df_corrected[f'{target_col}_catboost'] = pd.Series(predictions, index=X_aligned.index)

    # --- CORRECTION: Full post-processing and evaluation ---
    print("\nRecombining all vector components...")
    df_corrected['WindSpeed_catboost'], df_corrected['WindDirection_catboost'] = convert_uv_to_speed_dir(df_corrected['u_wind_catboost'], df_corrected['v_wind_catboost'])
    df_corrected['CurrSpd_catboost'], df_corrected['CurrDir_catboost'] = convert_uv_to_speed_dir(df_corrected['u_curr_catboost'], df_corrected['v_curr_catboost'])
    df_corrected['VMDR_catboost'] = convert_uv_to_speed_dir(df_corrected['u_mdir_catboost'], df_corrected['v_mdir_catboost'])[1] # We only need direction
    
    evaluation_metrics = []
    EVAL_MAP = {
        'VHM0': 'Hm0', 'VTPK': 'Tp', 'VMDR': 'Mdir',
        'eastward_wind': 'WindSpeed', # Note: Evaluating against original speed/dir
        'uo': 'CurrSpd'
    }

    # Map corrected columns to raw columns for evaluation
    CORR_COL_MAP = {
        'VHM0': 'Hm0_catboost', 'VTPK': 'Tp_catboost', 'VMDR': 'VMDR_catboost',
        'eastward_wind': 'WindSpeed_catboost', 'uo': 'CurrSpd_catboost'
    }

    for mod_col, obs_col in EVAL_MAP.items():
        raw_series = df_reanalysis[mod_col] if 'wind' not in mod_col and 'uo' not in mod_col else np.sqrt(df_reanalysis['eastward_wind']**2 + df_reanalysis['northward_wind']**2) if 'wind' in mod_col else np.sqrt(df_reanalysis['uo']**2 + df_reanalysis['vo']**2)
        corr_series = df_corrected[CORR_COL_MAP[mod_col]]
        obs_series = df_buoy[obs_col]

        generate_evaluation_plots(raw_series, corr_series, obs_series, mod_col, FIGURES_OUTPUT_DIR / f'CatBoost_Evaluation_{mod_col}.png')
        
        eval_df = pd.DataFrame({'raw': raw_series, 'corr': corr_series, 'obs': obs_series}).dropna()
        rmse_raw = np.sqrt(np.mean((eval_df['raw'] - eval_df['obs'])**2))
        rmse_corr = np.sqrt(np.mean((eval_df['corr'] - eval_df['obs'])**2))
        mae_raw = np.mean(np.abs(eval_df['raw'] - eval_df['obs']))
        mae_corr = np.mean(np.abs(eval_df['corr'] - eval_df['obs']))
        
        evaluation_metrics.append({'Variable': mod_col, 'RMSE_Raw': rmse_raw, 'RMSE_Corrected': rmse_corr, 'MAE_Raw': mae_raw, 'MAE_Corrected': mae_corr})

    df_metrics = pd.DataFrame(evaluation_metrics)
    metrics_path = CATBOOST_OUTPUT_DIR / 'evaluation_metrics_catboost.csv'
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nEvaluation metrics saved to: {metrics_path}\n{df_metrics}")

    print("\nSaving bias-corrected data to yearly CSV files...")
    df_to_save = df_corrected.copy().join(df_buoy, rsuffix='_buoy')
    for year in range(2018, 2025):
        yearly_data = df_to_save[df_to_save.index.year == year]
        if not yearly_data.empty:
            output_file = CATBOOST_OUTPUT_DIR / f'{year}_corrected_catboost.csv'
            yearly_data.to_csv(output_file)
            print(f"Saved: {output_file}")
            
    print("\nCatBoost bias correction process complete.")


if __name__ == '__main__':
    main()