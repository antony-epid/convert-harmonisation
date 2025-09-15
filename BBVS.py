This code appears to be analyzing biobank validation study (BBVS) data with accelerometer measurements. Here's the Python equivalent:BBVS Data Analysis - Stata to Python ConversionCode import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def rolling_stats(df, var, time_col, window=5):
    """Calculate rolling statistics I've converted the Stata code to Python. Here are the key features of the conversion:
Main Components:

Data Processing Functions:

process_participant_data(): Handles individual CSV file processing
rolling_stats(): Calculates rolling window statistics (equivalent to rangestat)
calculate_baseline(): Computes daily baselines from the first hour of data
analyze_variable_pairs(): Performs pairwise analysis between variables


Key Processing Steps:

Creates total body movement variable as weighted combination
Consolidates PAEE variables
Recodes sex variable (1→0, 0→1)
Calculates BMI
Creates validity flags for each measurement
Converts datetime and creates time variables
Calculates rolling statistics with 5-minute windows
Computes baseline values and creates denoised variables
Performs binning analysis (0-11 bins) with deciles


Statistical Analysis:

Calculates correlations between variable pairs
Creates bins based on 99th percentile cutoffs
Computes bin statistics (means, fits, standard errors)
Filters results to specific variable combinations



Key Differences from Stata:

Uses pandas for data manipulation instead of Stata's data management
Implements rolling statistics manually since pandas doesn't have exact rangestat equivalent
Uses numpy and scipy for statistical calculations
File I/O handled with pandas read_csv/to_csv
Error handling added for robustness

Usage:

Update the file paths at the top of the main() function
Ensure your CSV files have the expected column structure
Run the script to process all CSV files in the specified directory
Results will be saved to the specified output path

The code maintains the same analytical logic as the original Stata script while leveraging Python's data science ecosystem for improved performance and flexibility.

#==========================
import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def rolling_stats(df, var, time_col, window=5):
    """Calculate rolling statistics for a given variable"""
    half_lo = (window // 2)
    half_hi = window // 2
    
    # Sort by time
    df = df.sort_values(time_col).copy()
    
    # Calculate rolling statistics
    rolling_data = df[var].rolling(window=window, center=True, min_periods=window)
    
    return {
        f'mean_{window}_{var}': rolling_data.mean(),
        f'sd_{window}_{var}': rolling_data.std(),
        f'n_{window}_{var}': rolling_data.count()
    }

def calculate_baseline(df, var, date_col, window=5):
    """Calculate daily baseline (first hour mean)"""
    baselines = []
    
    for date in df[date_col].unique():
        day_data = df[df[date_col] == date].copy()
        day_data = day_data.sort_values(f'sd_{window}_{var}')
        
        # Get first hour (60 minutes / window size)
        first_hour_points = 60 // window
        if len(day_data) >= first_hour_points:
            baseline = day_data[f'mean_{window}_{var}'].iloc[:first_hour_points].mean()
        else:
            baseline = np.nan
            
        baselines.append({'date': date, f'baseline_{window}_{var}': baseline})
    
    baseline_df = pd.DataFrame(baselines)
    return baseline_df

def process_participant_data(file_path):
    """Process individual participant CSV file"""
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Create total body movement variable
    df['total_body_movement'] = (0.5 * df['enmo_mean_thigh'] + 
                                0.25 * df['enmo_mean_ndw'] + 
                                0.25 * df['enmo_mean_dw'])
    
    # Consolidate PAEE from combined sensing
    df['PAI_use'] = df['PAI_Branch2'].fillna(df['PAI_Branch6'])
    
    # Recode sex (1->0, 0->1)
    df['sex'] = df['sex'].replace({1: 0, 0: 1})
    
    # Calculate BMI
    df['bmi'] = df['weight'] / (df['height'] ** 2)
    
    # Create validity flags for each variable
    validity_mapping = {
        'enmo_mean_dw': 'Pwear_dw',
        'hpfvm_mean_dw': 'Pwear_dw',
        'enmo_mean_ndw': 'Pwear_ndw',
        'hpfvm_mean_ndw': 'Pwear_ndw',
        'enmo_mean_thigh': 'Pwear_thigh',
        'hpfvm_mean_thigh': 'Pwear_thigh',
        'PAI_use': 'PWEAR',
        'ACC': 'PWEAR'
    }
    
    for var, wear_col in validity_mapping.items():
        if var in df.columns and wear_col in df.columns:
            df[f'valid_{var}'] = (df[wear_col] == 1).astype(int)
            df[f'valid_{var}'] = df[f'valid_{var}'].where(df[wear_col].notna(), 0)
    
    # Special case for total_body_movement
    df['valid_total_body_movement'] = 0
    valid_all = ((df['Pwear_thigh'] == 1) & 
                 (df['Pwear_ndw'] == 1) & 
                 (df['Pwear_dw'] == 1))
    df.loc[valid_all, 'valid_total_body_movement'] = 1
    
    # Convert datetime
    df['datetime_var'] = pd.to_datetime(df['DATETIME'], format='%d/%m/%Y %H:%M:%S')
    df['time_minutes'] = ((df['datetime_var'] - df['datetime_var'].iloc[0]).dt.total_seconds() / 60).astype(int)
    df['date_only'] = df['datetime_var'].dt.date
    
    # Sort by datetime
    df = df.sort_values(['id', 'datetime_var']).reset_index(drop=True)
    
    # Variables to process
    variables = ['enmo_mean_dw', 'enmo_mean_ndw', 'enmo_mean_thigh', 
                'hpfvm_mean_dw', 'hpfvm_mean_ndw', 'hpfvm_mean_thigh',
                'total_body_movement', 'PAI_use', 'ACC']
    
    # Process each variable
    for var in variables:
        if var not in df.columns or f'valid_{var}' not in df.columns:
            continue
            
        valid_count = df[f'valid_{var}'].sum()
        
        if valid_count >= 1440:  # At least 1 day of valid data
            # Calculate rolling statistics
            rolling_results = rolling_stats(df, var, 'time_minutes', window=5)
            for col, values in rolling_results.items():
                df[col] = values
            
            # Calculate baseline
            baseline_df = calculate_baseline(df, var, 'date_only', window=5)
            df = df.merge(baseline_df, left_on='date_only', right_on='date', how='left')
            
            # Get median baseline
            baseline_median = df[f'baseline_5_{var}'].median()
            df[f'baseline_{var}'] = baseline_median
            
            # Create denoised variable
            df[f'{var}_new'] = df[var] - baseline_median
            df[f'{var}_new'] = df[f'{var}_new'].clip(lower=0)
        else:
            df[f'baseline_{var}'] = np.nan
            df[f'{var}_new'] = np.nan
    
    return df

def analyze_variable_pairs(df, var_a, var_b, participant_id):
    """Analyze relationship between two variables"""
    
    # Filter valid data
    valid_mask = ((df[f'valid_{var_a}'] == 1) & 
                  (df[f'valid_{var_b}'] == 1) &
                  (df[var_a].notna()) & 
                  (df[var_b].notna()))
    
    if valid_mask.sum() == 0:
        return None
    
    analysis_df = df[valid_mask].copy()
    
    # Get participant characteristics
    participant_stats = {
        'id': participant_id,
        'sex': analysis_df['sex'].iloc[0],
        'age': analysis_df['age'].iloc[0],
        'bmi': analysis_df['bmi'].iloc[0],
        'N': len(analysis_df)
    }
    
    # Original variable statistics
    outcome_stats = {
        'outcome': var_a,
        'outcome_mean': analysis_df[var_a].mean(),
        'outcome_sd': analysis_df[var_a].std(),
        'outcome_baseline': analysis_df[f'baseline_{var_a}'].iloc[0] if f'baseline_{var_a}' in analysis_df.columns else np.nan
    }
    
    predictor_stats = {
        'predictor': var_b,
        'predictor_mean': analysis_df[var_b].mean(),
        'predictor_sd': analysis_df[var_b].std(),
        'predictor_baseline': analysis_df[f'baseline_{var_b}'].iloc[0] if f'baseline_{var_b}' in analysis_df.columns else np.nan
    }
    
    # Check if we have enough denoised data
    if f'{var_b}_new' not in analysis_df.columns or f'{var_a}_new' not in analysis_df.columns:
        return None
        
    valid_denoised = (analysis_df[f'{var_b}_new'] > 0).sum()
    
    if valid_denoised < 1440:  # Not enough data
        return None
    
    # Denoised variable statistics
    outcome_denoised = {
        'outcome_mean_denoised': analysis_df[f'{var_a}_new'].mean(),
        'outcome_sd_denoised': analysis_df[f'{var_a}_new'].std(),
        'outcome_p99_5': analysis_df[f'{var_a}_new'].quantile(0.995)
    }
    
    predictor_denoised = {
        'predictor_mean_denoised': analysis_df[f'{var_b}_new'].mean(),
        'predictor_sd_denoised': analysis_df[f'{var_b}_new'].std(),
        'predictor_p99_5': analysis_df[f'{var_b}_new'].quantile(0.995)
    }
    
    # Calculate correlation
    correlation = analysis_df[f'{var_a}_new'].corr(analysis_df[f'{var_b}_new'])
    
    # Create bins
    positive_data = analysis_df[analysis_df[f'{var_b}_new'] > 0][f'{var_b}_new']
    if len(positive_data) == 0:
        return None
        
    p99 = positive_data.quantile(0.99)
    
    # Initialize bins
    analysis_df['x_bin'] = np.nan
    
    # Bin 0: values <= 0
    analysis_df.loc[analysis_df[f'{var_b}_new'] <= 0, 'x_bin'] = 0
    
    # Bin 11: values > 99th percentile
    analysis_df.loc[analysis_df[f'{var_b}_new'] > p99, 'x_bin'] = 11
    
    # Bins 1-10: deciles of positive values <= 99th percentile
    positive_mask = (analysis_df[f'{var_b}_new'] > 0) & (analysis_df[f'{var_b}_new'] <= p99)
    if positive_mask.sum() > 0:
        analysis_df.loc[positive_mask, 'x_bin'] = pd.qcut(
            analysis_df.loc[positive_mask, f'{var_b}_new'], 
            q=10, labels=False
        ) + 1
    
    # Calculate bin statistics
    bin_stats = {}
    for bin_num in range(12):  # 0-11
        bin_data = analysis_df[analysis_df['x_bin'] == bin_num]
        if len(bin_data) > 0:
            bin_stats[f'bin_mean_{bin_num}'] = bin_data[f'{var_b}_new'].mean()
            bin_stats[f'bin_fit_{bin_num}'] = bin_data[f'{var_a}_new'].mean()
            bin_stats[f'bin_fit_se_{bin_num}'] = bin_data[f'{var_a}_new'].std() / np.sqrt(len(bin_data))
        else:
            bin_stats[f'bin_mean_{bin_num}'] = np.nan
            bin_stats[f'bin_fit_{bin_num}'] = np.nan
            bin_stats[f'bin_fit_se_{bin_num}'] = np.nan
    
    # Combine all results
    result = {**participant_stats, **outcome_stats, **predictor_stats, 
              **outcome_denoised, **predictor_denoised, **bin_stats, 
              'pearson': correlation}
    
    return result

def main():
    """Main processing function"""
    
    # Set file paths
    files_path = "Q:/Data/DATA BACKUP/Biobank_Validation/BBVS_MAINSTUDY/Merge_AH_AX3/1m/"
    output_path = "V:/P5_PhysAct/Projects/Fenland smartphone/Fenland_data/PHFENLANDR900002352022_16Feb2022_MatthewPearce/HumaStepData_2/participant_models/BINS/og/BBVS.csv"
    
    # Get list of CSV files
    csv_files = glob.glob(os.path.join(files_path, "*.csv"))
    
    all_results = []
    
    # Process each file
    for file_path in csv_files:
        print(f"Processing {os.path.basename(file_path)}")
        
        try:
            # Process participant data
            df = process_participant_data(file_path)
            
            if df is None or len(df) == 0:
                continue
                
            participant_id = df['id'].iloc[0] if 'id' in df.columns else os.path.basename(file_path).split('.')[0]
            
            # Define variable pairs to analyze
            variables = ['enmo_mean_dw', 'enmo_mean_ndw', 'enmo_mean_thigh', 
                        'hpfvm_mean_dw', 'hpfvm_mean_ndw', 'hpfvm_mean_thigh',
                        'total_body_movement', 'PAI_use', 'ACC']
            
            # Analyze all variable pairs
            for var_a in variables:
                for var_b in variables:
                    if var_a != var_b and var_a in df.columns and var_b in df.columns:
                        result = analyze_variable_pairs(df, var_a, var_b, participant_id)
                        if result is not None:
                            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Add study column
        results_df['study'] = 'BBVS'
        
        # Rename variables to match Stata output
        variable_mapping = {
            'PAI_use': 'AH_PAEE_kJ_min_kg',
            'enmo_mean_dw': 'AX_dw_enmo',
            'enmo_mean_ndw': 'AX_ndw_enmo',
            'enmo_mean_thigh': 'AX_thigh_enmo',
            'hpfvm_mean_dw': 'AX_dw_hpfvm',
            'hpfvm_mean_ndw': 'AX_ndw_hpfvm',
            'hpfvm_mean_thigh': 'AX_thigh_hpfvm',
            'total_body_movement': 'AX_dw_ndw_thigh_enmo'
        }
        
        for old_name, new_name in variable_mapping.items():
            results_df['outcome'] = results_df['outcome'].replace(old_name, new_name)
            results_df['predictor'] = results_df['predictor'].replace(old_name, new_name)
        
        # Filter to keep only specific variable combinations
        valid_combinations = [
            ('AX_dw_enmo', 'AX_ndw_enmo'),
            ('AX_dw_enmo', 'AX_thigh_enmo'),
            ('AX_dw_enmo', 'AX_dw_hpfvm'),
            ('AX_dw_enmo', 'AX_dw_ndw_thigh_enmo'),
            ('AX_dw_enmo', 'AH_PAEE_kJ_min_kg'),
            ('AX_dw_enmo', 'ACC'),
            ('AX_ndw_enmo', 'AX_dw_enmo'),
            ('AX_ndw_enmo', 'AX_thigh_enmo'),
            ('AX_ndw_enmo', 'AX_ndw_hpfvm'),
            ('AX_ndw_enmo', 'AX_dw_ndw_thigh_enmo'),
            ('AX_ndw_enmo', 'AH_PAEE_kJ_min_kg'),
            ('AX_ndw_enmo', 'ACC'),
            ('AX_thigh_enmo', 'AX_dw_enmo'),
            ('AX_thigh_enmo', 'AX_ndw_enmo'),
            ('AX_thigh_enmo', 'AX_thigh_hpfvm'),
            ('AX_thigh_enmo', 'AX_dw_ndw_thigh_enmo'),
            ('AX_thigh_enmo', 'AH_PAEE_kJ_min_kg'),
            ('AX_thigh_enmo', 'ACC'),
            ('AX_dw_ndw_thigh_enmo', 'AX_ndw_enmo'),
            ('AX_dw_ndw_thigh_enmo', 'AX_thigh_enmo'),
            ('AX_dw_ndw_thigh_enmo', 'AX_dw_enmo'),
            ('AX_dw_ndw_thigh_enmo', 'AH_PAEE_kJ_min_kg'),
            ('AX_dw_ndw_thigh_enmo', 'ACC'),
            ('AX_dw_hpfvm', 'AX_dw_enmo'),
            ('AX_ndw_hpfvm', 'AX_ndw_enmo'),
            ('AX_thigh_hpfvm', 'AX_thigh_enmo'),
            ('AH_PAEE_kJ_min_kg', 'AX_dw_enmo'),
            ('AH_PAEE_kJ_min_kg', 'AX_ndw_enmo'),
            ('AH_PAEE_kJ_min_kg', 'AX_dw_ndw_thigh_enmo'),
            ('AH_PAEE_kJ_min_kg', 'ACC'),
            ('AH_PAEE_kJ_min_kg', 'AX_thigh_enmo'),
            ('ACC', 'AX_dw_enmo'),
            ('ACC', 'AX_ndw_enmo'),
            ('ACC', 'AX_dw_ndw_thigh_enmo'),
            ('ACC', 'AH_PAEE_kJ_min_kg'),
            ('ACC', 'AX_thigh_enmo')
        ]
        
        # Filter results
        mask = pd.Series(False, index=results_df.index)
        for outcome, predictor in valid_combinations:
            mask |= ((results_df['outcome'] == outcome) & (results_df['predictor'] == predictor))
        
        results_df = results_df[mask]
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        print(f"Total rows: {len(results_df)}")
        
    else:
        print("No results generated")

if __name__ == "__main__":
    main()
