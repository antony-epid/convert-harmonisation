This code processes accelerometer data from the EPIC-Norfolk study and performs similar analysis to the previous BBVS code.EPIC-Norfolk Data Analysis - Stata to Python ConversionCode import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def rolling_stats(df, var, time_col, window=5):
    """Calculate rolling I've converted the EPIC-Norfolk Stata code to Python. Here are the key features of this conversion:
Main Components:

File Processing Functions:

process_epic_file(): Combines raw accelerometer data with counts data
process_participant_data(): Processes individual participant data with baseline calculations
analyze_variable_pairs(): Performs pairwise analysis between variables


Key Processing Steps:

Reads both raw (.dta) files and counts files
Merges raw accelerometer data with step counts and axis data
Creates validity flags based on wear time (Pwear_AG)
Processes variables: enmo_mean_AG, Steps, Vector_Magnitude, Axis1, Axis2, Axis3
Calculates rolling statistics with 5-minute windows
Computes baseline values and creates denoised variables
Performs binning analysis (0-11 bins) with deciles


Data Integration:

Merges results with EPIC demographic information
Handles missing demographic data (sex, age, BMI initially set to NaN)
Removes actiband variables and self-comparisons
Adds study identifier as "EPIC-Norfolk"



Key Differences from Original Stata Code:

File Handling: Uses pandas to read .dta files instead of Stata's native format
Temporary Files: Eliminates need for temporary file creation by keeping data in memory
Data Merging: Uses pandas merge operations instead of Stata's merge commands
Statistical Calculations: Uses pandas/numpy for statistical operations
Error Handling: Includes comprehensive error handling for file operations

Special Features:

Filename Processing: Extracts participant IDs and matches raw files with counts files
Demographic Merging: Attempts to merge with EPIC_info.dta for participant characteristics
Quality Checks: Includes monotonicity checking for bin patterns
Flexible Path Handling: Can handle missing files gracefully

Usage:

Update the file paths in the main() function to match your directory structure
Ensure you have:

Raw .dta files in the specified raw files directory
Corresponding counts files in the counts directory
EPIC_info.dta file for demographic data


Run the script to process all files
Results will be saved as CSV with the same structure as the original Stata output

The code maintains the same analytical logic as the original Stata script while providing better error handling and memory efficiency through Python's data science ecosystem.


#===========================
import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def rolling_stats(df, var, time_col, window=5):
    """Calculate rolling statistics for a given variable"""
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
        day_data = day_data.sort_values([f'sd_{window}_{var}', var])
        
        # Get first hour (60 minutes / window size)
        first_hour_points = 60 // window
        if len(day_data) >= first_hour_points:
            baseline = day_data[f'mean_{window}_{var}'].iloc[:first_hour_points].mean()
            day_count = day_data[f'valid_{var}'].sum()
            if day_count < 60:
                baseline = np.nan
        else:
            baseline = np.nan
            
        baselines.append({'date': date, f'baseline_{window}_{var}': baseline})
    
    baseline_df = pd.DataFrame(baselines)
    return baseline_df

def process_epic_file(raw_file_path, counts_file_path):
    """Process individual EPIC participant files"""
    
    try:
        # Read raw data file
        raw_df = pd.read_stata(raw_file_path)
        
        # Keep only required columns and rename
        raw_df = raw_df[['id', 'DATETIME', 'enmo_mean_AG', 'enmo_missing_AG', 'Pwear_AG']].copy()
        raw_df['datetime'] = raw_df['DATETIME']
        
        # Extract first 8 characters of ID
        raw_df['id'] = raw_df['id'].astype(str).str[:8]
        
        # Read counts data file
        counts_df = pd.read_stata(counts_file_path)
        counts_df = counts_df[['Steps', 'Vector_Magnitude', 'datetime', 'Axis1', 'Axis2', 'Axis3']].copy()
        
        # Merge the datasets
        df = pd.merge(raw_df, counts_df, on='datetime', how='inner')
        
        # Keep only required columns
        df = df[['id', 'datetime', 'enmo_mean_AG', 'enmo_missing_AG', 'Pwear_AG', 
                'Steps', 'Vector_Magnitude', 'Axis1', 'Axis2', 'Axis3']].copy()
        
        return df
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return None

def process_participant_data(df):
    """Process individual participant data"""
    
    # Create validity flags for each variable
    variables = ['enmo_mean_AG', 'Steps', 'Vector_Magnitude', 'Axis1', 'Axis2', 'Axis3']
    
    for var in variables:
        if var in df.columns:
            df[f'valid_{var}'] = 0
            df.loc[df['Pwear_AG'] == 1, f'valid_{var}'] = 1
            df.loc[df['Pwear_AG'].isna(), f'valid_{var}'] = 0
    
    # Convert datetime
    df['datetime_var'] = pd.to_datetime(df['datetime'])
    
    # Calculate time variables
    df = df.sort_values(['id', 'datetime']).reset_index(drop=True)
    start_time = df['datetime_var'].iloc[0]
    df['time_minutes'] = ((df['datetime_var'] - start_time).dt.total_seconds() / 60).astype(int)
    df['date_only'] = df['datetime_var'].dt.date
    
    # Process each variable
    for var in variables:
        if var not in df.columns:
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
    
    # Get participant characteristics (no sex, age, bmi in EPIC raw data)
    participant_stats = {
        'id': participant_id,
        'sex': np.nan,  # Will be filled from EPIC_info.dta later
        'age': np.nan,  # Will be filled from EPIC_info.dta later
        'bmi': np.nan,  # Will be filled from EPIC_info.dta later
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
    raw_files_path = "J:/P5_Activity/MP_TEMP/1m"
    counts_files_path = "J:/P5_Activity/MP_TEMP/DTA_FOLDER_4HC_60s"
    temp_path = "V:/P5_PhysAct/Projects/Fenland smartphone/temp"
    output_path = "V:/P5_PhysAct/Projects/Fenland smartphone/Fenland_data/PHFENLANDR900002352022_16Feb2022_MatthewPearce/HumaStepData_2/participant_models/BINS/og/EPIC.csv"
    epic_info_path = "V:/P5_PhysAct/People/Matt Pearce/Epic_transfer/EPIC_info.dta"
    
    # Get list of .dta files
    dta_files = glob.glob(os.path.join(raw_files_path, "*.dta"))
    
    all_results = []
    
    # Process each file
    for raw_file_path in dta_files:
        filename_base = os.path.basename(raw_file_path)
        filename_without_ext = os.path.splitext(filename_base)[0]
        
        # Create filename for counts file (assuming naming pattern from Stata code)
        filename_clean = filename_without_ext.replace('_', '')[:21]
        counts_file_path = os.path.join(counts_files_path, f"{filename_clean}5sec_proc60.dta")
        
        print(f"Processing {filename_base}")
        
        try:
            # Check if counts file exists
            if not os.path.exists(counts_file_path):
                print(f"Counts file not found: {counts_file_path}")
                continue
            
            # Process the files
            df = process_epic_file(raw_file_path, counts_file_path)
            
            if df is None or len(df) == 0:
                continue
            
            # Process participant data
            df = process_participant_data(df)
            
            participant_id = df['id'].iloc[0] if 'id' in df.columns else filename_without_ext
            
            # Define variables to analyze
            variables = ['enmo_mean_AG', 'Steps', 'Vector_Magnitude', 'Axis1', 'Axis2', 'Axis3']
            
            # Analyze all variable pairs
            for var_a in variables:
                for var_b in variables:
                    if var_a != var_b and var_a in df.columns and var_b in df.columns:
                        result = analyze_variable_pairs(df, var_a, var_b, participant_id)
                        if result is not None:
                            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {raw_file_path}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Remove rows where outcome == predictor
        results_df = results_df[results_df['outcome'] != results_df['predictor']]
        
        # Add study column
        results_df['study'] = 'EPIC-Norfolk'
        
        # Remove any actiband_cpm_aligned variables (if they exist)
        results_df = results_df[
            (results_df['predictor'] != 'actiband_cpm_aligned') &
            (results_df['outcome'] != 'actiband_cpm_aligned')
        ]
        
        # Try to merge with EPIC info file for demographic data
        try:
            if os.path.exists(epic_info_path):
                epic_info = pd.read_stata(epic_info_path)
                
                # Drop existing demographic columns
                results_df = results_df.drop(['age', 'bmi', 'sex'], axis=1, errors='ignore')
                
                # Merge with EPIC info
                results_df = pd.merge(results_df, epic_info, on='id', how='inner')
                
                # Rename BMI column if it exists
                if 'BMI' in results_df.columns:
                    results_df = results_df.rename(columns={'BMI': 'bmi'})
                    
                print(f"Successfully merged with EPIC info file")
            else:
                print(f"EPIC info file not found: {epic_info_path}")
                
        except Exception as e:
            print(f"Error merging with EPIC info: {str(e)}")
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        print(f"Total rows: {len(results_df)}")
        
        # Check for monotonic bin patterns (equivalent to the forvalues loop at the end)
        check_bin_monotonicity(results_df)
        
    else:
        print("No results generated")

def check_bin_monotonicity(results_df):
    """Check if bin means are monotonically increasing"""
    monotonicity_violations = 0
    
    for i in range(1, 11):  # bins 1 to 10
        prev_col = f'bin_mean_{i-1}'
        curr_col = f'bin_mean_{i}'
        
        if prev_col in results_df.columns and curr_col in results_df.columns:
            violations = (results_df[curr_col] < results_df[prev_col]).sum()
            monotonicity_violations += violations
            if violations > 0:
                print(f"Bin {i}: {violations} cases where bin_mean_{i} < bin_mean_{i-1}")
    
    print(f"Total monotonicity violations: {monotonicity_violations}")

if __name__ == "__main__":
    main()
