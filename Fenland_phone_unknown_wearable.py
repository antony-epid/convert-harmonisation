This code processes step data from unknown wearable devices and iPhone steps.Fenland Unknown Wearable Data Analysis - Stata to Python ConversionCode import pandas as pd

#==========================================
import numpy as np
import os
import glob
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def detect_zero_runs(series, max_consecutive_zeros=60):
    """Detect runI've converted the Fenland "unknown wearable" Stata code to Python. Here are the key features of this conversion:
Main Components:

Zero Run Detection:

detect_zero_runs(): Detects consecutive runs of zeros in step data and marks periods >60 minutes as invalid (equivalent to the Stata carryforward logic)


Data Processing Functions:

process_participant_data(): Handles individual participant file processing, BMI merging, and sex recoding
process_variables_baseline(): Processes variables and calculates baselines
analyze_variable_pairs(): Performs pairwise analysis between variables


Key Processing Steps:

Merges participant data with BMI information
Recodes sex variable: "M" → 0, "F" → 1 (matching original Stata logic)
Creates time variables (hour_of_day, minute_of_hour, minute_of_day)
Detects zero runs in step data for non-wear detection
Processes variables: OTHER_DEVICE_STEPS, IPHONE_STEPS
Calculates rolling statistics and baseline values
Creates denoised variables



Special Features:

Sex Variable Recoding:

Converts string values "M" and "F" to numeric 0 and 1
Handles potential missing values gracefully


Step Data Processing:

Both OTHER_DEVICE_STEPS and IPHONE_STEPS are processed identically
Zero-run detection marks consecutive periods of 0 steps ≥61 minutes as invalid
This is crucial for distinguishing true inactivity from device non-wear


Pairwise Analysis:

Only analyzes OTHER_DEVICE_STEPS vs IPHONE_STEPS (and vice versa)
Excludes self-comparisons automatically
Requires minimum 1440 minutes of valid data for analysis



Key Differences from Original Stata Code:

Sex Recoding: Uses pandas string replacement and numeric conversion instead of Stata's destring
Zero Run Detection: Custom algorithm instead of Stata's carryforward
File Handling: Uses pandas for .dta file reading
Time Processing: Uses pandas datetime operations

Data Structure Expected:
The code expects participant files with columns:

id: Participant identifier
hour, minute: Time components
date: Date information
OTHER_DEVICE_STEPS: Step counts from unknown wearable device
IPHONE_STEPS: Step counts from iPhone
sex: Participant sex ("M" or "F")
age: Participant age

Usage:

Update the file paths in the main() function:

files_path: Directory containing participant .dta files
bmi_file_path: Path to BMI data file
output_path: Where to save results


Ensure your data files have the expected structure
Run the script to process all participant files

The code will output a CSV file with the same structure as the original Stata analysis, including:

Participant characteristics (id, sex, age, BMI)
Variable statistics (means, standard deviations, baselines)
Correlation coefficients
Bin analysis results (bins 0-11 with means, fits, and standard errors)

This allows for direct comparison between iPhone step counts and unknown wearable device step counts in the Fenland study population.

#=========================
import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def detect_zero_runs(series, max_consecutive_zeros=60):
    """Detect runs of consecutive zeros and mark valid data"""
    valid_flags = np.ones(len(series), dtype=int)
    
    # Find positions where the value is 0 or NaN
    is_zero = (series == 0) | series.isna()
    
    if not is_zero.any():
        return valid_flags
    
    # Find consecutive runs of zeros
    zero_runs = []
    start_idx = None
    
    for i, is_z in enumerate(is_zero):
        if is_z and start_idx is None:
            start_idx = i
        elif not is_z and start_idx is not None:
            zero_runs.append((start_idx, i - 1))
            start_idx = None
    
    # Handle case where zeros run to the end
    if start_idx is not None:
        zero_runs.append((start_idx, len(is_zero) - 1))
    
    # Mark runs longer than max_consecutive_zeros as invalid
    for start, end in zero_runs:
        run_length = end - start + 1
        if run_length >= max_consecutive_zeros:
            valid_flags[start:end+1] = 0
    
    return valid_flags

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

def process_participant_data(file_path, bmi_file_path):
    """Process individual participant data file"""
    
    try:
        # Read participant data
        df = pd.read_stata(file_path)
        
        # Read BMI data
        bmi_df = pd.read_stata(bmi_file_path)
        
        # Merge with BMI data
        df = pd.merge(df, bmi_df, on='id', how='inner')
        
        # Drop height and weight columns
        df = df.drop(['height', 'weight'], axis=1, errors='ignore')
        
        # Recode sex: "M" -> 0, "F" -> 1
        if 'sex' in df.columns:
            df['sex'] = df['sex'].replace({'M': '0', 'F': '1'})
            df['sex'] = pd.to_numeric(df['sex'], errors='coerce')
        
        # Create time variables
        df['hour_of_day'] = df['hour']
        df['minute_of_hour'] = df['minute']
        df['minute_of_day'] = df['hour_of_day'] * 60 + df['minute_of_hour']
        
        # Process step variables - detect zero runs
        step_variables = ['OTHER_DEVICE_STEPS', 'IPHONE_STEPS']
        
        for var in step_variables:
            if var in df.columns:
                valid_flags = detect_zero_runs(df[var], max_consecutive_zeros=61)
                df[f'valid_{var}'] = valid_flags
        
        # Create datetime variable
        df['datetime_var'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour_of_day'], unit='h') + pd.to_timedelta(df['minute_of_hour'], unit='m')
        
        # Calculate time variables
        df = df.sort_values(['id', 'datetime_var']).reset_index(drop=True)
        start_time = df['datetime_var'].iloc[0]
        df['time_minutes'] = ((df['datetime_var'] - start_time).dt.total_seconds() / 60).astype(int)
        df['date_only'] = df['datetime_var'].dt.date
        
        return df
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def process_variables_baseline(df):
    """Process variables and calculate baselines"""
    
    variables = ['OTHER_DEVICE_STEPS', 'IPHONE_STEPS']
    
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
        'sex': analysis_df['sex'].iloc[0] if 'sex' in analysis_df.columns else np.nan,
        'age': analysis_df['age'].iloc[0] if 'age' in analysis_df.columns else np.nan,
        'bmi': analysis_df['bmi'].iloc[0] if 'bmi' in analysis_df.columns else np.nan,
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
    files_path = "V:/P5_PhysAct/Projects/Fenland smartphone/Fenland_data/PHFENLANDR900002352022_16Feb2022_MatthewPearce/HumaStepData_2/participant/phone_unknown_FenlandCOVID"
    bmi_file_path = "V:/P5_PhysAct/Projects/Fenland smartphone/Fenland_data/PHFENLANDR900002352022_16Feb2022_MatthewPearce/HumaStepData_2/bmi.dta"
    output_path = "V:/P5_PhysAct/Projects/Fenland smartphone/Fenland_data/PHFENLANDR900002352022_16Feb2022_MatthewPearce/HumaStepData_2/participant_models/BINS/og/Fenland_phone_unknown_wearable.csv"
    
    # Get list of .dta files
    dta_files = glob.glob(os.path.join(files_path, "*.dta"))
    
    all_results = []
    
    # Process each file
    for file_path in dta_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}")
        
        try:
            # Process participant data
            df = process_participant_data(file_path, bmi_file_path)
            
            if df is None or len(df) == 0:
                continue
            
            # Process variables and calculate baselines
            df = process_variables_baseline(df)
            
            participant_id = df['id'].iloc[0] if 'id' in df.columns else os.path.splitext(filename)[0]
            
            # Define variables to analyze
            variables = ['OTHER_DEVICE_STEPS', 'IPHONE_STEPS']
            
            # Analyze all variable pairs (excluding self-comparisons)
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
        results_df['study'] = 'The Fenland Study'
        
        # Remove rows where outcome == predictor (should already be excluded, but double-check)
        results_df = results_df[results_df['outcome'] != results_df['predictor']]
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        print(f"Total rows: {len(results_df)}")
        
        # Print summary
        print("\nSummary of variable pairs analyzed:")
        pair_counts = results_df.groupby(['outcome', 'predictor']).size().reset_index(name='count')
        for _, row in pair_counts.iterrows():
            print(f"  {row['outcome']} vs {row['predictor']}: {row['count']} participants")
        
        # Print sample statistics
        print(f"\nParticipant characteristics:")
        print(f"  Total participants: {results_df['id'].nunique()}")
        if 'sex' in results_df.columns and results_df['sex'].notna().any():
            sex_counts = results_df['sex'].value_counts()
            print(f"  Sex distribution: {dict(sex_counts)}")
        if 'age' in results_df.columns and results_df['age'].notna().any():
            print(f"  Age: mean={results_df['age'].mean():.1f}, std={results_df['age'].std():.1f}")
        if 'bmi' in results_df.columns and results_df['bmi'].notna().any():
            print(f"  BMI: mean={results_df['bmi'].mean():.1f}, std={results_df['bmi'].std():.1f}")
        
    else:
        print("No results generated")

if __name__ == "__main__":
    main()
