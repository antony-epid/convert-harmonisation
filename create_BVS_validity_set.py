this BBVS.do is a processing pipeline that:

Imports multiple CSV files,

Creates derived variables (total_body_movement, PAI_use, bmi),

Generates validity flags per device/wear-location,

Builds windowed statistics (mean, sd, counts over moving windows) with rangestat,

Establishes baseline corrections per day,

Saves processed .dta files, appends them, merges with training split,

Drops unneeded columns, applies baseline subtraction,

Outputs a final processed validation dataset.

In Python, we can do this with pandas, numpy, and scipy (for rolling stats). The rangestat logic maps naturally to rolling(window=w, center=True) in pandas.

Here’s a Python equivalent skeleton (I’ve kept it structured so you can slot in file paths and verify step-by-step):

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------
# 1. Input/Output directories
# -------------------------------------------------------------------
FILES = Path(r"Q:\Data\DATA BACKUP\Biobank_Validation\BBVS_MAINSTUDY\Merge_AH_AX3\1m")
TEMP_DIR = Path(r"J:\P5_Activity\MP_TEMP\bvs_files_for_validity")
TRAIN_FILE = Path(r"V:\P5_PhysAct\Projects\Fenland smartphone\train.dta")
FINAL_OUT = Path(r"V:\P5_PhysAct\Projects\Fenland smartphone\do_files\processed_validation_dataset.dta")

# Helper: convert Stata dta with pyreadstat if needed
import pyreadstat

# -------------------------------------------------------------------
# 2. Process each file
# -------------------------------------------------------------------
all_files = glob.glob(str(FILES / "*.csv"))
processed = []

for f in all_files:
    df = pd.read_csv(f)

    # Derived variables
    df["total_body_movement"] = (
        0.5*df["enmo_mean_thigh"] + 
        0.25*df["enmo_mean_ndw"] + 
        0.25*df["enmo_mean_dw"]
    )
    df["PAI_use"] = df["PAI_Branch2"].combine_first(df["PAI_Branch6"])
    df["sex"] = df["sex"].replace({1:0, 0:1})
    df["bmi"] = df["weight"] / (df["height"]**2)

    # Validity flags
    for var in ["enmo_mean_dw","hpfvm_mean_dw"]:
        df[f"valid_{var}"] = (df["Pwear_dw"] == 1).astype(int)

    for var in ["enmo_mean_ndw","hpfvm_mean_ndw"]:
        df[f"valid_{var}"] = (df["Pwear_ndw"] == 1).astype(int)

    for var in ["enmo_mean_thigh","hpfvm_mean_thigh"]:
        df[f"valid_{var}"] = (df["Pwear_thigh"] == 1).astype(int)

    for var in ["PAI_use","ACC"]:
        df[f"valid_{var}"] = (df["PWEAR"] == 1).astype(int)

    df["valid_total_body_movement"] = (
        (df["Pwear_thigh"]==1) & (df["Pwear_ndw"]==1) & (df["Pwear_dw"]==1)
    ).astype(int)

    # Timestamp handling
    df["datetime_var"] = pd.to_datetime(df["DATETIME"], format="%d%m%Y %H:%M:%S", errors="coerce")
    df = df.sort_values(["id","datetime_var"])
    df["time_minutes"] = (df["datetime_var"] - df["datetime_var"].iloc[0]).dt.total_seconds()//60
    df["date_only"] = df["datetime_var"].dt.date

    # ----------------------------------------------------------------
    # 3. Rolling stats (equivalent to rangestat with window=5 minutes)
    # ----------------------------------------------------------------
    window = 5
    half_lo = int(np.ceil(window/2)-1)
    half_hi = int(np.floor(window/2))

    for var in [
        "enmo_mean_dw","enmo_mean_ndw","enmo_mean_thigh",
        "hpfvm_mean_dw","hpfvm_mean_ndw","hpfvm_mean_thigh",
        "total_body_movement","PAI_use","ACC"
    ]:
        if df[f"valid_{var}"].sum() >= 1440:
            df[f"n_{window}_{var}"] = (
                df[f"valid_{var}"]
                .rolling(window=window, center=True, min_periods=1)
                .sum()
            )
            df[f"sd_{window}_{var}"] = (
                df[var]
                .rolling(window=window, center=True, min_periods=1)
                .std()
            )
            df[f"mean_{window}_{var}"] = (
                df[var]
                .rolling(window=window, center=True, min_periods=1)
                .mean()
            )

            # daily baseline (first 60 minutes averages)
            df["rank"] = df.groupby("date_only").cumcount()+1
            df[f"baseline_{window}_{var}"] = (
                df.groupby("date_only")[f"mean_{window}_{var}"]
                .transform(lambda x: x.head(60//window).mean())
            )

            # subtract baseline, truncate at 0
            baseline_val = df[f"baseline_{window}_{var}"].median(skipna=True)
            df[f"baseline_{var}"] = baseline_val
            df[f"{var}_new"] = (df[var] - baseline_val).clip(lower=0)
        else:
            df[f"baseline_{var}"] = np.nan
            df[f"{var}_new"] = df[var]

    # Save intermediate as dta
    out_file = TEMP_DIR / (Path(f).stem + ".dta")
    pyreadstat.write_dta(df, out_file)
    processed.append(out_file)

# -------------------------------------------------------------------
# 4. Append all intermediate .dta files
# -------------------------------------------------------------------
dfs = [pyreadstat.read_dta(f)[0] for f in processed]
df_all = pd.concat(dfs, ignore_index=True)

# -------------------------------------------------------------------
# 5. Merge with train split
# -------------------------------------------------------------------
train_df, _ = pyreadstat.read_dta(TRAIN_FILE)
df_all = df_all.merge(train_df, on="id", how="inner")

# -------------------------------------------------------------------
# 6. Drop unneeded columns (equivalent to drop in Stata)
# -------------------------------------------------------------------
drop_cols = [c for c in df_all.columns if "baseline_5" in c or "n_5" in c or "sd_5" in c or "day_count" in c]
drop_cols += ["max_ibi_1_in_milliseconds","HR","PAI_Branch2","PAI_Branch6","PAI_Branch7"]
df_all = df_all.drop(columns=[c for c in drop_cols if c in df_all.columns], errors="ignore")

# Rename _new versions back
rename_map = {
    "enmo_mean_dw_new":"enmo_mean_dw",
    "enmo_mean_ndw_new":"enmo_mean_ndw",
    "enmo_mean_thigh_new":"enmo_mean_thigh",
    "hpfvm_mean_dw_new":"hpfvm_mean_dw",
    "hpfvm_mean_ndw_new":"hpfvm_mean_ndw",
    "hpfvm_mean_thigh_new":"hpfvm_mean_thigh",
    "total_body_movement_new":"total_body_movement",
    "PAI_use_new":"PAI_use",
    "ACC_new":"ACC"
}
df_all = df_all.rename(columns=rename_map)

# -------------------------------------------------------------------
# 7. Merge with external results (DLW)
# -------------------------------------------------------------------
dlw_df, _ = pyreadstat.read_dta(
    r"V:\P5_PhysAct\People\Matt Pearce\july_update\datasets\BVS_DLW_RESULT.dta"
)
dlw_df = dlw_df.rename(columns={"orig_id":"id"})
merged = dlw_df.merge(df_all, on="id", how="inner")

# -------------------------------------------------------------------
# 8. Save final processed dataset
# -------------------------------------------------------------------
pyreadstat.write_dta(merged, FINAL_OUT)
merged.to_csv(FINAL_OUT_CSV, index=False)

🔑 Key Notes:

rangestat → replaced with rolling(window=5, center=True).

Daily baseline → grouped by date, taking mean of first 60 minutes.

Stata merge → pandas.merge.

drop & rename → handled with pandas equivalents.

.dta read/write → handled by pyreadstat.
