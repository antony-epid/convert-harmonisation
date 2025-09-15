this is a long Stata script that does several things:

runs multiple .do files to generate participant-level bins,

appends multiple datasets,

standardizes variable names (predictor/outcome mappings),

generates device/location/units/construct fields,

encodes categorical variables,

assigns train/test splits,

exports multiple CSV summaries.

In Python, the natural equivalents would use pandas for data manipulation and possibly statsmodels/scikit-learn for modeling. Since your script is mainly data wrangling, here’s a structured Python version (using pandas, numpy, glob, and re).
#===========================

# Convert Stata wrangling code to Python
# Matthew Pearce equivalent (Python version)

import pandas as pd
import numpy as np
import glob
import re
from pathlib import Path

# -------------------------------------------------------------------
# 1. Load all participant-level bin CSVs (converted from Stata .dta)
# -------------------------------------------------------------------

DATA_DIR = Path(r"V:\P5_PhysAct\Projects\Fenland smartphone\Fenland_data\PHFENLANDR900002352022_16Feb2022_MatthewPearce\HumaStepData_2\participant_models\BINS\og")

# Load all .csv files into one DataFrame
all_files = glob.glob(str(DATA_DIR / "*.csv"))
dfs = [pd.read_csv(f) for f in all_files]
df = pd.concat(dfs, ignore_index=True)

# -------------------------------------------------------------------
# 2. Drop duplicates, clean outcomes/predictors
# -------------------------------------------------------------------

df = df[df["outcome"] != df["predictor"]]
df = df.drop(columns=["new_id"], errors="ignore")

# Mapping rules (outcome and predictor get same mapping)
rename_map = {
    "Steps": "Actigraph hip steps",
    "enmo_mean_AG": "Actigraph hip ENMO",
    "Vector_Magnitude": "Actigraph hip VM counts",
    "PAI_use": "Actiheart branched equation PAEE",
    "AH_PAEE_aligned": "Actiheart branched equation PAEE",
    "enmo_mean_p2": "GENEActiv NDW ENMO",
    "hpfvm_mean_p2": "GENEActiv NDW HPFVM",
    "AH_trunk_acc_aligned": "Actiheart trunk acceleration",
    "ACC": "Actiheart trunk acceleration",
    "APPLE_WATCH_STEPS": "Apple Watch steps",
    "IPHONE_STEPS": "iPhone steps",
    "PHONE_STEPS": "iPhone steps",
    "OTHER_DEVICE_STEPS": "Other wearable steps",
    "OTHER_STEPS": "Other wearable steps",
    "PAEE": "Actiheart branched equation PAEE",
    "PAI_Branch2": "Actiheart branched equation PAEE",
    "WATCH_STEPS": "Apple Watch steps",
    "actiband_cpm_aligned": "Actiband hip counts",
    "actical_cpm_aligned": "Actical hip counts",
    "enmo_mean_dw": "Axivity DW ENMO",
    "enmo_mean_ndw": "Axivity NDW ENMO",
    "enmo_mean_thigh": "Axivity thigh ENMO",
    "hpfvm_mean_dw": "Axivity DW HPFVM",
    "hpfvm_mean_ndw": "Axivity NDW HPFVM",
    "hpfvm_mean_thigh": "Axivity thigh HPFVM",
    "mti_cpm": "Actigraph hip vertical counts",
    "total_body_movement": "Axivity DW NDW thigh ENMO",
    "enmo_mean_AP": "ActivPAL thigh ENMO",
    "AG_AP_hipthigh_enmo": "Actigraph hip ActivPAL thigh ENMO",
    "AH_PAEE_kJ_min_kg": "Actiheart branched equation PAEE",
    "AX_dw_enmo": "Axivity DW ENMO",
    "AX_dw_hpfvm": "Axivity DW HPFVM",
    "AX_dw_ndw_thigh_enmo": "Axivity DW NDW thigh ENMO",
    "AX_ndw_enmo": "Axivity NDW ENMO",
    "AX_ndw_hpfvm": "Axivity NDW HPFVM",
    "AX_thigh_enmo": "Axivity thigh ENMO",
    "AX_thigh_hpfvm": "Axivity thigh HPFVM",
    "Axis1": "Actigraph hip vertical counts",
    "Axis2": "Actigraph hip horizontal counts",
    "Axis3": "Actigraph hip lateral counts",
    "PAI_flexHR9": "Actiheart flex-HR PAEE",
}

df["outcome"] = df["outcome"].replace(rename_map)
df["predictor"] = df["predictor"].replace(rename_map)

# -------------------------------------------------------------------
# 3. Drop invalid outcome-predictor pairs
# -------------------------------------------------------------------
drop_rules = [
    ("GENEActiv NDW HPFVM", "GENEActiv NDW ENMO"),
    ("Axivity DW HPFVM", "Axivity DW ENMO"),
    ("Axivity NDW HPFVM", "Axivity NDW ENMO"),
    ("Axivity thigh HPFVM", "Axivity thigh ENMO"),
]

for outcome, valid_pred in drop_rules:
    df = df[~((df["outcome"] == outcome) & (df["predictor"] != valid_pred))]

# -------------------------------------------------------------------
# 4. Add device/location/units/construct fields using regex
# -------------------------------------------------------------------

def classify_device(name: str) -> str:
    if pd.isna(name): return ""
    if re.search("Actical", name): return "Actical"
    if re.search("Actigraph", name): return "Actigraph"
    if re.search("Actiheart|flex", name): return "Actiheart"
    if re.search("Apple", name): return "Apple Watch"
    if re.search("Axivity", name): return "Axivity"
    if re.search("GENEActiv", name): return "GENEActiv"
    if re.search("iPhone", name): return "iPhone"
    if re.search("ActivPAL", name): return "ActivPAL"
    if re.search("Other", name): return "Other wearable"
    return ""

def classify_location(name: str) -> str:
    if pd.isna(name): return ""
    if re.search("hip", name, re.I): return "Hip"
    if re.search("Actiheart|flex", name): return "Trunk"
    if re.search("Apple", name): return "Wrist"
    if re.search("DW", name): return "Dominant wrist"
    if re.search("NDW", name): return "Non dominant wrist"
    if re.search("thigh|ActivPAL", name): return "Thigh"
    if re.search("Phone", name): return "Phone"
    if re.search("Other", name): return "NA"
    return ""

def classify_units(name: str) -> str:
    if pd.isna(name): return ""
    if "counts" in name: return "counts/min"
    if "flex" in name or "PAEE" in name: return "J/min/kg"
    if "steps" in name: return "steps/min"
    if "ENMO" in name or "HPFVM" in name: return "milli-g"
    if "acceleration" in name: return "m/s/s"
    return ""

def classify_construct(name: str) -> str:
    if pd.isna(name): return ""
    if "flex" in name: return "Flex HR PAEE"
    if "PAEE" in name: return "Branched equation PAEE"
    if "ENMO" in name: return "ENMO"
    if "VM" in name: return "Actlife vector magnitude counts"
    if "HPFVM" in name: return "HPFVM"
    if "Apple" in name or "iPhone" in name or "Other" in name: return "Apple Health steps"
    if "vertical" in name: return "Actilife vertical counts"
    if "lateral" in name: return "Actilife lateral counts"
    if "horizontal" in name: return "Actilife horizontal counts"
    if "Actigraph hip steps" in name: return "Actilife steps counts"
    if "trunk" in name: return "Uniaxial acceleration"
    if "Actical" in name: return "Kinseoft counts"
    return ""

for prefix in ["o", "p"]:
    col = "outcome" if prefix == "o" else "predictor"
    df[f"{prefix}_device"] = df[col].apply(classify_device)
    df[f"{prefix}_location"] = df[col].apply(classify_location)
    df[f"{prefix}_units"] = df[col].apply(classify_units)
    df[f"{prefix}_construct"] = df[col].apply(classify_construct)

# -------------------------------------------------------------------
# 5. Encode categorical fields
# -------------------------------------------------------------------
cat_vars = [
    "outcome","predictor",
    "o_device","o_location","o_units","o_construct",
    "p_device","p_location","p_units","p_construct"
]
for var in cat_vars:
    df[var] = df[var].astype("category").cat.codes

# -------------------------------------------------------------------
# 6. Train/test split
# -------------------------------------------------------------------
np.random.seed(11432)
df["rand"] = np.random.rand(len(df))
df["train"] = (df.groupby("study")["rand"]
                 .transform(lambda x: x.rank(method="first") <= np.ceil(len(x)*0.6))
                 .astype(int))

# -------------------------------------------------------------------
# 7. Save outputs
# -------------------------------------------------------------------
df.to_csv(r"V:\P5_PhysAct\Projects\Fenland smartphone\models_by_participant_and_device_pair_BINS_window.csv", index=False)

# Device summary
device_summary = (
    df.assign(x=1)
      .groupby(["predictor","p_device","p_location","p_construct","p_units",
                "outcome","o_device","o_location","o_construct","o_units","study"])
      .agg(
          sex=("sex","mean"),
          mean_age=("age","mean"),
          mean_bmi=("bmi","mean"),
          sd_age=("age","std"),
          sd_bmi=("bmi","std"),
          Participants=("x","count"),
          Person_minutes=("n","sum")
      )
      .reset_index()
)
device_summary.to_csv(r"V:\P5_PhysAct\Projects\Fenland smartphone\devices_BINS_window.csv", index=False)

# Device table (unique pairs only)
device_table = device_summary.copy()
device_table["pair"] = device_table.apply(
    lambda row: min(row["outcome"], row["predictor"]) + 1000*max(row["outcome"], row["predictor"]),
    axis=1
)
device_table = device_table.drop_duplicates(subset="pair").drop(columns="pair")
device_table.to_csv(r"V:\P5_PhysAct\Projects\Fenland smartphone\devices_table_BINS_window.csv", index=False)
