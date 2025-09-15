Great — this Stata script (MASC.do) is doing a visual QC:

Loop over .dta files,

Load data, sort by DATETIME,

Generate minute_of_recording as observation index,

Keep only rows where Pwear_ndw == 1,

If ≥1440 rows remain (≈1 day of data), create 3 scatterplots of enmo_mean_ndw vs time, applying cutoffs (all, <10, <3),

Save each plot as .png with the subject id in the filename.

Here’s the equivalent Python version using pandas, matplotlib, and pyreadstat (to read .dta files):

#=============================
import pandas as pd
import matplotlib.pyplot as plt
import pyreadstat
import glob
from pathlib import Path

# -------------------------------------------------------------------
# 1. Input/Output directories
# -------------------------------------------------------------------
FILES = Path(r"Q:\Data\DATA BACKUP\Biobank_Validation\BBVS_MAINSTUDY\Merge_AH_AX3\1m")
OUT_DIR = Path(r"V:\P5_PhysAct\Projects\Fenland smartphone\figures\bvs_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# 2. Loop through all .dta files
# -------------------------------------------------------------------
all_files = glob.glob(str(FILES / "*.dta"))

for f in all_files:
    # Load Stata file
    df, meta = pyreadstat.read_dta(f)

    # Sort by DATETIME
    if "DATETIME" in df.columns:
        df = df.sort_values("DATETIME")

    # Generate minute_of_recording
    df["minute_of_recording"] = range(1, len(df) + 1)

    # Convert id to numeric (first observation)
    try:
        subject_id = int(df["id"].iloc[0])
    except Exception:
        subject_id = str(df["id"].iloc[0])

    # Keep only rows with Pwear_ndw == 1
    if "Pwear_ndw" not in df.columns:
        continue
    df = df[df["Pwear_ndw"] == 1]

    # Require at least 1440 rows
    if len(df) < 1440:
        continue

    # ----------------------------------------------------------------
    # 3. Plotting helper
    # ----------------------------------------------------------------
    def scatter_plot(data, cutoff=None, suffix=""):
        fig, ax = plt.subplots(figsize=(9, 3))  # matches xsize(3), ysize(1) roughly
        if cutoff is not None:
            data = data[data["enmo_mean_ndw"] < cutoff]

        ax.scatter(
            data["minute_of_recording"],
            data["enmo_mean_ndw"],
            s=2,  # small marker size (like msize(tiny))
            color="black"
        )
        ax.set_xlabel("")  # suppress x-labels
        ax.set_ylabel("enmo_mean_ndw")
        ax.set_title(f"ID {subject_id} {suffix}")
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        out_file = OUT_DIR / f"{suffix}check_{subject_id}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ----------------------------------------------------------------
    # 4. Generate three plots
    # ----------------------------------------------------------------
    scatter_plot(df, cutoff=None, suffix="")
    scatter_plot(df, cutoff=10, suffix="sub10_")
    scatter_plot(df, cutoff=3, suffix="sub3_")

#==============================
🔑 Notes:

pyreadstat.read_dta() reads Stata .dta files into pandas.

Plots are saved in OUT_DIR as:

check_<id>.png

sub10_check_<id>.png

sub3_check_<id>.png

figsize=(9,3) approximates ysize(1) xsize(3) from Stata.

Marker size s=2 gives a similar "tiny" look.
