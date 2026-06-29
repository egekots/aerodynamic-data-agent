# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Identify the file paths relative to this file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_csv_paths = [
    os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "merged_data_L303.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "data", "merged_data_L303.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "..", "data", "merged_data_L303.csv")),
]
possible_xc_paths = [
    os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "xc_locations.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "data", "xc_locations.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "..", "data", "xc_locations.csv")),
]

csv_path = None
for path in possible_csv_paths:
    if os.path.exists(path):
        csv_path = path
        break
if csv_path is None:
    csv_path = "data/merged_data_L303.csv"

xc_path = None
for path in possible_xc_paths:
    if os.path.exists(path):
        xc_path = path
        break
if xc_path is None:
    xc_path = "data/xc_locations.csv"


def plot_pressure_distribution(run_ids: list[int], target_alpha: float, output_filename: str) -> str:
    """Finds the row with AOA closest to target_alpha for each run and plots Cp vs x/c geometry.

    Saves the generated plot silently to the specified output filename. Inverts the Y-axis.

    Args:
        run_ids: A list of RunNo integers to filter the data.
        target_alpha: The target angle of attack (AOA) in degrees.
        output_filename: The file path to save the generated plot image.

    Returns:
        A success message string: "Plot saved as <output_filename>"
    """
    if not os.path.exists(csv_path):
        return f"Error: CSV data file not found at {csv_path}."
    if not os.path.exists(xc_path):
        return f"Error: Geometry file not found at {xc_path}."

    try:
        df = pd.read_csv(csv_path)
        xc_df = pd.read_csv(xc_path)
    except Exception as e:
        return f"Error reading data files: {e}"

    x = xc_df["x/c"].values
    if len(x) != 63:
        return f"Error: Geometry coordinates length is {len(x)}, expected 63."

    # Ensure RunNo is treated numerically
    df["RunNo"] = pd.to_numeric(df["RunNo"], errors="coerce")
    df["AOA (deg)"] = pd.to_numeric(df["AOA (deg)"], errors="coerce")

    # Ensure parent directories exist
    out_dir = os.path.dirname(os.path.abspath(output_filename))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_runs = []
    cp_cols = [f"Cp_{i}" for i in range(1, 64)]

    # A collection of distinct premium colors
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for idx, run_id in enumerate(run_ids):
        df_run = df[df["RunNo"] == run_id]
        if df_run.empty:
            continue

        # Find row closest to target_alpha
        closest_idx = (df_run["AOA (deg)"] - target_alpha).abs().idxmin()
        row = df_run.loc[closest_idx]
        actual_aoa = row["AOA (deg)"]

        cp = row[cp_cols].values.astype(float)
        color = colors[idx % len(colors)]
        ax.plot(x, cp, marker="o", markersize=4, linestyle="-", color=color, label=f"Run {run_id} (AOA: {actual_aoa:.2f}°)")
        plotted_runs.append(run_id)

    if not plotted_runs:
        plt.close(fig)
        return f"Error: No data found for the specified runs: {run_ids}."

    ax.set_xlabel("x/c (Chordwise Position)", fontsize=11)
    ax.set_ylabel("Cp (Pressure Coefficient)", fontsize=11)
    ax.set_title(f"Pressure Distribution (Target AOA: {target_alpha:.2f}°)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # Invert Y-axis: negative Cp up
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)

    return f"Plot saved as {output_filename}"


def compare_integrated_vs_reported(run_id: int, output_filename: str) -> str:
    """Calculates integrated Cl, Cdp, and Cm from the 63 Cp columns and compares them with reported values.

    Generates a 3-subplot comparison figure with a y=x reference line and saves it silently.

    Args:
        run_id: The RunNo integer to filter the data.
        output_filename: The file path to save the generated plot image.

    Returns:
        A success message string: "Plot saved as <output_filename>"
    """
    if not os.path.exists(csv_path):
        return f"Error: CSV data file not found at {csv_path}."
    if not os.path.exists(xc_path):
        return f"Error: Geometry file not found at {xc_path}."

    try:
        df = pd.read_csv(csv_path)
        xc_df = pd.read_csv(xc_path)
    except Exception as e:
        return f"Error reading data files: {e}"

    x = xc_df["x/c"].values
    if len(x) != 63:
        return f"Error: Geometry coordinates length is {len(x)}, expected 63."

    # Ensure RunNo is treated numerically
    df["RunNo"] = pd.to_numeric(df["RunNo"], errors="coerce")
    df_filtered = df[df["RunNo"] == run_id]

    if df_filtered.empty:
        return f"Error: No data found for RunNo {run_id}."

    dx = np.diff(x)
    x_mid = (x[:-1] + x[1:]) / 2

    cl_rep = df_filtered["Cl"].values
    cdp_rep = df_filtered["Cdp"].values
    cm_rep = df_filtered["Cm"].values
    alpha = np.radians(df_filtered["AOA (deg)"].values)

    cp_cols = [f"Cp_{i}" for i in range(1, 64)]
    cps = df_filtered[cp_cols].values

    cl_calc = []
    cdp_calc = []
    cm_calc = []

    for idx, cp in enumerate(cps):
        cp_mid = (cp[:-1] + cp[1:]) / 2
        cn = -np.sum(cp_mid * dx)
        cl = cn
        cdp = cn * np.sin(alpha[idx])
        cm = np.sum(cp_mid * (x_mid - 0.25) * dx)
        cl_calc.append(cl)
        cdp_calc.append(cdp)
        cm_calc.append(cm)

    cl_calc = np.array(cl_calc)
    cdp_calc = np.array(cdp_calc)
    cm_calc = np.array(cm_calc)

    # Ensure parent directories exist
    out_dir = os.path.dirname(os.path.abspath(output_filename))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Subplot 1: Cl
    axes[0].scatter(cl_rep, cl_calc, color="royalblue", alpha=0.7, label="Data points", s=15)
    all_cl = np.concatenate([cl_rep, cl_calc])
    min_cl, max_cl = all_cl.min(), all_cl.max()
    axes[0].plot([min_cl, max_cl], [min_cl, max_cl], "r--", label="y=x")
    axes[0].set_xlabel("Reported Cl", fontsize=10)
    axes[0].set_ylabel("Calculated Cl", fontsize=10)
    axes[0].set_title("Cl Comparison", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Subplot 2: Cdp
    axes[1].scatter(cdp_rep, cdp_calc, color="darkorange", alpha=0.7, label="Data points", s=15)
    all_cdp = np.concatenate([cdp_rep, cdp_calc])
    min_cdp, max_cdp = all_cdp.min(), all_cdp.max()
    axes[1].plot([min_cdp, max_cdp], [min_cdp, max_cdp], "r--", label="y=x")
    axes[1].set_xlabel("Reported Cdp", fontsize=10)
    axes[1].set_ylabel("Calculated Cdp", fontsize=10)
    axes[1].set_title("Cdp Comparison", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # Subplot 3: Cm
    axes[2].scatter(cm_rep, cm_calc, color="forestgreen", alpha=0.7, label="Data points", s=15)
    all_cm = np.concatenate([cm_rep, cm_calc])
    min_cm, max_cm = all_cm.min(), all_cm.max()
    axes[2].plot([min_cm, max_cm], [min_cm, max_cm], "r--", label="y=x")
    axes[2].set_xlabel("Reported Cm", fontsize=10)
    axes[2].set_ylabel("Calculated Cm", fontsize=10)
    axes[2].set_title("Cm Comparison", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    fig.suptitle(f"Run {run_id} - Integrated vs Reported Coefficients", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)

    return f"Plot saved as {output_filename}"
