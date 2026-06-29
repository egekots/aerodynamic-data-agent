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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Identify the CSV path relative to this file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Default relative path from app/app_utils/plot_tools.py to data/merged_data_L303.csv is ../../../data/merged_data_L303.csv
possible_paths = [
    os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "merged_data_L303.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "data", "merged_data_L303.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "..", "data", "merged_data_L303.csv")),
]

csv_path = None
for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    # If not found during import, we fallback to relative path from workspace root
    # but don't fail immediately to allow testing or alternative run environments
    csv_path = "data/merged_data_L303.csv"


def plot_time_trend(run_id: int, output_filename: str) -> str:
    """Filters the merged aerodynamic CSV data for the given RunNo and plots Cl, Cdp, and Cm over time.

    The plot consists of 3 vertically stacked subplots sharing the X-axis ('Time (sec)').
    It saves the plot silently to the specified output filename without showing it.

    Args:
        run_id: The RunNo integer to filter the data.
        output_filename: The file path to save the generated plot image.

    Returns:
        A success message string: "Plot saved as <output_filename>"
    """
    if not os.path.exists(csv_path):
        return f"Error: CSV data file not found at {csv_path}."

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV file: {e}"

    # Ensure RunNo is treated numerically to match against run_id
    df["RunNo"] = pd.to_numeric(df["RunNo"], errors="coerce")
    df_filtered = df[df["RunNo"] == run_id]

    if df_filtered.empty:
        return f"No data found for RunNo {run_id}."

    # Sort by Time (sec) to ensure the line plot is ordered correctly
    df_filtered = df_filtered.sort_values(by="Time (sec)")

    # Ensure parent directories for output_filename exist
    out_dir = os.path.dirname(os.path.abspath(output_filename))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Create figure with 3 vertically stacked subplots sharing X-axis ('Time (sec)')
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    # Plot Cl
    axes[0].plot(df_filtered["Time (sec)"], df_filtered["Cl"], color="royalblue", linewidth=1.5)
    axes[0].set_ylabel("Cl", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].set_title(f"Run {run_id} - Lift Coefficient (Cl)", fontsize=10)

    # Plot Cdp
    axes[1].plot(df_filtered["Time (sec)"], df_filtered["Cdp"], color="darkorange", linewidth=1.5)
    axes[1].set_ylabel("Cdp", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].set_title(f"Run {run_id} - Parasitic Drag Coefficient (Cdp)", fontsize=10)

    # Plot Cm
    axes[2].plot(df_filtered["Time (sec)"], df_filtered["Cm"], color="forestgreen", linewidth=1.5)
    axes[2].set_ylabel("Cm", fontsize=10)
    axes[2].set_xlabel("Time (sec)", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].set_title(f"Run {run_id} - Pitching Moment Coefficient (Cm)", fontsize=10)

    fig.suptitle(f"Aerodynamic Coefficients Time Trend: Run {run_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)

    return f"Plot saved as {output_filename}"


def plot_runs_comparison(run_ids: list[int], output_filename: str) -> str:
    """Filters the merged aerodynamic CSV data for the provided RunNo list and plots AOA vs Cl, Cdp, and Cm.

    The plot consists of 3 rows of subplots ('AOA (deg)' vs 'Cl', 'Cdp', 'Cm' respectively),
    plotting each run as a scatter plot with distinct colors and a legend.
    It saves the plot silently to the specified output filename without showing it.

    Args:
        run_ids: A list of RunNo integers to filter the data.
        output_filename: The file path to save the generated plot image.

    Returns:
        A success message string: "Plot saved as <output_filename>"
    """
    if not os.path.exists(csv_path):
        return f"Error: CSV data file not found at {csv_path}."

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV file: {e}"

    # Ensure RunNo is treated numerically
    df["RunNo"] = pd.to_numeric(df["RunNo"], errors="coerce")

    # Filter for the provided RunNos
    df_filtered = df[df["RunNo"].isin(run_ids)]
    if df_filtered.empty:
        return f"No data found for the specified runs: {run_ids}."

    # Ensure parent directories for output_filename exist
    out_dir = os.path.dirname(os.path.abspath(output_filename))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Create figure with 3 rows of subplots (sharing X-axis 'AOA (deg)' is logical and clean)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

    # A collection of distinct premium colors
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for idx, run_id in enumerate(run_ids):
        run_df = df_filtered[df_filtered["RunNo"] == run_id]
        if run_df.empty:
            continue
        color = colors[idx % len(colors)]

        # Scatter plot for Cl
        axes[0].scatter(run_df["AOA (deg)"], run_df["Cl"], color=color, label=f"Run {run_id}", alpha=0.7, s=15)
        # Scatter plot for Cdp
        axes[1].scatter(run_df["AOA (deg)"], run_df["Cdp"], color=color, label=f"Run {run_id}", alpha=0.7, s=15)
        # Scatter plot for Cm
        axes[2].scatter(run_df["AOA (deg)"], run_df["Cm"], color=color, label=f"Run {run_id}", alpha=0.7, s=15)

    # Configure axes details
    axes[0].set_ylabel("Cl (Lift Coefficient)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="best")
    axes[0].set_title("AOA vs Lift Coefficient (Cl)", fontsize=10)

    axes[1].set_ylabel("Cdp (Drag Coefficient)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="best")
    axes[1].set_title("AOA vs Drag Coefficient (Cdp)", fontsize=10)

    axes[2].set_ylabel("Cm (Pitching Moment Coefficient)", fontsize=10)
    axes[2].set_xlabel("AOA (deg)", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="best")
    axes[2].set_title("AOA vs Pitching Moment Coefficient (Cm)", fontsize=10)

    fig.suptitle("Aerodynamic Run Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)

    return f"Plot saved as {output_filename}"
