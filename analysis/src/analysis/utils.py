import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import holidays
from scipy.stats import pearsonr, spearmanr
import yaml
import os

us_holidays = holidays.US(years=[2024, 2025])
base_path = os.path.dirname(os.path.abspath(__file__))
resources = f'{base_path}/../resources/'

def add_holiday_flag(df):
    df["is_holiday"] = df["date"].apply(lambda d: d in us_holidays)
    return df

# return yaml configurations as dict
def load_yaml(yaml_file) -> dict:
    with open(f'{yaml_file}.yml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# return yaml configurations as dict
def load_yaml_from_resources(yaml_filename) -> dict:
    return load_yaml(f'{resources}{yaml_filename}')

def preprocess_timestamp(df, table_name):
    if "Timestamp" in df.columns:
        time_col = "Timestamp"
    elif "DateTime" in df.columns:
        time_col = "DateTime"
    else:
        raise ValueError(f"No timestamp column found in table '{table_name}', headers: {df.columns}")

    df.loc[:, "parsed_time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    df = df.dropna(subset=["parsed_time"]).copy()

    df.loc[:, "date"] = df["parsed_time"].dt.date
    df.loc[:, "day_of_week"] = df["parsed_time"].dt.dayofweek

    return df.drop(columns=["parsed_time"])

def plot_data_split(dataset, path_to_save, name, split_ranges, graphs_per_row=3):

    dataset_to_use = dataset.sort_values("position")

    metrics = ["CTR", "CR", "eRPM", "eRPC", "CTR_smooth"]
    metrics = [m for m in metrics if m in dataset_to_use.columns]
    colors = ["blue", "green", "orange", "red", "grey"]

    # Calculate total number of subplots = metrics × number of splits
    n_ranges = len(split_ranges)
    total_plots = len(metrics) * n_ranges
    n_rows = (total_plots + graphs_per_row - 1) // graphs_per_row

    fig, axes = plt.subplots(n_rows, graphs_per_row, figsize=(14, n_rows * 3), layout="constrained")
    axes = axes.flatten()

    plot_idx = 0
    for metric_idx, metric in enumerate(metrics):
        color = colors[metric_idx % len(colors)]

        for range_idx, (start, end) in enumerate(split_ranges):
            # Slice the data in the given position range
            subset = dataset_to_use[
                (dataset_to_use["position"] >= start) &
                ((dataset_to_use["position"] <= end) if end is not None else True)
            ]

            if subset.empty:
                continue

            ax = axes[plot_idx]
            ax.plot(subset["position"], subset[metric], color=color, linewidth=1)
            ax.set_title(f"{metric} ({start}–{end or subset['position'].max()})",
                         fontsize=11, fontweight="bold", color=color)
            ax.set_xlabel("Position", fontsize=9)
            ax.set_ylabel(metric, fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(axis="both", labelsize=8)

            plot_idx += 1

    # Hide unused subplots
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Performance Metrics by Position (split ranges)", fontsize=14, fontweight="bold")
    fig.savefig(f"{path_to_save}{name}_metrics_split.pdf", format="pdf")
    plt.close(fig)

def plot_data(dataset, path_to_save, name, graphs_per_row=2):
    dataset_to_use = dataset.sort_values("position")

    metrics = ["CTR", "CR", "eRPM", "eRPC", "CTR_smooth"]
    metrics = [metric for metric in metrics if metric in dataset_to_use.columns]
    colors = ["blue", "green", "orange", "red", "grey"]

    # graphs printed as grid
    n_rows = (len(metrics) + 1) // graphs_per_row
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 5), layout="constrained")

    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(
            dataset_to_use["position"],
            dataset_to_use[metric],
            color=colors[i],
            linewidth=1
        )
        ax.set_title(metric, fontsize=11, fontweight="bold", color=colors[i])
        ax.set_xlabel("Position", fontsize=9)
        ax.set_ylabel(metric, fontsize=9)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.tick_params(axis="both", labelsize=8)

    # If you have an odd number of metrics, hide the last empty subplot
    if len(metrics) % 2 != 0:
        fig.delaxes(axes[-1])

    fig.suptitle("Performance Metrics by Position", fontsize=13, fontweight="bold")

    plt.savefig(f"{path_to_save}{name}_metrics.pdf", format="pdf")
    plt.close()

def compute_correlations(df, group_name, decimals=4):
    metrics = ['CTR', 'CR', 'eRPM', 'eRPC', 'total_publisher_revenue']
    results = []
    for m in metrics:
        valid = df[[m, 'position']].dropna()
        if len(valid) > 2:
            p, _ = pearsonr(valid['position'], valid[m])
            s, _ = spearmanr(valid['position'], valid[m])
            results.append({
                "group": group_name,
                "metric": m,
                "pearson": round(p, decimals),
                "spearman": round(s, decimals),
                "num_samples": len(valid)
            })
        else:
            results.append({
                "group": group_name,
                "metric": m,
                "pearson": None,
                "spearman": None,
                "num_samples": len(valid)
            })
    return pd.DataFrame(results)

def plot_knee(df, knee, path_to_save, name):
    # Plot it
    plt.figure(figsize=(6, 4))
    plt.plot(df['position'], df['cum_rev_share'], color='tab:blue', lw=2)
    plt.axvline(knee, color='tab:red', linestyle='--', label=f'Knee ≈ {knee}')
    plt.title("Cumulative Revenue Share vs. Position")
    plt.xlabel("Position")
    plt.ylabel("Cumulative Revenue Share")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_to_save}{name}_knee.pdf", format="pdf")
    plt.close()

def plot_anomaly(df, anomaly_column, path_to_save, name):
    plt.figure(figsize=(7, 4))
    plt.scatter(df['num_impressions'], df['eRPM'], c=df[anomaly_column], cmap='coolwarm', alpha=0.7)
    plt.xscale('log')
    plt.xlabel("Impressions (log scale)")
    plt.ylabel("eRPM")
    plt.title("eRPM vs. Impressions (Anomalies Highlighted)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{path_to_save}{name}_anomaly.pdf", format="pdf")
    plt.close()

def plot_anomaly_by_metric(df, metric, path_to_save, name):
    plt.figure(figsize=(7, 4))
    plt.scatter(
        df["position"], df[metric],
        c=df[f"anomaly_{metric}_w"],
        cmap="coolwarm", alpha=0.7
    )
    plt.title(f"{metric} by Position (Weighted z-score anomalies)")
    plt.xlabel("Position")
    plt.ylabel(metric)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{path_to_save}{name}.pdf", format="pdf")
    plt.close()

def plot_adunits(df, path_to_save, name, bins, graphs_per_row=2):
    metrics = ["CTR", "CR", "eRPM", "eRPC"]
    palette = sns.color_palette("Spectral", n_colors=bins)

    n_rows = (len(metrics) + 1) // graphs_per_row
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 10), layout="constrained")

    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.lineplot(
            data=df,
            x="position_bin",
            y=metric,
            hue="revenue_category",
            palette=palette,
            marker="o",
            linewidth=1.3,
            ax=axes[i]
        )
        # Remove markers from legend handles
        handles, labels = axes[i].get_legend_handles_labels()

        for h in handles:
            h.set_marker("")  # remove legend marker symbol

        axes[i].legend(handles, labels, title="Revenue Category")
        axes[i].set_title(f"{metric} by Position Bin and Adunit Revenue Bin", fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Position Bin")
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, linestyle="--", alpha=0.5)

    # Remove any unused subplots (if len(metrics) < total subplots)
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Metrics by Position Bin and Adunit Revenue Bin", fontsize=14, fontweight="bold")

    # Save one combined figure
    fig.savefig(f"{path_to_save}{name}_metrics.pdf", format="pdf")
    plt.close()

def plot_day_of_week(df, path_to_save, name, graphs_per_row=2):
    metrics = ["CTR", "CR", "eRPM", "eRPC"]
    palette = sns.color_palette("Set2", n_colors=df["day_label"].nunique())

    n_rows = (len(metrics) + graphs_per_row - 1) // graphs_per_row
    fig, axes = plt.subplots(n_rows, graphs_per_row, figsize=(12, 6), layout="constrained")

    axes = axes.flatten()
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for i, metric in enumerate(metrics):
        sns.lineplot(
            data=df,
            x="position_bin",
            y=metric,
            hue="day_label",
            palette=palette,
            marker="o",
            linewidth=1.5,
            ax=axes[i]
        )

        axes[i].set_title(f"{metric} by Position Bin and Day of Week", fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Position Bin", fontsize=9)
        axes[i].set_ylabel(metric, fontsize=9)
        axes[i].tick_params(axis="x", rotation=45, labelsize=8)
        axes[i].tick_params(axis="y", labelsize=8)
        axes[i].grid(True, linestyle="--", alpha=0.4)

        handles, labels = axes[i].get_legend_handles_labels()

        for h in handles:
            h.set_marker("")  # remove legend marker symbol
        sorted_pairs = sorted(zip(labels, handles), key=lambda x: order.index(x[0]) if x[0] in order else 999)
        labels, handles = zip(*sorted_pairs)

        axes[i].legend(handles, labels, title="Day", fontsize=8, title_fontsize=9)

    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Metrics by Position Bin and Day of Week", fontsize=14, fontweight="bold")

    fig.savefig(f"{path_to_save}{name}_metrics.pdf", format="pdf")
    plt.close(fig)

def plot_holiday(df, path_to_save, name, graphs_per_row=2):
    metrics = ["CTR", "CR", "eRPM", "eRPC"]
    palette = sns.color_palette("Set2", n_colors=df["holiday_label"].nunique())

    n_rows = (len(metrics) + graphs_per_row - 1) // graphs_per_row
    fig, axes = plt.subplots(n_rows, graphs_per_row, figsize=(12, 6), layout="constrained")

    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.lineplot(
            data=df,
            x="position_bin",
            y=metric,
            hue="holiday_label",
            palette=palette,
            marker="o",
            linewidth=1.5,
            ax=axes[i]
        )

        # Remove markers from legend handles
        handles, labels = axes[i].get_legend_handles_labels()

        for h in handles:
            h.set_marker("")  # remove legend marker symbol

        axes[i].legend(handles, labels, title="Day")

        axes[i].set_title(f"{metric} by Position Bin and Holiday or Workday", fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Position Bin", fontsize=9)
        axes[i].set_ylabel(metric, fontsize=9)
        axes[i].tick_params(axis="x", rotation=45, labelsize=8)
        axes[i].tick_params(axis="y", labelsize=8)
        axes[i].grid(True, linestyle="--", alpha=0.4)

    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Metrics by Position Bin and Holiday or Workday", fontsize=14, fontweight="bold")

    fig.savefig(f"{path_to_save}{name}_metrics.pdf", format="pdf")
    plt.close(fig)

def get_query(table_name):
    return f"""
        SELECT *
        FROM `{table_name}`
    """

def calculate_ctr(input_df, decimals):
    return np.where(
        input_df["num_impressions"] > 0,
        round(input_df["num_clicks"] / input_df["num_impressions"], decimals + 1),
        0
    )

def calculate_cr(input_df, decimals):
    return np.where(
        input_df["num_clicks"] > 0,
        round(input_df["num_events"] / input_df["num_clicks"], decimals + 1),
        0
    )

def calculate_eRPM(input_df, decimals):
    return np.where(
        input_df["num_impressions"] > 0,
        round(input_df["total_publisher_revenue"] / input_df["num_impressions"] * 1000, decimals),
        0
    )

def calculate_eRPC(input_df, decimals):
    return np.where(
        input_df["num_clicks"] > 0,
        round(input_df["total_publisher_revenue"] / input_df["num_clicks"], decimals + 1),
        0
    )

def weighted_zscore(values, weights):
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    std = np.sqrt(variance)
    return (values - mean) / std

def format_bin_label(interval):
    return f"[{interval.left:,.0f}–{interval.right:,.0f})" if abs(int(interval.left)) != abs(int(interval.right)) else abs(interval.left)

def predict_metric_change(model, pos_from, pos_to):
    # Extract coefficients
    intercept = model.params.get("Intercept", 0)
    slope = model.params.get("log_position", 0)

    # Predict metric values
    metric_from = intercept + slope * np.log1p(pos_from)
    metric_to = intercept + slope * np.log1p(pos_to)
    delta = metric_to - metric_from

    return delta