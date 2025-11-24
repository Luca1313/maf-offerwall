import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import os
from matplotlib.lines import Line2D

us_holidays = holidays.US(years=[2024, 2025])

base_path = os.path.dirname(os.path.abspath(__file__))
resources = f'{base_path}/../resources/'

def add_holiday_flag(df):
    df["is_holiday"] = df["date"].apply(lambda d: d in us_holidays)
    return df

def make_window_case(days, field_name):
    # Generate CASE WHEN ... THEN statements dynamically
    cases = "\n      ".join(
        [f"WHEN TIMESTAMP({field_name}) >= TIMESTAMP_SUB(max_timestamp, INTERVAL {d} DAY) THEN '{d}'" for d in days]
    )
    return f"CASE\n      {cases}\n      ELSE 'older'\n    END AS window_type"

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

# return yaml configurations as dict
def load_yaml(yaml_file) -> dict:
    with open(f'{yaml_file}.yml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# return yaml configurations as dict
def load_yaml_from_resources(yaml_filename) -> dict:
    return load_yaml(f'{resources}{yaml_filename}')

def shuffle_near_equals(df, tol):
    groups = []
    temp = []
    prev_score = None

    rand_state = 42

    for _, row in df.iterrows():
        if prev_score is not None and abs(row["score"] - prev_score) / max(prev_score, 1e-9) <= tol:
            temp.append(row)
        else:
            if temp:
                temp_df = pd.DataFrame(temp).sample(frac=1, random_state=rand_state)
                groups.append(temp_df)
            temp = [row]
        prev_score = row["score"]

    if temp:
        groups.append(pd.DataFrame(temp).sample(frac=1, random_state=rand_state))

    return pd.concat(groups, ignore_index=True)

def format_bin_label(interval):
    return f"[{interval.left:,.0f}–{interval.right:,.0f})" if abs(int(interval.left)) != abs(int(interval.right)) else abs(interval.left)

def plot_expected_data(dataset, path_to_save, name, bins, graphs_per_row=2):
    metrics = ["CTR_expected", "eRPM_expected"]
    palette = sns.color_palette("Spectral", n_colors=bins)

    n_rows = (len(metrics) + 1) // graphs_per_row
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 10), layout="constrained")

    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.lineplot(
            data=dataset,
            x="proposed_position_bin",
            y=metric,
            hue="revenue_category",
            palette=palette,
            marker="o",
            linewidth=1.3,
            ax=axes[i]
        )
        axes[i].set_title(f"{metric} by Expected Position Bin and Adunit Revenue Category", fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Expected Position Bin")
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, linestyle="--", alpha=0.5)

    # Remove any unused subplots (if len(metrics) < total subplots)
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    # Save one combined figure
    fig.savefig(f"{path_to_save}{name}_metrics.pdf", format="pdf")
    plt.close()

def plot_standard_and_expected_data(dataset, path_to_save, name, bins, graphs_per_row=2):
    paired_metrics = [
        ("CTR_expected", "CTR"),
        ("eRPM_expected", "eRPM")
    ]
    palette = sns.color_palette("Spectral", n_colors=bins)

    n_rows = (len(paired_metrics) + 1) // graphs_per_row
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 10), layout="constrained")

    axes = axes.flatten()

    for i, (expected_metric, observed_metric) in enumerate(paired_metrics):
        ax = axes[i]

        width=8

        # seaborn plot (color legend auto-attached to ax)
        sns.lineplot(
            data=dataset,
            x="proposed_position_bin",
            y=expected_metric,
            hue="revenue_category",
            palette=palette,
            marker="o",
            markersize=width,
            linewidth=1.4,
            ax=ax,
            )

        # Remove markers from legend handles
        handles, labels = ax.get_legend_handles_labels()

        for h in handles:
            h.set_marker("")  # remove legend marker symbol

        ax.legend(handles, labels, title="Revenue Category")

        # Observed metric as triangle marker
        sns.lineplot(
            data=dataset,
            x="proposed_position_bin",
            y=observed_metric,
            hue="revenue_category",
            palette=palette,
            marker="^",
            markersize=width,
            linewidth=1.4,
            ax=ax,
            legend=False
        )

        ax.set_title(
            f"{observed_metric} Observed vs Expected by Position Bin and Adunit Revenue Bin",
            fontsize=13,
            fontweight="bold"
        )
        ax.set_xlabel("Expected Position Bin")
        ax.set_ylabel(observed_metric)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Add marker legend to the existing seaborn handles
        handles, labels = ax.get_legend_handles_labels()

        # build marker meaning handles
        marker_handles = [
            Line2D([0], [0], marker="o", linestyle="None", color="black", label="Expected"),
            Line2D([0], [0], marker="^", linestyle="None", color="black", label="Observed")
        ]

        # append the marker handles to seaborn's
        handles = handles + marker_handles
        labels = labels + ["Expected", "Observed"]

        ax.legend(handles, labels, title="Revenue Category")

    # Remove unused axes
    for j in range(len(paired_metrics), len(axes)):
        fig.delaxes(axes[j])

    fig.savefig(f"{path_to_save}{name}_metrics.pdf", format="pdf")
    plt.close()