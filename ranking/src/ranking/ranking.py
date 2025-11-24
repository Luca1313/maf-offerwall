import pandas as pd
import numpy as np
from google.cloud import bigquery
import os
from utils import calculate_ctr, calculate_eRPM, shuffle_near_equals, make_window_case, plot_expected_data, format_bin_label, load_yaml_from_resources, plot_standard_and_expected_data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
import argparse
from dotenv import load_dotenv

parser = argparse.ArgumentParser(description="Process configurations.")

load_dotenv()

parser.add_argument(
    '--force_query',
    action='store_true',
    help='Set whether to reload the dataset')

args = parser.parse_args()

PROJECT_ID = os.getenv("PROJECT_ID")
client = bigquery.Client(project=PROJECT_ID)
ROOT_DIR = os.path.abspath(os.curdir)
dataset_folder = f'{ROOT_DIR}/dataset/'
output_folder = f'{ROOT_DIR}/results/'
os.makedirs(dataset_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
ranking_output = 'ranking_proposal'

configs = load_yaml_from_resources('ranking_config')

main_table = configs['TABLE_NAME']

decimals = configs['DECIMALS']
NEAR_EQUAL_TOL = configs['NEAR_EQUAL_TOL']
MIN_EPSILON = configs['MIN_EPSILON']

# Time windows weights
window_weights = configs['WINDOW_WEIGHTS']

tables = dict()

for table, name in configs['TABLES'].items():
    tables[table] = f"{main_table}.{name}"

days_intervals = [day for day in window_weights.keys() if day != 'older']

window_case_clicks = make_window_case(days_intervals, "Timestamp")
window_case_impr = make_window_case(days_intervals, "Timestamp")
window_case_events = make_window_case(days_intervals, "DateTime")

query_to_use = f"""
WITH max_ts AS (
  SELECT MAX(latest_ts) AS max_timestamp
  FROM (
    SELECT MAX(TIMESTAMP) AS latest_ts FROM `{tables['clicks']}`
    UNION ALL
    SELECT MAX(TIMESTAMP) AS latest_ts FROM `{tables['impressions']}`
    UNION ALL
    SELECT MAX(TIMESTAMP(DateTime)) AS latest_ts FROM `{tables['events']}`
  )
),
clicks_grouped AS (
  SELECT
    position,
    campaign_id,
    adunit_id,
    {window_case_clicks},
    COUNT(*) AS num_clicks
  FROM `{tables['clicks']}`, max_ts
  GROUP BY position, campaign_id, adunit_id, window_type
),
impr_grouped AS (
  SELECT
    position,
    campaign_id,
    adunit_id,
    {window_case_impr},
    COUNT(*) AS num_impressions
  FROM `{tables['impressions']}`, max_ts
  GROUP BY position, campaign_id, adunit_id, window_type
),
events_grouped AS (
  SELECT
    position,
    campaign_id,
    adunit_id,
    {window_case_events},
    COUNT(event_id) AS num_events,
    ROUND(SUM(publisher_payout), 2) AS total_publisher_revenue,
    ROUND(SUM(user_payout), 2) AS total_user_revenue
  FROM `{tables['events']}`, max_ts
  GROUP BY position, campaign_id, adunit_id, window_type
)

SELECT
  COALESCE(c.position, i.position, e.position) AS position,
  COALESCE(c.campaign_id, i.campaign_id, e.campaign_id) AS campaign_id,
  COALESCE(c.adunit_id, i.adunit_id, e.adunit_id) AS adunit_id,
  COALESCE(c.window_type, i.window_type, e.window_type) AS window_type,
  IFNULL(num_clicks, 0) AS num_clicks,
  IFNULL(num_impressions, 0) AS num_impressions,
  IFNULL(num_events, 0) AS num_events,
  IFNULL(total_publisher_revenue, 0) AS total_publisher_revenue,
  IFNULL(total_user_revenue, 0) AS total_user_revenue
FROM clicks_grouped AS c
FULL OUTER JOIN impr_grouped AS i
  ON c.position = i.position AND c.campaign_id = i.campaign_id AND c.adunit_id = i.adunit_id AND c.window_type = i.window_type
FULL OUTER JOIN events_grouped AS e
  ON COALESCE(c.position, i.position) = e.position
  AND COALESCE(c.campaign_id, i.campaign_id) = e.campaign_id
  AND COALESCE(c.adunit_id, i.adunit_id) = e.adunit_id
  AND COALESCE(c.window_type, i.window_type) = e.window_type
"""

file_name = 'ranking.csv'
file_path = os.path.join(dataset_folder, file_name)

if os.path.exists(file_path) and not args.force_query:
    print(f"Loading dataset from local dataset")
    ranking_dataset = pd.read_csv(file_path)
else:
    print(f"Downloading dataset from BigQuery")
    ranking_dataset = client.query(query_to_use).to_dataframe()
    ranking_dataset[["num_impressions", "num_events"]] = ranking_dataset[
        ["num_impressions", "num_events"]].astype(int)
    ranking_dataset.to_csv(file_path, index=False)
    print(f"Saved dataset to {file_path}")

# Compute basic metrics
ranking_dataset["CTR"] = calculate_ctr(ranking_dataset, decimals)
ranking_dataset["eRPM"] = calculate_eRPM(ranking_dataset, decimals)

ranking_dataset = ranking_dataset.dropna(subset=["CTR", "position", "eRPM"])

pos_stats = (
    ranking_dataset
    .groupby("position", as_index=False)
    .agg(
        pos_impr=("num_impressions", "sum"),
        CTR_pos_mean=("CTR", "mean"),
        eRPM_pos_mean=("eRPM", "mean")
    )
)

# Estimate simple position bias curves, avoiding positions with less than MIN_IMPR_FOR_BIAS impressions
MIN_IMPR_FOR_BIAS = configs["MIN_IMPR_FOR_BIAS"]
pos_stats = pos_stats[pos_stats["pos_impr"] >= MIN_IMPR_FOR_BIAS].copy()

global_ctr_mean = ranking_dataset["CTR"].mean()
global_erpm_mean = ranking_dataset["eRPM"].mean()

# Correct too low biases

EPS = 1e-8

pos_stats["CTR_bias"] = np.where(
    global_ctr_mean <= EPS,
    1.0,
    pos_stats["CTR_pos_mean"] / global_ctr_mean
)
pos_stats["eRPM_bias"] = np.where(
    global_erpm_mean <= EPS,
    1.0,
    pos_stats["eRPM_pos_mean"] / global_erpm_mean
)

# Map bias back to main dataset (1.0 neutral bias if missing)
bias_map_ctr = pos_stats.set_index("position")["CTR_bias"].to_dict()
bias_map_erpm = pos_stats.set_index("position")["eRPM_bias"].to_dict()

ranking_dataset["CTR_bias"] = ranking_dataset["position"].map(bias_map_ctr).fillna(1.0)
ranking_dataset["eRPM_bias"] = ranking_dataset["position"].map(bias_map_erpm).fillna(1.0)

# Build debiased targets

ranking_dataset["CTR_debiased"] = np.where(
    ranking_dataset["CTR_bias"] <= EPS,
    0.0,
    ranking_dataset["CTR"] / ranking_dataset["CTR_bias"]
)
ranking_dataset["eRPM_debiased"] = np.where(
    ranking_dataset["eRPM_bias"] <= EPS,
    0.0,
    ranking_dataset["eRPM"] / ranking_dataset["eRPM_bias"]
)

# Sanity cap to avoid extreme de-biasing from tiny denominators
MAX_DEBIAS_FACTOR = configs["MAX_DEBIAS_FACTOR"]
ranking_dataset["CTR_debiased"] = np.minimum(ranking_dataset["CTR_debiased"], ranking_dataset["CTR"] * MAX_DEBIAS_FACTOR)
ranking_dataset["eRPM_debiased"] = np.minimum(ranking_dataset["eRPM_debiased"], ranking_dataset["eRPM"] * MAX_DEBIAS_FACTOR)

# Keep reference for future comparison
metrics_dataset = ranking_dataset.copy()

# Encode categories
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
ranking_dataset["campaign_encoded"] = enc.fit_transform(ranking_dataset[["campaign_id"]])
ranking_dataset["adunit_encoded"] = enc.fit_transform(ranking_dataset[["adunit_id"]])

# Train models per window and predict expected CTR/eRPM
forecast_accumulator = []

for window in days_intervals + ['older']:
    subset = ranking_dataset[ranking_dataset["window_type"] == str(window)]
    if subset.empty:
        continue

    X = np.column_stack([
        subset["campaign_encoded"],
        subset["adunit_encoded"],
        np.log1p(subset["num_clicks"]),
        np.log1p(subset["num_impressions"]),
        np.log1p(subset["num_events"]),
        np.log1p(subset["total_publisher_revenue"]),
        np.log1p(subset["total_user_revenue"]),
    ])

    # Train on debiased targets, log1p to stabilize skew
    y_ctr = np.log1p(subset["CTR_debiased"].clip(lower=0.0))
    y_erpm = np.log1p(subset["eRPM_debiased"].clip(lower=0.0))

    ctr_model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    ctr_model.fit(X, y_ctr)

    erpm_model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    erpm_model.fit(X, y_erpm)

    X_full = np.column_stack([
        ranking_dataset["campaign_encoded"],
        ranking_dataset["adunit_encoded"],
        np.log1p(ranking_dataset["num_clicks"]),
        np.log1p(ranking_dataset["num_impressions"]),
        np.log1p(ranking_dataset["num_events"]),
        np.log1p(ranking_dataset["total_publisher_revenue"]),
        np.log1p(ranking_dataset["total_user_revenue"]),
    ])

    # Model outputs log-domain, expm1 to bring back original scale
    ctr_pred = np.expm1(ctr_model.predict(X_full)).clip(min=0.0)
    erpm_pred = np.expm1(erpm_model.predict(X_full)).clip(min=0.0)

    forecast_accumulator.append(pd.DataFrame({
        "campaign_id": ranking_dataset["campaign_id"].values,
        "adunit_id": ranking_dataset["adunit_id"].values,
        "window_type": str(window),
        "CTR_pred": ctr_pred,
        "eRPM_pred": erpm_pred,
        "weight": window_weights[window]
    }))

# Aggregate windowed predictions at (campaign_id, adunit_id) level
forecast_df = pd.concat(forecast_accumulator, ignore_index=True)

# Weighted accumulation per (campaign, adunit)
aggregated = (
    forecast_df
    .assign(CTR_w=lambda df: df["CTR_pred"] * df["weight"],
            eRPM_w=lambda df: df["eRPM_pred"] * df["weight"])
    .groupby(["campaign_id", "adunit_id"], as_index=False)
    .agg(
        CTR_expected=("CTR_w", "sum"),
        eRPM_expected=("eRPM_w", "sum"),
        total_weight=("weight", "sum")
    )
)

# Normalize by total weights (normalize in case of some windows missing)
aggregated["CTR_expected"] = aggregated["CTR_expected"] / aggregated["total_weight"]
aggregated["eRPM_expected"] = aggregated["eRPM_expected"] / aggregated["total_weight"]

# Compute combined score
aggregated["score"] = aggregated["CTR_expected"] * aggregated["eRPM_expected"]

offers = aggregated.copy().sort_values(by="score", ascending=False).reset_index(drop=True)

final_rankings = []

# Apply epsilon-greedy per campaign
for _, group in offers.groupby("campaign_id", group_keys=False):
    offers = group.copy().reset_index(drop=True)
    ranked = []

    for i in range(len(offers)):
        EPSILON = max(MIN_EPSILON, 0.1 * np.exp(-0.1 * i))
        rand = np.random.rand()

        if rand < EPSILON and len(offers) > 0:
            # Exploration: pick random offer
            chosen = offers.sample(1)
        else:
            # Exploitation: pick top-scoring offer
            chosen = offers.nlargest(1, "score")

        ranked.append(chosen)
        offers = offers.drop(chosen.index)

    ranked_df = pd.concat(ranked).reset_index(drop=True)
    final_rankings.append(ranked_df)

# Combine all campaigns into a single ranking table shuffling similar offers
ranking = shuffle_near_equals(pd.concat(final_rankings).reset_index(drop=True), NEAR_EQUAL_TOL)

ranking["proposed_position"] = ranking.groupby("campaign_id").cumcount() + 1

comparison_df = (metrics_dataset
    .groupby(["position", "campaign_id", "adunit_id", "CTR_bias", "CTR_debiased", "eRPM_bias", "eRPM_debiased"], as_index=False)
    .agg({
        "num_clicks": "sum",
        "num_impressions": "sum",
        "num_events": "sum",
        "total_publisher_revenue": "sum",
        "total_user_revenue": "sum"
    })
    .merge(ranking[["campaign_id", "adunit_id", "CTR_expected", "eRPM_expected", "score", "proposed_position"]],
        on=["campaign_id", "adunit_id"],
        how="left")
).sort_values(by="score", ascending=False)

# Compute basic metrics
comparison_df["CTR"] = calculate_ctr(comparison_df, decimals)
comparison_df["eRPM"] = calculate_eRPM(comparison_df, decimals)

comparison_df["position_shift"] = comparison_df["position"] - comparison_df["proposed_position"]

# Compute deltas
comparison_df["delta_ctr"] = comparison_df["CTR_expected"] - comparison_df["CTR"]
comparison_df["delta_erpm"] = comparison_df["eRPM_expected"] - comparison_df["eRPM"]

order_for_dump = [
    "campaign_id",
    "adunit_id",
    "score",
    "position",
    "proposed_position",
    "position_shift",
    "CTR",
    "CTR_expected",
    "CTR_bias",
    "CTR_debiased",
    "delta_ctr",
    "eRPM",
    "eRPM_expected",
    "eRPM_bias",
    "eRPM_debiased",
    "delta_erpm",
    "num_clicks",
    "num_impressions",
    "num_events",
    "total_publisher_revenue",
    "total_user_revenue",
]

comparison_df[order_for_dump].to_csv(f"{output_folder}{ranking_output}.csv")

# Aggregate mean change
avg_delta_ctr, avg_delta_erpm = comparison_df["delta_ctr"].mean(), comparison_df["delta_erpm"].mean()
weighted_avg_delta_ctr = np.average(comparison_df["delta_ctr"], weights=comparison_df.get("num_impressions", np.ones(len(comparison_df))))
weighted_avg_delta_erpm = np.average(comparison_df["delta_erpm"], weights=comparison_df.get("num_impressions", np.ones(len(comparison_df))))

summary_df = pd.DataFrame({
    "avg_delta_ctr": [f'{avg_delta_ctr:.{decimals}e}'],
    "weighted_avg_delta_ctr": [f'{weighted_avg_delta_ctr:.{decimals}e}'],
    "avg_delta_erpm": [f'{avg_delta_erpm:.{decimals}e}'],
    "weighted_avg_delta_erpm": [f'{weighted_avg_delta_erpm:.{decimals}e}'],
})

summary_df.to_csv(f"{output_folder}{ranking_output}_metrics.csv")

by_campaign = (
    comparison_df.groupby("campaign_id", as_index=False)
    .agg(
        avg_delta_ctr=("delta_ctr", "mean"),
        avg_delta_erpm=("delta_erpm", "mean"),
        avg_position_shift=("position_shift", "mean")
    )
)
by_campaign.to_csv(f"{output_folder}{ranking_output}_comparison_by_campaign.csv", float_format=f"%.{decimals}e")

# Plot data

vendor_bins = configs['VENDOR_BINS']
position_bins = configs['POSITION_BINS']
position_bins_step = configs['POSITION_BIN_STEP']

plot_name = 'summary'

# Compute total revenue per adunit
revenue_per_adunit = (
    comparison_df.groupby("adunit_id", as_index=False)["total_publisher_revenue"].sum()
    .rename(columns={"total_publisher_revenue": "adunit_total_revenue"})
)

# Merge back to main summary
summary_grouped_adunit_position = comparison_df.merge(revenue_per_adunit, on="adunit_id", how="left")

# Assign each adunit to a revenue bin (auto labels with actual ranges)
revenue_bins = pd.qcut(
    summary_grouped_adunit_position["adunit_total_revenue"],
    q=vendor_bins,
    duplicates="drop"
)

summary_grouped_adunit_position["revenue_category"] = revenue_bins.map(format_bin_label)

# Bin positions
bin_edges = list(range(0, position_bins * position_bins_step + 1, position_bins_step)) + [summary_grouped_adunit_position["position"].max() + 1]
bin_labels = [f"{i+1}-{i + position_bins_step}" for i in range(0, position_bins * position_bins_step, position_bins_step)] + [f">{position_bins * position_bins_step}"]
summary_grouped_adunit_position["proposed_position_bin"] = pd.cut(
    summary_grouped_adunit_position["proposed_position"], bins=bin_edges, labels=bin_labels, right=False
)

# Aggregate by revenue_category + position_bin
agg_adunit = (
    summary_grouped_adunit_position.groupby(["revenue_category", "proposed_position_bin"], as_index=False, observed=False)
    .agg({
        "num_impressions": "sum",
        "num_clicks": "sum",
        "num_events": "sum",
        "total_publisher_revenue": "sum",
        "total_user_revenue": "sum",
    })
)

agg_adunit["CTR_expected"] = (
    summary_grouped_adunit_position.groupby(["revenue_category", "proposed_position_bin"], observed=True)
    .apply(lambda g: np.average(g["CTR_expected"], weights=g["num_impressions"]))
    .values
)

agg_adunit["eRPM_expected"] = (
    summary_grouped_adunit_position.groupby(["revenue_category", "proposed_position_bin"], observed=True)
    .apply(lambda g: np.average(g["eRPM_expected"], weights=g["num_impressions"]))
    .values
)

agg_adunit["CTR"] = (
    summary_grouped_adunit_position.groupby(["revenue_category", "proposed_position_bin"], observed=True)
    .apply(lambda g: np.average(g["CTR"], weights=g["num_impressions"]))
    .values
)

agg_adunit["eRPM"] = (
    summary_grouped_adunit_position.groupby(["revenue_category", "proposed_position_bin"], observed=True)
    .apply(lambda g: np.average(g["eRPM"], weights=g["num_impressions"]))
    .values
)

plot_standard_and_expected_data(agg_adunit, output_folder, plot_name+'_standard_and_expected', vendor_bins)