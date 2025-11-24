from google.cloud import bigquery
from google.cloud import bigquery_storage
import os
import pandas as pd
from kneed import KneeLocator
from dotenv import load_dotenv

from utils import get_query, plot_data, calculate_ctr, calculate_cr, calculate_eRPC, calculate_eRPM, plot_anomaly, \
    plot_adunits, preprocess_timestamp, add_holiday_flag, plot_day_of_week, plot_knee, compute_correlations, \
    plot_data_split, plot_holiday, format_bin_label, load_yaml_from_resources

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
client = bigquery.Client(project=PROJECT_ID)
bqstorage = bigquery_storage.BigQueryReadClient()

configs = load_yaml_from_resources('analysis_config')

ROOT_DIR = os.path.abspath(os.curdir)
output_folder = f'{ROOT_DIR}/results/'
dataset_folder = f'{ROOT_DIR}/dataset/'
os.makedirs(dataset_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

main_table = configs['TABLE_NAME']
decimals = configs['DECIMALS']

tables = dict()

for table, name in configs['TABLES'].items():
    tables[table] = f"{main_table}.{name}"

queries = {}

for tab, tab_name in tables.items():
    file_path = os.path.join(dataset_folder, f"{tab}.csv")

    if os.path.exists(file_path):
        print(f"Loading {tab} from local dataset")
        df = pd.read_csv(file_path)
    else:
        print(f"Downloading {tab} from BigQuery")
        df = (
            client.query(get_query(tab_name))
            .to_dataframe(bqstorage_client=bqstorage)
            .dropna()
        )
        df.to_csv(file_path, index=False)
        print(f"Saved {tab} to {file_path}")

    df = add_holiday_flag(preprocess_timestamp(df, tab))

    queries[tab] = df

# 1) Extract metrics based on position grouping

summary_name = 'summary_by_position'
groupby_query = ["position"]

clicks_grouped_position = (
    queries['clicks']
    .groupby(groupby_query, as_index=False)
    .size()
    .rename(columns={"size": "num_clicks"})
)

impr_grouped_position = (
    queries['impressions']
    .groupby(groupby_query, as_index=False)
    .size()
    .rename(columns={"size": "num_impressions"})
)

events_grouped_position = (
    queries["events"]
    .groupby(groupby_query, as_index=False)
    .agg(
        num_events=("event_id", "count"),
        total_publisher_revenue=("publisher_payout", "sum"),
        total_user_revenue=("user_payout", "sum"),
    )
)

events_grouped_position[["total_publisher_revenue", "total_user_revenue"]] = \
    events_grouped_position[["total_publisher_revenue", "total_user_revenue"]].round(decimals)

# Merge all grouped data on 'position'
summary_grouped_position = (
    clicks_grouped_position
    .merge(impr_grouped_position, on="position", how="outer")
    .merge(events_grouped_position, on="position", how="outer")
    .fillna(0)
)
summary_grouped_position[["num_impressions", "num_events"]] = summary_grouped_position[["num_impressions", "num_events"]].astype(int)

# Calculate metrics

summary_grouped_position["CTR"] = calculate_ctr(summary_grouped_position, decimals)
summary_grouped_position["CR"] = calculate_cr(summary_grouped_position, decimals)
summary_grouped_position["eRPM"] = calculate_eRPM(summary_grouped_position, decimals)
summary_grouped_position["eRPC"] = calculate_eRPC(summary_grouped_position, decimals)

# Add a CTR smooth measure by windows to smooth peaks

window = configs['SMOOTH_WINDOW']

summary_grouped_position['CTR_smooth'] = round(summary_grouped_position['CTR'].rolling(window=window, center=True, min_periods=1).mean(), decimals)

# Save dataset and plot data

summary_grouped_position.to_csv(f"{output_folder}{summary_name}.csv", index=False, float_format=f"%.{decimals}e")

plot_data(summary_grouped_position, output_folder, summary_name)

split_ranges=[(1, 320), (321, 342), (343, None)]

plot_data_split(summary_grouped_position, output_folder, summary_name, split_ranges)

compute_correlations(summary_grouped_position, "by_position").to_csv(f'{output_folder}{summary_name}_correlation.csv', index=False, float_format=f"%.{decimals}e")

# Add cumulative measures for clicks and revenue share

summary_grouped_position['revenue_share'] = round(summary_grouped_position['total_publisher_revenue'] / summary_grouped_position['total_publisher_revenue'].sum(), decimals + 1)

summary_grouped_position['cum_click'] = round(summary_grouped_position['num_clicks'].cumsum() / summary_grouped_position['num_clicks'].sum(), decimals + 1)
summary_grouped_position['cum_rev_share'] = round(summary_grouped_position['revenue_share'].cumsum() / summary_grouped_position['revenue_share'].sum(), decimals + 1)

# Detect knee for revenue

knee = KneeLocator(
    summary_grouped_position['position'], summary_grouped_position['cum_rev_share'],
    curve='concave', direction='increasing'
)

knee_position = knee.knee

plot_knee(summary_grouped_position, knee_position, output_folder, summary_name)

# 2) Look for anomalies

eRPM_median_factor = configs['ERPM_MEDIAN_FACTOR']

# By quantile

quantile = configs['QUANTILE']
IMPR_THRESH = summary_grouped_position['num_impressions'].quantile(quantile)

median_eRPM = summary_grouped_position['eRPM'].median()
anomaly_quantile = 'eRPM_by_quantile'
summary_grouped_position[anomaly_quantile] = (summary_grouped_position['num_impressions'] < IMPR_THRESH) & (summary_grouped_position['eRPM'] > eRPM_median_factor * median_eRPM)

# By SE

ctr_avg = summary_grouped_position['CTR'].mean() / 100
target_se = configs['TARGET_SE']
IMPR_THRESH = (ctr_avg * (1 - ctr_avg)) / (target_se ** 2)

anomaly_se = 'eRPM_by_se'
summary_grouped_position[anomaly_se] = (summary_grouped_position['num_impressions'] < IMPR_THRESH) & (summary_grouped_position['eRPM'] > eRPM_median_factor * median_eRPM)

# Plot both

plot_anomaly(summary_grouped_position, anomaly_quantile, output_folder, f'{summary_name}_{anomaly_quantile}')
plot_anomaly(summary_grouped_position, anomaly_se, output_folder, f'{summary_name}_{anomaly_se}')

# 3) Analysis on adunit

vendor_bins = configs['VENDOR_BINS']
position_bins = configs['POSITION_BINS']
position_bins_step = configs['POSITION_BIN_STEP']

summary_name = 'summary_by_adunit_position'
groupby_query = ["adunit_id", "position"]

clicks_grouped_adunit_position = (
    queries['clicks']
    .groupby(groupby_query, as_index=False)
    .size()
    .rename(columns={"size": "num_clicks"})
)

impr_grouped_adunit_position = (
    queries['impressions']
    .groupby(groupby_query, as_index=False)
    .size()
    .rename(columns={"size": "num_impressions"})
)

events_grouped_adunit_position = (
    queries["events"]
    .groupby(groupby_query, as_index=False)
    .agg(
        num_events=("event_id", "count"),
        total_publisher_revenue=("publisher_payout", "sum"),
        total_user_revenue=("user_payout", "sum"),
    )
)

events_grouped_adunit_position[["total_publisher_revenue", "total_user_revenue"]] = \
    events_grouped_adunit_position[["total_publisher_revenue", "total_user_revenue"]].round(decimals)

summary_grouped_adunit_position = (
    clicks_grouped_adunit_position
    .merge(impr_grouped_adunit_position, on=groupby_query, how="outer")
    .merge(events_grouped_adunit_position, on=groupby_query, how="outer")
    .fillna(0)
)

summary = summary_grouped_adunit_position.copy()

# Compute total revenue per adunit
revenue_per_adunit = (
    summary_grouped_adunit_position.groupby("adunit_id", as_index=False)["total_publisher_revenue"].sum()
    .rename(columns={"total_publisher_revenue": "adunit_total_revenue"})
)

# Merge back to main summary
summary_grouped_adunit_position = summary_grouped_adunit_position.merge(revenue_per_adunit, on="adunit_id", how="left")

# Assign each adunit to a revenue bin (auto labels with actual ranges)
revenue_bins = pd.qcut(
    summary_grouped_adunit_position["adunit_total_revenue"],
    q=vendor_bins,
    duplicates="drop"
)

summary_grouped_adunit_position["revenue_category"] = revenue_bins.map(format_bin_label)

# Bin positions
bin_edges_adunit = list(range(0, position_bins * position_bins_step + 1, position_bins_step)) + [summary_grouped_adunit_position["position"].max() + 1]
bin_labels_adunit = [f"{i+1}-{i + position_bins_step}" for i in range(0, position_bins * position_bins_step, position_bins_step)] + [f">{position_bins * position_bins_step}"]

summary_grouped_adunit_position["position_bin"] = pd.cut(
    summary_grouped_adunit_position["position"], bins=bin_edges_adunit, labels=bin_labels_adunit, right=False
)

# Aggregate by revenue_category + position_bin
agg_adunit = (
    summary_grouped_adunit_position.groupby(["revenue_category", "position_bin"], as_index=False, observed=False)
    .agg({
        "num_impressions": "sum",
        "num_clicks": "sum",
        "num_events": "sum",
        "total_publisher_revenue": "sum",
        "total_user_revenue": "sum"
    })
)

# Recalculate derived metrics per group
agg_adunit["CTR"] = calculate_ctr(agg_adunit, decimals)
agg_adunit["CR"] = calculate_cr(agg_adunit, decimals)
agg_adunit["eRPM"] = calculate_eRPM(agg_adunit, decimals)
agg_adunit["eRPC"] = calculate_eRPC(agg_adunit, decimals)

plot_adunits(agg_adunit, output_folder, summary_name, vendor_bins)

# 4) Analysis on day of week

summary_name = 'summary_by_day_position'
groupby_query = ["day_of_week", "position"]

clicks_grouped_day_position = (
    queries['clicks']
    .groupby(groupby_query, as_index=False)
    .size()
    .rename(columns={"size": "num_clicks"})
)

impr_grouped_day_position = (
    queries['impressions']
    .groupby(groupby_query, as_index=False)
    .size()
    .rename(columns={"size": "num_impressions"})
)

events_grouped_day_position = (
    queries["events"]
    .groupby(groupby_query, as_index=False)
    .agg(
        num_events=("event_id", "count"),
        total_publisher_revenue=("publisher_payout", "sum"),
        total_user_revenue=("user_payout", "sum"),
    )
)

events_grouped_day_position[["total_publisher_revenue", "total_user_revenue"]] = \
    events_grouped_day_position[["total_publisher_revenue", "total_user_revenue"]].round(decimals)

summary_grouped_day_position = (
    clicks_grouped_day_position
    .merge(impr_grouped_day_position, on=groupby_query, how="outer")
    .merge(events_grouped_day_position, on=groupby_query, how="outer")
    .fillna(0)
)

# Map day of week to label
day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
summary_grouped_day_position["day_label"] = summary_grouped_day_position["day_of_week"].map(day_map)

bin_edges_day = list(range(0, position_bins * position_bins_step + 1, position_bins_step)) + [summary_grouped_day_position["position"].max() + 1]
bin_labels_day = [f"{i+1}-{i+position_bins_step}" for i in range(0, position_bins * position_bins_step, position_bins_step)] + [f">{position_bins * position_bins_step}"]

summary_grouped_day_position["position_bin"] = pd.cut(summary_grouped_day_position["position"], bins=bin_edges_day, labels=bin_labels_day, right=False)

agg_day_of_week = (
    summary_grouped_day_position.groupby(["day_label", "position_bin"], as_index=False, observed=False)
    .agg({
        "num_impressions": "sum",
        "num_clicks": "sum",
        "num_events": "sum",
        "total_publisher_revenue": "sum",
        "total_user_revenue": "sum"
    })
)

# Recalculate derived metrics per group
agg_day_of_week["CTR"] = calculate_ctr(agg_day_of_week, decimals)
agg_day_of_week["CR"] = calculate_cr(agg_day_of_week, decimals)
agg_day_of_week["eRPM"] = calculate_eRPM(agg_day_of_week, decimals)
agg_day_of_week["eRPC"] = calculate_eRPC(agg_day_of_week, decimals)

plot_day_of_week(agg_day_of_week, output_folder, summary_name)

# Analysis for holidays

summary_name = 'summary_by_holiday_position'
groupby_query = ["is_holiday", "position"]

clicks_grouped_holiday_position = (
    queries['clicks']
    .groupby(groupby_query, as_index=False)
    .size()
    .rename(columns={"size": "num_clicks"})
)

impr_grouped_holiday_position = (
    queries['impressions']
    .groupby(groupby_query, as_index=False)
    .size()
    .rename(columns={"size": "num_impressions"})
)

events_grouped_holiday_position = (
    queries["events"]
    .groupby(groupby_query, as_index=False)
    .agg(
        num_events=("event_id", "count"),
        total_publisher_revenue=("publisher_payout", "sum"),
        total_user_revenue=("user_payout", "sum"),
    )
)

events_grouped_holiday_position[["total_publisher_revenue", "total_user_revenue"]] = \
    events_grouped_holiday_position[["total_publisher_revenue", "total_user_revenue"]].round(decimals)

summary_grouped_holiday_position = (
    clicks_grouped_holiday_position
    .merge(impr_grouped_holiday_position, on=groupby_query, how="outer")
    .merge(events_grouped_holiday_position, on=groupby_query, how="outer")
    .fillna(0)
)

# Map boolean to labels
holiday_map = { False: "Workday", True: "Holiday" }
summary_grouped_holiday_position["holiday_label"] = summary_grouped_holiday_position["is_holiday"].map(holiday_map)

bin_edges_holiday = list(range(0, position_bins * position_bins_step + 1, position_bins_step)) + [summary_grouped_holiday_position["position"].max() + 1]
bin_labels_holiday = [f"{i+1}-{i+position_bins_step}" for i in range(0, position_bins * position_bins_step, position_bins_step)] + [f">{position_bins * position_bins_step}"]

summary_grouped_holiday_position["position_bin"] = pd.cut(summary_grouped_holiday_position["position"], bins=bin_edges_holiday, labels=bin_labels_holiday, right=False)

agg_holiday = (
    summary_grouped_holiday_position.groupby(["holiday_label", "position_bin"], as_index=False, observed=False)
    .agg({
        "num_impressions": "sum",
        "num_clicks": "sum",
        "num_events": "sum",
        "total_publisher_revenue": "sum",
        "total_user_revenue": "sum"
    })
)

# Recalculate derived metrics per group
agg_holiday["CTR"] = calculate_ctr(agg_holiday, decimals)
agg_holiday["CR"] = calculate_cr(agg_holiday, decimals)
agg_holiday["eRPM"] = calculate_eRPM(agg_holiday, decimals)
agg_holiday["eRPC"] = calculate_eRPC(agg_holiday, decimals)

plot_holiday(agg_holiday, output_folder, summary_name)