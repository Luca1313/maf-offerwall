import os
import pandas as pd
import statsmodels.formula.api as smf
from google.cloud import bigquery
from utils import calculate_ctr, calculate_cr, calculate_eRPC, calculate_eRPM, predict_metric_change, load_yaml_from_resources
import numpy as np
from dotenv import load_dotenv

# Results fitting a simple model metrics ~ position

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
client = bigquery.Client(project=PROJECT_ID)

configs = load_yaml_from_resources('forecast_config')

ROOT_DIR = os.path.abspath(os.curdir)
output_folder = f'{ROOT_DIR}/results/'
dataset_folder = f'{ROOT_DIR}/dataset/'
os.makedirs(dataset_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
forecast_output = 'forecast_results'

main_table = configs['TABLE_NAME']
decimals = configs['DECIMALS']

tables = dict()

for table, name in configs['TABLES'].items():
    tables[table] = f"{main_table}.{name}"

# Analysis on log position change, inter-page change, intra-page change and further intra-page change
positions = [tuple(pos) for pos in configs["POSITIONS"]]
metrics = configs["METRICS"]


query_to_use = f"""
WITH clicks_grouped AS (
  SELECT
    position,
    campaign_id,
    COUNT(*) AS num_clicks
  FROM `{tables['clicks']}`
  GROUP BY position, campaign_id
),
impr_grouped AS (
  SELECT
    position,
    campaign_id,
    COUNT(*) AS num_impressions
  FROM `{tables['impressions']}`
  GROUP BY position, campaign_id
),
events_grouped AS (
  SELECT
    position,
    campaign_id,
    COUNT(event_id) AS num_events,
    ROUND(SUM(publisher_payout), 2) AS total_publisher_revenue,
    ROUND(SUM(user_payout), 2) AS total_user_revenue
  FROM `{tables['events']}`
  GROUP BY position, campaign_id
)

SELECT
  COALESCE(c.position, i.position, e.position) AS position,
  COALESCE(c.campaign_id, i.campaign_id, e.campaign_id) AS campaign_id,
  IFNULL(num_clicks, 0) AS num_clicks,
  IFNULL(num_impressions, 0) AS num_impressions,
  IFNULL(num_events, 0) AS num_events,
  IFNULL(total_publisher_revenue, 0) AS total_publisher_revenue,
  IFNULL(total_user_revenue, 0) AS total_user_revenue
FROM clicks_grouped AS c
FULL OUTER JOIN impr_grouped AS i
  ON c.position = i.position AND c.campaign_id = i.campaign_id
FULL OUTER JOIN events_grouped AS e
  ON COALESCE(c.position, i.position) = e.position
  AND COALESCE(c.campaign_id, i.campaign_id) = e.campaign_id
"""

file_name = 'forecast.csv'
file_path = os.path.join(dataset_folder, file_name)

if os.path.exists(file_path):
    print(f"Loading dataset from local dataset")
    summary_grouped_position_campaign = pd.read_csv(file_path)
else:
    print(f"Downloading dataset from BigQuery")
    summary_grouped_position_campaign = client.query(query_to_use).to_dataframe()
    summary_grouped_position_campaign[["num_impressions", "num_events"]] = summary_grouped_position_campaign[
        ["num_impressions", "num_events"]].astype(int)
    summary_grouped_position_campaign.to_csv(file_path, index=False)
    print(f"Saved dataset to {file_path}")

# Calculate metrics

summary_grouped_position_campaign["CTR"] = calculate_ctr(summary_grouped_position_campaign, decimals)
summary_grouped_position_campaign["CR"] = calculate_cr(summary_grouped_position_campaign, decimals)
summary_grouped_position_campaign["eRPM"] = calculate_eRPM(summary_grouped_position_campaign, decimals)
summary_grouped_position_campaign["eRPC"] = calculate_eRPC(summary_grouped_position_campaign, decimals)

m = summary_grouped_position_campaign.dropna(subset=['CTR', 'position', 'campaign_id'])
m["log_position"] = np.log1p(m["position"])

models = dict()

for metric in metrics:
    model = smf.mixedlm(f"{metric} ~ log_position", m, groups=m["campaign_id"])
    result = model.fit()
    models[metric] = result

results = []

for metric, model in models.items():
    deltas = []
    for (pos_from, pos_to) in positions:
        delta = predict_metric_change(model, pos_from=pos_from, pos_to=pos_to)
        deltas.append(delta)
        results.append({
            "metric": metric,
            "pos_from": pos_from,
            "pos_to": pos_to,
            "delta": delta
        })

results_df = pd.DataFrame(results)
results_df["delta"] = results_df["delta"].round(6)

# Save to CSV
file_path = os.path.join(output_folder, f"{forecast_output}.csv")
results_df.to_csv(file_path, index=False, float_format=f"%.{decimals}e")
