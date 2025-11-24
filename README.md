# maf-offerwall

## Set up commands

In order to authenticate on gcloud-cli, following commands must be executed, providing authentication token of google:

```bash
brew install --cask gcloud-cli
```

```bash
gcloud auth application-default login
```

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry --version
```

## Configurations

For both tasks, a `.env` file is required inside the `src` package to state project id, an example is provided in `.env-example`.
Moreover, in both `resources` packages, some `[*_]config.yml` files has to be provided following `[*_]config-example.yml` detailing the followings.

### `analysis` package

The file `analysis_config.yml` must be created with these properties:
- `TABLE` -- main table name (i.e. `myappfreeapp.public_exports`)
- `TABLES` -- a dictionary with table category name and table name (i.e. `'clicks': 'clicks_2025_05'`)
- `DECIMALS` -- number of decimals for numerical data stored in scientific notation
- `SMOOTH_WINDOW` -- window size for CTR smoothing (odd number)
- `ERPM_MEDIAN_FACTOR` -- factor used to define the threshold above the median to consider outliers to be filtered combined with impression threshold
- `QUANTILE` -- the quantile used to obtain threshold for impression number, used to detect anomalies
- `TARGET_SE` -- target standard error (e.g., 0.01) to obtain minimum number of impressions required for statistically reliable CTR estimate. Formula: IMPR_THRESH = (ctr_avg * (1 - ctr_avg)) / (TARGET_SE ** 2)
- `VENDOR_BINS` -- number of bins to categorize vendors for aggregated summaries
- `POSITION_BINS` -- number of bins to categorize positions for aggregated summaries (depending on the step, an extra one is added i.e. for step 10 and 10 bins -> 1-10, 11-20, ..., 91-100, 100+)
- `POSITION_BIN_STEP` -- step of position's bins

The file `forecast_config.yml` must be created with these properties:
- `TABLE` -- main table name (i.e. `myappfreeapp.public_exports`)
- `TABLES` -- a dictionary with table category name and table name (i.e. `'clicks': 'clicks_2025_05'`)
- `DECIMALS` -- number of decimals for numerical data stored in scientific notation
- `POSITIONS` -- a list of lists with 2 elements, [starting_position, ending_position] to analyze position changes
- `METRICS` -- list of metrics to consider

### `ranking` package

The file `ranking_config.yml` must be created with these properties:
- `TABLE` -- main table name (i.e. `myappfreeapp.public_exports`)
- `TABLES` -- a dictionary with table category name and table name (i.e. `'clicks': 'clicks_2025_05'`)
- `DECIMALS` -- number of decimals for numerical data stored in scientific notation
- `NEAR_EQUAL_TOL` -- threshold defining the tolerance (e.g. 0.01 = 1%) for considering two offers’ scores as near-equal. Offers within this range are randomly shuffled to avoid deterministic ordering among statistically equivalent items
- `MIN_EPSILON` -- minimum exploration probability used in ε-greedy ranking, ensuring that a small percentage of offers are randomly explored even after the decay of the main ε schedule
- `MIN_IMPR_FOR_BIAS` -- minimum number of impressions for position bias calculation, if not reached a bias 1 is considered
- `MAX_DEBIAS_FACTOR` -- limits over-correction by capping the debiased metric to at most the original observed value multiplied by this factor
- `WINDOW_WEIGHTS` -- dictionary assigning weights to different temporal windows when aggregating forecasts or historical performance. More recent data should be prioritized (e.g. 7: 0.5 means that window of day previous_day(0 if not provided) - 7 are weighted 0.5). Weights' sum should be 1
- `VENDOR_BINS` -- number of bins to categorize vendors for aggregated summaries
- `POSITION_BINS` -- number of bins to categorize positions for aggregated summaries (depending on the step, an extra one is added i.e. for step 10 and 10 bins -> 1-10, 11-20, ..., 91-100, 100+)
- `POSITION_BIN_STEP` -- step of position's bins

## Task A

### Analysis

To suite the task A, the `analysis` package has been provided.
To run the script `analysis.py` using virtual environment provided by `poetry`, the following command must be executed
`poetry run python {path-to-src/analysis/}analysis.py`

#### Output

The generated output is collected in `results` folder, and consists of:
- `summary_by_position.csv` -- aggregated dataset grouped by position, containing computed metrics (i.e. CT, CTR, eRPM and eRPC)
- `summary_by_{adunit_|day_|holiday_|""}position.pdf` -- collection of plots to visualize metric variations across key dimensions, including ad_unit & position, day & position, holiday & position and position
- `summary_by_position_correlation.csv` -- correlation matrix quantifying relationships between performance metrics and position changes
- `summary_by_position_eRPM_by_{quantile|se}_anomaly.pdf` -- plots looking for anomalies based on quantiles and on Standard error to detect unreliable or excessively high eRPM values
- `summary_by_position_knee.pdf` -- curve showing cumulative revenue vs. position change, highlighting the point of diminishing returns (knee point)
- `summary_by_position_metrics_split.pdf` -- comparative visualization splitting metrics by position ranges (e.g. 1–300 vs. 400–max) to inspect behavior before and after detected anomalies or structural breaks

### Forecast

To analyze the impact of position changes, the `forecast.py` script has been provided.
To run the script `forecast.py` using virtual environment provided by `poetry`, the following command must be executed
`poetry run python {path-to-src/analysis/}forecast.py`

#### Model Description

The forecasting step uses a Mixed-Effects Linear Model (MixedLM) to estimate how each metric varies as a function of position.
For every metric under analysis the script fits a model of the form:

```python
model = smf.mixedlm(
    f"{metric} ~ log_position",
    dataset,
    groups=dataset["campaign_id"]
)
result = model.fit()
```

It provides dual effects, a fixed one using log_position, to capture the diminishing impact of rank on performance and a random one by campaign_id, allowing each campaign to have its own baseline behavior, improving robustness and avoiding bias from heterogeneous campaign distributions.
This mixed-effects structure ensures that global trends are learned while accounting for campaign-level variability.

For each trained model and each pair of positions the script computes the forecasted change in the metric

```python
delta = predict_metric_change(model, pos_from, pos_to)
```

#### Output

The generated output is collected in `results` folder, and consists of:
- `forecast_results.csv` -- contains metric considered, position to start and position to end and the forecasted metric delta

## Task B

To suite the task B, the `ranking` package has been provided.
To run the script `ranking.py` using virtual environment provided by `poetry`, the following command must be executed
`poetry run python {path-to-src/analysis/}ranking.py`

### Model Description

To generate the ranking proposal, the system trains two separate forecasting models: one for CTR and one for eRPM.
Both models use a Gradient Boosting Regressor with the following configuration:

```python
GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
```

This setup balances stability and flexibility, enabling the model to capture non-linear relationships in the data while reducing overfitting through subsampling and shallow trees.

#### Input Features

The model predicts expected debiased CTR and eRPM using the following features:
- `campaign_encoded` – encoded identifier of the campaign
- `adunit_encoded` – encoded identifier of the ad unit
- `log(num_clicks + 1)` – past click volume
- `log(num_impressions + 1)` – historical impression volume
- `log(num_events + 1)` – engagement events
- `log(total_publisher_revenue + 1)` – revenue generated for publisher
- `log(total_user_revenue + 1)` – revenue generated for the user

All count/revenue features use `log1p` transformation to reduce skew, stabilize variance, and improve model behavior on long-tailed distributions.
The predicted CTR and eRPM values are then used to compute optimized ranking positions and to analyze the expected uplift across campaigns.

#### Notably excluded

Position is deliberately excluded from the feature set to avoid circularity and ensure that predictions capture inherent item quality, not existing ranking bias.
Instead, position bias is handled explicitly through the debiasing stage, which involves:
- Estimating empirical CTR and eRPM curves per position, aggregating historical impressions, clicks and revenue for each rank
- Computing a position-bias multiplier as the ratio between each position’s average performance and the global average
- Constructing debiased targets by dividing the observed values by their corresponding position-bias factors

### Output

The generated output is collected in `results` folder, and consists of:
- `ranking_proposal.csv` -- contains the full ranking proposal generated by the model: for each (campaign, ad_unit, position) entry, CTR/eRPM have been predicted and used to provide new ordering based also on optimization rules
- `ranking_proposal_comparison_by_campaign.csv` -- contains an analysis on average delta of CTR, eRPM and position shift per campaign
- `ranking_proposal_metrics.csv` -- provides an insight on average delta for CTR and eRPM both standard and weighted by number of impressions considering whole dataset

## Poetry tasks

`Poe` tasks has been defined to make easier to run all the provided programs from both the root project and the root of packages

```bash
poetry poe run_analysis
poetry poe run_forecast
poetry poe run_ranking
```

Each package should be added to the root project in editable mode:

```bash
poetry add -e ./analysis
poetry add -e ./ranking
```

Then, install all dependencies in the root environment, in the root project directory:

```bash
poetry install --no-root
```
