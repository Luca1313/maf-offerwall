# Note for Task A - Estimating position impact

## Scope

This analysis quantifies how offer position within the offerwall influences key engagement and monetization metrics.
Specifically, it examines trends in **CTR, CR, eRPM, eRPC** and **total publisher revenue** as the offer position changes.
The analysis aggregates and compares these metrics by position, ad unit, and day of the week, including holidays division.
Positions are grouped into bin intervals to smooth noise and reveal broader structural effects.
Additional correlation tests with Pearson and Spearman are performed to measure relationships between position and performance.
A further analysis on the impact of moving offers has been provided forecasting the value of metrics from a position to another one.
A Python project [[Link](https://github.com/Luca1313/maf-offerwall)] has been set up to suite these tasks.

## Assumption

- **Data reliability** — The dataset is assumed to result from a robust ETL process; however, a dual smoothed analysis (original vs. smoothed) is provided to mitigate the effect of low-impression counts or strong outliers, allowing clearer interpretation of both structural trends and (possible) noise-driven anomalies.
- **Binning strategy** — Adunits are grouped into deciles based on total publisher revenue to capture relative performance scale. Positions are grouped into bins of size 10 up to position 100, with all higher positions merged into a single bin to improve visualization and interpretability of long-tail behavior.
- **Binning rationale** — The bin size of 10 was selected to approximate typical offerwall pagination or viewport size, where users generally see around 10 offers per screen. This grouping captures natural visibility breaks and reduces random noise, making trends more interpretable across exposure levels.

## Key Takeaways

- **Position is strongly related with visibility metrics** -- CTR and eRPM typically decline with lower positions, though secondary peaks (positions $\approx$ 300) suggest niche but high-value placements.
- **Position remains the dominant driver of engagement and monetization** -- correlation analysis confirms a strong overall decreasing trend of engagement and monetization metrics with position (Spearman ρ$\approx$−0.75 for CTR and eRPM). While the decay is broadly monotonic, localized peaks around certain positions (e.g. 250–300 and 1000+) suggest context-dependent or niche high-performing placements rather than a perfectly smooth decline.
- **Conversion behavior stabilizes beyond early ranks** -- CR shows limited sensitivity to position after the first few bins, suggesting that users who scroll deeper may be more intent-driven.
- **Revenue concentration is top-heavy** -- the majority of impressions and revenue occur within the top bins ($\approx$ positions 1–50), but certain long-tail positions deliver exceptional outlier performance.
- **Temporal and contextual effects are position-dependent** -- weekends show limited variation in CTR and eRPM, suggesting stable user behavior across the week. However, holidays significantly boost CR and eRPC in the top positions ($\approx$ 1–80), indicating that high-visibility offers convert better when user intent or leisure availability is higher.
- **Anomalies detected** -- extremely high CTR/eRPM values with few impressions likely reflect statistical noise or isolated large-payout events rather than structural effects.