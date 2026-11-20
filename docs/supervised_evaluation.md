# Supervised incident baseline: reproducibility and evaluation

This project uses **synthetic** `network_kpi_logs.csv` data. Results are demonstrations of an evaluation workflow, **not** measured telecom incident-forecasting performance.

## Offline reproduction

From the repository root, in a disposable Python environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/
MPLBACKEND=Agg python train_model.py
```

The training script writes `models/incident_prediction_model.joblib`, `outputs/model_report.txt`, and `outputs/feature_importance.png`. Review generated artifacts before committing; they are not needed for this documentation change. No cloud credentials or network services are required for training after dependencies are installed.

## Evaluation contract

- `netops_ai/supervised_split.py` orders observations by UTC timestamp and assigns **distinct timestamps** to disjoint train, validation, and final-test windows (target fractions 60/20/20 of unique timestamps). Rows from different sites at the same timestamp cannot straddle a boundary.
- `train_model.py` fits logistic regression (with scaling) and random forest on the training window only. It chooses between candidates using **validation ROC-AUC**, then reports metrics for the selected candidate on the final test window once.
- The timestamp and target `incident` are removed from the model input. Hour and weekday are derived from each observation's own timestamp; this alone does **not** establish that every remaining KPI is available before an incident.
- Train and validation must each contain both labels. A one-class test window yields no ROC-AUC. With rare positives, report class counts and confusion matrices rather than accuracy alone.
- The checked-in synthetic-data tests assert temporal disjointness, predictor-column exclusions, and class availability. They do not establish deployment readiness.

## Known limitations / leakage audit

1. **Prediction horizon is unspecified.** Define whether the prediction is at observation time, 5 minutes ahead, or another horizon before describing the model as early warning.
2. **Feature availability is unverified.** Audit every KPI, VPN, firewall, and system column for event time, ingestion delay, and whether its value is measured *after* incident onset. Dropping the label and timestamp is necessary but insufficient.
3. **Synthetic generation can encode the label.** Inspect the data generator for direct or indirect target construction from features; unusually high performance may reflect the simulation rules.
4. **Chronological holdout is only one period.** Evaluate rolling-origin windows and incident prevalence drift before drawing broader conclusions.
5. **No calibrated operating threshold or cost model.** AUC does not quantify alert burden, false-positive cost, or incident response improvements.
6. **No real-world validation.** Do not claim reduced MTTR, avoided outages, or production forecasting accuracy.

## Safe next checks

Run the full test suite and supervised training on the checked-in 1,800-row synthetic CSV, record actual class counts and output metrics, then add a deterministic feature-availability audit tied to a declared prediction horizon. Keep all changes on the draft PR branch; no deployment, secrets, real telecom records, or paid resources.
