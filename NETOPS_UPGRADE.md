# NetOps AI extension

This repository retains the original supervised network incident prediction workflow. This upgrade adds a separate **unsupervised network anomaly detection** pipeline based on Isolation Forest, plus a KPI monitoring dashboard and continuous integration tests.

## Run

```bash
pip install pandas numpy scikit-learn streamlit plotly joblib pytest
python train_anomaly.py
streamlit run anomaly_dashboard.py
python -m pytest -q tests/test_anomaly.py
```

To run the original supervised project, use its existing `app.py` and training instructions.

## Methodology

All additional data is **synthetic**, generated deterministically from seeded random distributions with injected incident scenarios. The anomaly model uses a chronological 70/30 split and fits exclusively on normal observations in the first 70% of timestamps. Its holdout precision, recall, and F1 are written to `models/anomaly/metrics.json` when training. This is a prototype, not validated on real operator telemetry.

## Resume summary

Built a telecom network incident intelligence extension with Isolation Forest anomaly detection, chronological holdout evaluation, a Streamlit KPI dashboard, and automated GitHub Actions tests. Retained the original supervised classifier to demonstrate two complementary detection approaches.
