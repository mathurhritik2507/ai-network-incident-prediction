# AI Network Incident Prediction System

This project predicts whether a network incident is likely using telecom-style KPI, VPN, firewall, and system telemetry data.

## Why I Built This
I previously worked in network automation and analytics, where KPI monitoring, alarm analysis, firewall/VPN logs, and incident response were important parts of operations. This project connects that experience with machine learning by building an early-warning incident prediction model.

## Tech Stack
Python, Pandas, Scikit-learn, Random Forest, Logistic Regression, Matplotlib, Streamlit

## Workflow
1. Load network KPI dataset
2. Perform exploratory data analysis
3. Train ML models
4. Evaluate using accuracy, precision, recall, F1-score, ROC-AUC
5. Show feature importance
6. Run a Streamlit prediction app

## How to Run
```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

## Business Impact
A model like this can help network teams identify high-risk incidents earlier, reduce MTTR, and prioritize alarms more effectively.

## What I Learned
This project helped me understand how operational network data can be converted into features for predictive modeling and how recall is important when missing an incident can be costly.
