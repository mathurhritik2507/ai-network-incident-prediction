"""Build the supplementary unsupervised anomaly detection model."""
import json
from pathlib import Path
import joblib
from netops_ai.analytics import generate_data, fit_evaluate

if __name__=="__main__":
    model,test,metrics=fit_evaluate(generate_data())
    Path("models/anomaly").mkdir(parents=True,exist_ok=True)
    joblib.dump(model,"models/anomaly/model.joblib")
    Path("models/anomaly/metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics,indent=2))
