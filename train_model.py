from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "network_kpi_logs.csv"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
MODELS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

X = df.drop(columns=["timestamp", "incident"])
y = df["incident"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "random_forest": RandomForestClassifier(
        n_estimators=250, max_depth=8, random_state=42, class_weight="balanced"
    )
}

best_name, best_auc, best_model = None, -1, None
report_lines = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    report_lines.append(f"\nMODEL: {name}\n")
    report_lines.append(classification_report(y_test, preds))
    report_lines.append(f"ROC-AUC: {auc:.4f}\n")
    report_lines.append(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}\n")
    if auc > best_auc:
        best_name, best_auc, best_model = name, auc, model

joblib.dump(best_model, MODELS / "incident_prediction_model.joblib")
(OUTPUTS / "model_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

if best_name == "random_forest":
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
else:
    importances = pd.Series(abs(best_model.named_steps["model"].coef_[0]), index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
importances.head(10).plot(kind="bar")
plt.title("Top Features for Network Incident Prediction")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUTS / "feature_importance.png", dpi=160)

print(f"Best model: {best_name} | ROC-AUC: {best_auc:.4f}")
print(f"Saved model to {MODELS / 'incident_prediction_model.joblib'}")
