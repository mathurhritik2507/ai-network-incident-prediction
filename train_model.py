from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from netops_ai.supervised_split import chronological_train_validation_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "network_kpi_logs.csv"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
MODELS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

train_df, validation_df, test_df = chronological_train_validation_test_split(df)
X_train = train_df.drop(columns=["timestamp", "incident"])
y_train = train_df["incident"]
X_validation = validation_df.drop(columns=["timestamp", "incident"])
y_validation = validation_df["incident"]
X_test = test_df.drop(columns=["timestamp", "incident"])
y_test = test_df["incident"]

if y_train.nunique() < 2 or y_validation.nunique() < 2:
    raise ValueError("Train and validation windows must each contain both incident classes")

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "random_forest": RandomForestClassifier(
        n_estimators=250, max_depth=8, random_state=42, class_weight="balanced"
    )
}

best_name, best_validation_auc, best_model = None, -1, None
report_lines = [
    "Synthetic network telemetry; not a production or prospective performance claim.",
    "Chronological train/validation/test split by distinct timestamp (60/20/20 target).",
    "Select the candidate using validation ROC-AUC; evaluate final test once after selection.",
    f"Train: {len(train_df)} rows ending {train_df.timestamp.max()}.",
    f"Validation: {len(validation_df)} rows from {validation_df.timestamp.min()} to {validation_df.timestamp.max()}.",
    f"Test: {len(test_df)} rows starting {test_df.timestamp.min()}.",
]

for name, model in models.items():
    model.fit(X_train, y_train)
    validation_auc = roc_auc_score(y_validation, model.predict_proba(X_validation)[:, 1])
    report_lines.append(f"Validation ROC-AUC ({name}): {validation_auc:.4f}")
    if validation_auc > best_validation_auc:
        best_name, best_validation_auc, best_model = name, validation_auc, model

# Final test is never consulted to select or fit the candidate model.
test_preds = best_model.predict(X_test)
test_probs = best_model.predict_proba(X_test)[:, 1]
report_lines.append(f"\nSelected by validation: {best_name} (ROC-AUC {best_validation_auc:.4f})")
report_lines.append("\nFINAL CHRONOLOGICAL TEST (selected model only)")
report_lines.append(classification_report(y_test, test_preds, zero_division=0))
if y_test.nunique() == 2:
    test_auc = roc_auc_score(y_test, test_probs)
    report_lines.append(f"Final test ROC-AUC: {test_auc:.4f}")
else:
    report_lines.append("Final test ROC-AUC: undefined (test period has only one class)")
report_lines.append(f"Confusion Matrix:\n{confusion_matrix(y_test, test_preds, labels=[0, 1])}")

joblib.dump(best_model, MODELS / "incident_prediction_model.joblib")
(OUTPUTS / "model_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

if best_name == "random_forest":
    importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
else:
    importances = pd.Series(abs(best_model.named_steps["model"].coef_[0]), index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
importances.head(10).plot(kind="bar")
plt.title("Top Features for Network Incident Prediction")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUTS / "feature_importance.png", dpi=160)

print(f"Selected: {best_name} | validation ROC-AUC: {best_validation_auc:.4f} (synthetic)")
print("Final test metrics saved to outputs/model_report.txt; no test-set model selection")
print(f"Saved model to {MODELS / 'incident_prediction_model.joblib'}")
