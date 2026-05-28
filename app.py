from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "incident_prediction_model.joblib"

st.set_page_config(page_title="AI Network Incident Prediction", layout="centered")
st.title("AI Network Incident Prediction System")
st.write("Predict incident risk using telecom-style KPI and security telemetry.")

if not MODEL_PATH.exists():
    st.warning("Run `python src/train_model.py` first to create the model.")
    st.stop()

model = joblib.load(MODEL_PATH)

latency_ms = st.slider("Latency (ms)", 5, 200, 65)
packet_loss_pct = st.slider("Packet Loss (%)", 0.0, 10.0, 1.0)
cpu_usage_pct = st.slider("CPU Usage (%)", 0, 100, 55)
memory_usage_pct = st.slider("Memory Usage (%)", 0, 100, 60)
auth_failures = st.slider("Authentication Failures", 0, 50, 4)
vpn_sessions = st.slider("VPN Sessions", 0, 1000, 300)
firewall_blocks = st.slider("Firewall Blocks", 0, 200, 20)
alarm_count = st.slider("Alarm Count", 0, 100, 5)
throughput_mbps = st.slider("Throughput (Mbps)", 10, 1500, 700)
hour = st.slider("Hour of Day", 0, 23, 12)
day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

sample = pd.DataFrame([{
    "latency_ms": latency_ms,
    "packet_loss_pct": packet_loss_pct,
    "cpu_usage_pct": cpu_usage_pct,
    "memory_usage_pct": memory_usage_pct,
    "auth_failures": auth_failures,
    "vpn_sessions": vpn_sessions,
    "firewall_blocks": firewall_blocks,
    "alarm_count": alarm_count,
    "throughput_mbps": throughput_mbps,
    "hour": hour,
    "day_of_week": day_of_week
}])

risk = model.predict_proba(sample)[0, 1]
prediction = model.predict(sample)[0]

st.metric("Incident Risk Score", f"{risk:.1%}")
st.write("Prediction:", "High Incident Risk" if prediction == 1 else "Normal / Low Risk")
