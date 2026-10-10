"""Network KPI anomaly exploration dashboard."""
from pathlib import Path
import joblib
import plotly.express as px
import streamlit as st
from netops_ai.analytics import FEATURES, generate_data, fit_evaluate

st.set_page_config(page_title="NetOps AI — Anomaly Intelligence",layout="wide")
st.title("NetOps AI — Network Incident Intelligence")
st.caption("Anomaly detection, KPI monitoring, and incident triage")
st.warning("Demonstration uses synthetic network telemetry. Alerts are not verified root causes.")
df=generate_data()
sites=st.sidebar.multiselect("Sites",sorted(df.site_id.unique()),default=sorted(df.site_id.unique())[:3])
if not sites: st.stop()
model_path=Path("models/anomaly/model.joblib")
if model_path.exists():
    model=joblib.load(model_path)
    scores=df[df.site_id.isin(sites)].copy()
    scores["anomaly_score"]=-model.decision_function(scores[FEATURES])
    scores["anomaly"]=(scores.anomaly_score>0).astype(int)
else:
    st.info("Using an in-memory demo model. Run python train_anomaly.py to save the trained model.")
    _,scores,_=fit_evaluate(df[df.site_id.isin(sites)])
c1,c2=st.columns(2)
c1.metric("Observations",len(scores))
c2.metric("Detected anomalies",int(scores.anomaly.sum()))
fig=px.line(scores,x="timestamp",y="latency_ms",color="site_id",title="Network latency over time")
st.plotly_chart(fig,use_container_width=True)
st.subheader("Unusual observations")
flagged=scores[scores.anomaly==1].sort_values("anomaly_score",ascending=False)
st.dataframe(flagged,use_container_width=True,hide_index=True)
st.download_button("Download anomalies CSV",flagged.to_csv(index=False),"anomalies.csv")
