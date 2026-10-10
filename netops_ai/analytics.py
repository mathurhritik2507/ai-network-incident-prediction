"""Deterministic synthetic network telemetry and anomaly scoring."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

FEATURES = ["traffic_gbps", "latency_ms", "packet_loss_pct", "cpu_pct", "availability_pct"]

def generate_data(sites=6, hours=200, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for site in range(sites):
        for hour in range(hours):
            traffic = max(.1, 12 + 5*np.sin(hour*np.pi/12)+rng.normal(0, 1.2))
            latency = max(2, 20 + .8*traffic+rng.normal(0, 4))
            loss = max(0, rng.normal(.3, .15))
            cpu = float(np.clip(25+traffic*1.8+rng.normal(0, 6), 0, 100))
            availability = float(np.clip(99.9+rng.normal(0, .05), 0, 100))
            incident = int(hour>20 and rng.random()<.06)
            if incident:
                kind = rng.integers(0,3)
                if kind == 0: latency+=120;loss+=5
                elif kind == 1: cpu=99;traffic*=1.7
                else: availability-=45;traffic*=.1
            rows.append([pd.Timestamp("2026-01-01",tz="UTC")+pd.Timedelta(hours=hour),
                         f"SITE-{site+1:03d}",traffic,latency,loss,cpu,availability,incident])
    return pd.DataFrame(rows,columns=["timestamp","site_id",*FEATURES,"incident"])

def fit_evaluate(df):
    """Chronological split; fit on normal training samples only."""
    df=df.copy()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
    for name in FEATURES: df[name]=pd.to_numeric(df[name],errors="coerce")
    times=sorted(df.timestamp.unique())
    split=times[int(.7*len(times))]
    train=df[(df.timestamp<split)&(df.incident==0)]
    test=df[df.timestamp>=split].copy()
    model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                        IsolationForest(n_estimators=160,contamination=.06,random_state=42))
    model.fit(train[FEATURES])
    test["anomaly_score"]=-model.decision_function(test[FEATURES])
    test["anomaly"]=(test.anomaly_score>0).astype(int)
    return model,test,{"precision":float(precision_score(test.incident,test.anomaly,zero_division=0)),
                       "recall":float(recall_score(test.incident,test.anomaly,zero_division=0)),
                       "f1":float(f1_score(test.incident,test.anomaly,zero_division=0)),
                       "train_rows":len(train),"test_rows":len(test)}
