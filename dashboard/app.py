import os
import glob
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Cybersecurity Log Analytics & Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_data
def load_predictions():
    parquet_files = glob.glob("data/processed/ml_predictions.parquet/*.parquet")
    if not parquet_files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    return df

st.title("🛡️ Big Data Cybersecurity Log Analytics & Threat Classification")
st.caption("Powered by Apache Spark 4.3.0 • PySpark MLlib • UNSW-NB15 Intrusion Dataset")

df = load_predictions()

if df.empty:
    st.error("No prediction data found. Run `python3 src/train_models.py` first.")
    st.stop()

# Sidebar Filters
st.sidebar.header("🔍 Threat Filter Controls")
categories = ["All"] + sorted(df["attack_cat"].unique().tolist())
selected_cat = st.sidebar.selectbox("Filter by Threat Category", categories)

protocols = ["All"] + sorted(df["proto"].unique().tolist())
selected_proto = st.sidebar.selectbox("Filter by Network Protocol", protocols)

filtered_df = df.copy()
if selected_cat != "All":
    filtered_df = filtered_df[filtered_df["attack_cat"] == selected_cat]
if selected_proto != "All":
    filtered_df = filtered_df[filtered_df["proto"] == selected_proto]

# Top KPI Row
total_events = len(filtered_df)
attack_events = int((filtered_df["label"] == 1).sum())
normal_events = int((filtered_df["label"] == 0).sum())
burst_anomalies = int((filtered_df["sbytes"] > 50000).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test Logs Analyzed", f"{total_events:,}")
c2.metric("Detected Cyberattacks", f"{attack_events:,}", delta=f"{(attack_events/max(total_events,1))*100:.1f}% of traffic", delta_color="inverse")
c3.metric("Normal Traffic Events", f"{normal_events:,}")
c4.metric("High-Byte Burst Anomalies", f"{burst_anomalies:,}")

st.divider()

# Charts Row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("Attack Category Breakdown")
    cat_counts = filtered_df["attack_cat"].value_counts().reset_index()
    cat_counts.columns = ["Threat Category", "Event Count"]
    fig_cat = px.bar(
        cat_counts,
        x="Threat Category",
        y="Event Count",
        color="Threat Category",
        text_auto=True
    )
    fig_cat.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_cat, use_container_width=True)

with col2:
    st.subheader("Spark MLlib Model Benchmark (UNSW-NB15)")
    leaderboard_df = pd.DataFrame({
        "Model": ["Decision Tree Classifier", "Random Forest Classifier", "Logistic Regression"],
        "Accuracy (%)": [74.98, 72.23, 70.33],
        "F1-Score (%)": [72.33, 67.58, 65.56]
    })
    fig_ml = px.bar(
        leaderboard_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=True
    )
    fig_ml.update_layout(yaxis_range=[0, 100], height=380)
    st.plotly_chart(fig_ml, use_container_width=True)

# Charts Row 2
col3, col4 = st.columns(2)

with col3:
    st.subheader("Top Targeted Services Under Attack")
    attacks_only = filtered_df[filtered_df["label"] == 1]
    if not attacks_only.empty:
        srv_counts = attacks_only["service"].value_counts().head(8).reset_index()
        srv_counts.columns = ["Network Service", "Attack Count"]
        fig_srv = px.pie(srv_counts, names="Network Service", values="Attack Count", hole=0.4)
        fig_srv.update_layout(height=360)
        st.plotly_chart(fig_srv, use_container_width=True)
    else:
        st.info("No attack records in current filter selection.")

with col4:
    st.subheader("K-Means Anomaly Cluster Distribution")
    cluster_counts = filtered_df.groupby(["anomaly_cluster", "attack_cat"]).size().reset_index(name="Count")
    cluster_counts["anomaly_cluster"] = cluster_counts["anomaly_cluster"].map({0: "Cluster 0 (Baseline)", 1: "Cluster 1 (Outlier/Burst)"})
    fig_km = px.bar(
        cluster_counts,
        x="anomaly_cluster",
        y="Count",
        color="attack_cat",
        barmode="stack"
    )
    fig_km.update_layout(height=360, xaxis_title="K-Means Cluster")
    st.plotly_chart(fig_km, use_container_width=True)

st.divider()

# Live Security Log Table
st.subheader("📋 Sample Classified Cybersecurity Events")
display_cols = ["id", "proto", "service", "state", "dur", "sbytes", "dbytes", "attack_cat", "anomaly_cluster"]
st.dataframe(filtered_df[display_cols].head(50), use_container_width=True)
