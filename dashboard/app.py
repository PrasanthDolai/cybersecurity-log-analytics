import json
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Big Data Cybersecurity Analytics (HDFS + Spark)",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Distributed Cybersecurity Log Analytics & Threat Classification")
st.caption("Powered by Apache Hadoop HDFS (1.02 GB CSE-CIC-IDS2018) • Apache Spark SQL • Spark MLlib")

with open("outputs/run_metadata.json") as f:
    meta = json.load(f)

models_df = pd.read_csv("outputs/model_comparison.csv")
attack_df = pd.read_csv("outputs/attack_summary.csv")
port_df = pd.read_csv("outputs/port_summary.csv")
sample_df = pd.read_csv("outputs/dashboard_sample.csv")

best_model = models_df.sort_values("Accuracy (%)", ascending=False).iloc[0]

# Top KPI Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("HDFS Dataset Size", f"{meta['hdfs_size_gb']} GB", "CSE-CIC-IDS2018")
c2.metric("Total Network Flows", f"{meta['total_records']:,}", "Hadoop HDFS Parquet")
c3.metric("Best ML Classifier", best_model["Model"], f"{best_model['Accuracy (%)']}% Accuracy")
c4.metric("K-Means Silhouette (k=7)", f"{meta['kmeans_silhouette']}", "Unsupervised Anomaly Score")

st.divider()

# Row 1: Traffic Breakdown & Targeted Ports
col1, col2 = st.columns(2)
with col1:
    st.subheader("Network Traffic & Attack Category Distribution")
    fig_cat = px.bar(
        attack_df,
        x="attack_cat",
        y="flow_count",
        color="attack_cat",
        text_auto=".2s",
        labels={"attack_cat": "Traffic / Attack Class", "flow_count": "Flow Count"}
    )
    fig_cat.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_cat, use_container_width=True)

with col2:
    st.subheader("Top Targeted Destination Ports Under Attack")
    port_df["target_port"] = port_df["target_port"].astype(str)
    fig_port = px.bar(
        port_df.head(10),
        x="target_port",
        y="attack_events",
        color="attack_cat",
        labels={"target_port": "Destination Port", "attack_events": "Attack Flows", "attack_cat": "Attack Type"}
    )
    fig_port.update_layout(height=400)
    st.plotly_chart(fig_port, use_container_width=True)

st.divider()

# Row 2: Spark MLlib Leaderboard & Live Flow Explorer
col3, col4 = st.columns([1.2, 1])
with col3:
    st.subheader("Spark MLlib Multi-Class Classification Benchmark")
    st.dataframe(models_df, use_container_width=True, hide_index=True)

    fig_ml = px.bar(
        models_df.melt(id_vars="Model", value_vars=["Accuracy (%)", "F1-Score (%)", "Precision (%)", "Recall (%)"]),
        x="Model",
        y="value",
        color="variable",
        barmode="group",
        labels={"value": "Score (%)", "variable": "Metric"}
    )
    fig_ml.update_layout(height=340)
    st.plotly_chart(fig_ml, use_container_width=True)

with col4:
    st.subheader("Interactive Flow Telemetry Explorer")
    selected_class = st.selectbox("Filter Traffic Class:", ["All"] + list(attack_df["attack_cat"].unique()))
    filtered = sample_df if selected_class == "All" else sample_df[sample_df["attack_cat"] == selected_class]
    st.dataframe(
        filtered[["dst_port", "protocol", "flow_duration", "tot_fwd_pkts", "totlen_fwd_pkts", "flow_pkts_s", "attack_cat"]].head(25),
        use_container_width=True,
        hide_index=True
    )
