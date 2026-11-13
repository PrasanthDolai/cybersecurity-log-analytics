# 🛡️ Distributed Cybersecurity Log Analytics & Threat Classification (Hadoop HDFS + Apache Spark)

An end-to-end Big Data cybersecurity pipeline built on **Apache Hadoop HDFS 3.3.6**, **Apache Spark (PySpark & Spark SQL)**, **Spark MLlib**, and **Streamlit**. The pipeline ingests **1.02 GB (3,145,724 network flow records)** of the **CSE-CIC-IDS2018** intrusion detection dataset from distributed HDFS storage, converts raw CSV telemetry into partitioned Parquet, runs distributed Spark SQL threat queries, and benchmarks multi-class intrusion classifiers alongside unsupervised K-Means anomaly clustering.

---

## 🏗️ Big Data Architecture

1. **Distributed Storage (`Hadoop HDFS 3.3.6`):** Raw CSE-CIC-IDS2018 CSV logs (`1.02 GB`) stored under `hdfs://localhost:9000/cybersecurity/raw/` and written back as attack-partitioned Parquet under `hdfs://localhost:9000/cybersecurity/processed/clean_logs.parquet`.
2. **Distributed ETL & Threat Analytics (`src/etl_and_analytics.py`):** Cleans `3,145,724` network flows, handles malformed headers/`NaN`/`Infinity` values, and executes Spark SQL aggregations across attack classes (`Benign`, `DoS attacks-Hulk`, `FTP-BruteForce`, `SSH-Bruteforce`, `DoS attacks-SlowHTTPTest`, `DoS attacks-GoldenEye`, `DoS attacks-Slowloris`) and targeted ports (`80`, `21`, `22`).
3. **Distributed ML Benchmark (`src/train_models.py`):** Trains and evaluates three supervised multi-class classifiers (`80/20` split: `2,516,558` train / `629,166` test flows) and unsupervised K-Means clustering (`k=7`) using Spark MLlib.
4. **Interactive SOC Dashboard (`dashboard/app.py`):** Streamlit + Plotly telemetry dashboard visualizing HDFS dataset KPIs, attack distributions, port targets, and ML model benchmarks.

---

## 📊 Spark MLlib Model Benchmark (3.14M HDFS Flows)

| Model | Accuracy (%) | F1-Score (%) | Precision (%) | Recall (%) | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree Classifier** | **97.31%** | **97.06%** | **97.72%** | **97.31%** | **4.19s** |
| **Random Forest Classifier** | 96.99% | 96.83% | 97.06% | 96.99% | 11.39s |
| **Logistic Regression** | 86.84% | 86.69% | 87.63% | 86.84% | 14.09s |

* **Unsupervised Anomaly Clustering (K-Means, `k=7`):** `0.4824` Silhouette Score (`1.83s`).

---

## 🚀 How to Run

```bash
# 1. Start Hadoop HDFS & verify dataset (>1 GB)
start-dfs.sh
hdfs dfs -du -h /cybersecurity/raw/

# 2. Run Spark ETL & SQL Threat Analytics on HDFS
python3 src/etl_and_analytics.py

# 3. Train Spark MLlib Classifiers & K-Means Clustering
python3 src/train_models.py

# 4. Launch Streamlit Security Operations Dashboard
streamlit run dashboard/app.py
