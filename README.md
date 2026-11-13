# 🛡️ Cybersecurity Log Analytics and Threat Classification (Apache Spark)

End-to-end Big Data analytics and machine learning pipeline built with **Apache Spark (PySpark)** and **Spark MLlib** to process network intrusion logs, execute distributed Spark SQL threat queries, and classify cyberattacks.

## 📌 Architecture & Pipeline
1. **Distributed Ingestion & Storage (`src/etl_and_analytics.py`)**: Ingests **257,673** network log records from the **UNSW-NB15** dataset, standardizes schemas, and writes partitioned **Parquet** files (`attack_cat` partitioning).
2. **Distributed Threat Querying (Spark SQL)**: Identifies attack category distributions, targeted network services (`DNS`, `HTTP`, `SMTP`, `FTP`), and high-volume DoS/Exploit traffic bursts.
3. **Spark MLlib Classification & Anomaly Detection (`src/train_models.py`)**:
   - **Multi-Class Threat Classification**: Evaluates **Decision Tree**, **Random Forest**, and **Logistic Regression** across 10 traffic classes (`Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`).
   - **Unsupervised Anomaly Detection**: Applies **K-Means Clustering** to separate baseline traffic from burst anomalies.
4. **Interactive Security Dashboard (`dashboard/app.py`)**: Built with **Streamlit** and **Plotly** for interactive threat filtering and inspection.

## 📊 Spark MLlib Model Comparison Results
| Model Name | Accuracy | F1-Score |
| :--- | :---: | :---: |
| **Decision Tree Classifier (Best)** | **74.98%** | **72.33%** |
| Random Forest Classifier | 72.23% | 67.58% |
| Logistic Regression | 70.33% | 65.56% |

## 🚀 How to Run
```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run Spark ETL & Distributed SQL Queries
python3 src/etl_and_analytics.py

# 3. Train & Benchmark Spark MLlib Models
python3 src/train_models.py

# 4. Launch Interactive Dashboard
streamlit run dashboard/app.py
