import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

def main():
    spark = (
        SparkSession.builder
        .appName("CSE-CIC-IDS2018_HDFS_ETL_Analytics")
        .config("spark.driver.memory", "3g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    hdfs_raw_path = "hdfs://localhost:9000/cybersecurity/raw/*.csv"
    hdfs_parquet_path = "hdfs://localhost:9000/cybersecurity/processed/clean_logs.parquet"

    print(f"--- [1/3] Reading 1+ GB CSE-CIC-IDS2018 Dataset from HDFS: {hdfs_raw_path} ---")
    raw_df = spark.read.option("header", "true").option("inferSchema", "false").csv(hdfs_raw_path)

    # Filter out repeated header rows embedded inside CSVs and null labels
    df = raw_df.filter((F.col("Label").isNotNull()) & (F.col("Label") != "Label"))

    # Select core network flow features for analytics and ML
    selected_cols = {
        "Dst Port": "dst_port",
        "Protocol": "protocol",
        "Flow Duration": "flow_duration",
        "Tot Fwd Pkts": "tot_fwd_pkts",
        "Tot Bwd Pkts": "tot_bwd_pkts",
        "TotLen Fwd Pkts": "totlen_fwd_pkts",
        "TotLen Bwd Pkts": "totlen_bwd_pkts",
        "Fwd Pkt Len Max": "fwd_pkt_len_max",
        "Bwd Pkt Len Max": "bwd_pkt_len_max",
        "Flow Byts/s": "flow_byts_s",
        "Flow Pkts/s": "flow_pkts_s",
        "Fwd IAT Mean": "fwd_iat_mean",
        "Bwd IAT Mean": "bwd_iat_mean",
        "SYN Flag Cnt": "syn_flag_cnt",
        "RST Flag Cnt": "rst_flag_cnt",
        "ACK Flag Cnt": "ack_flag_cnt",
        "Init Fwd Win Byts": "init_fwd_win_byts",
        "Label": "attack_cat"
    }

    exprs = []
    for orig_col, clean_col in selected_cols.items():
        if clean_col == "attack_cat":
            exprs.append(F.trim(F.col(orig_col)).alias(clean_col))
        else:
            # Cast numeric columns safely and replace NaN/Infinity with 0.0
            casted = F.col(orig_col).cast(DoubleType())
            cleaned = (
                F.when(casted.isNull() | F.isnan(casted) | (casted == float("inf")) | (casted == float("-inf")), 0.0)
                .otherwise(casted)
                .alias(clean_col)
            )
            exprs.append(cleaned)

    clean_df = df.select(*exprs).withColumn(
        "is_attack", F.when(F.col("attack_cat") == "Benign", 0).otherwise(1)
    )

    total_records = clean_df.count()
    print(f"Total Valid Network Flow Records Ingested from HDFS: {total_records:,}")

    # Write partitioned Parquet back to HDFS
    print(f"--- [2/3] Writing Partitioned Parquet to HDFS: {hdfs_parquet_path} ---")
    clean_df.write.mode("overwrite").partitionBy("attack_cat").parquet(hdfs_parquet_path)

    # Distributed Spark SQL Threat Analytics
    print("--- [3/3] Executing Distributed Spark SQL Threat Queries ---")
    clean_df.createOrReplaceTempView("network_flows")

    print("\n1. Attack Category Distribution (CSE-CIC-IDS2018):")
    spark.sql("""
        SELECT
            attack_cat,
            COUNT(*) AS flow_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_traffic,
            ROUND(AVG(flow_duration), 2) AS avg_duration_us,
            ROUND(AVG(totlen_fwd_pkts), 2) AS avg_fwd_bytes
        FROM network_flows
        GROUP BY attack_cat
        ORDER BY flow_count DESC
    """).show(truncate=False)

    print("2. Top Targeted Destination Ports Under Cyberattack:")
    spark.sql("""
        SELECT
            CAST(dst_port AS INT) AS target_port,
            attack_cat,
            COUNT(*) AS attack_events
        FROM network_flows
        WHERE is_attack = 1
        GROUP BY dst_port, attack_cat
        ORDER BY attack_events DESC
        LIMIT 10
    """).show(truncate=False)

    print("3. High-Volume Burst Anomalies (Forward Bytes > 50,000):")
    spark.sql("""
        SELECT
            attack_cat,
            COUNT(*) AS high_byte_bursts,
            ROUND(MAX(totlen_fwd_pkts), 2) AS max_fwd_bytes
        FROM network_flows
        WHERE totlen_fwd_pkts > 50000
        GROUP BY attack_cat
        ORDER BY high_byte_bursts DESC
    """).show(truncate=False)

    spark.stop()
    print("HDFS ETL & Spark SQL Analytics Completed Successfully!")

if __name__ == "__main__":
    main()
