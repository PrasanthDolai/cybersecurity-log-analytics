import os
import json
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, ClusteringEvaluator

def main():
    spark = (
        SparkSession.builder
        .appName("CSE-CIC-IDS2018_HDFS_MLlib_Benchmark")
        .config("spark.driver.memory", "3g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    hdfs_parquet_path = "hdfs://localhost:9000/cybersecurity/processed/clean_logs.parquet"
    print(f"--- [1/4] Loading Cleaned Parquet Data from HDFS: {hdfs_parquet_path} ---")
    df = spark.read.parquet(hdfs_parquet_path)

    feature_cols = [
        "dst_port", "protocol", "flow_duration",
        "tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts", "totlen_bwd_pkts",
        "fwd_pkt_len_max", "bwd_pkt_len_max", "flow_byts_s", "flow_pkts_s",
        "fwd_iat_mean", "bwd_iat_mean", "syn_flag_cnt", "rst_flag_cnt",
        "ack_flag_cnt", "init_fwd_win_byts"
    ]

    # Index target labels (7 classes: Benign + 6 Attack Types)
    label_indexer = StringIndexer(inputCol="attack_cat", outputCol="label", handleInvalid="skip")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)

    prep_pipeline = Pipeline(stages=[label_indexer, assembler, scaler])
    prep_model = prep_pipeline.fit(df)
    prepared_df = prep_model.transform(df).select("features", "label", "attack_cat", "is_attack")

    # 80/20 Train-Test Split across 3.14M records
    train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()
    test_df.cache()

    train_count = train_df.count()
    test_count = test_df.count()
    print(f"Training Set: {train_count:,} flows | Testing Set: {test_count:,} flows")

    # Evaluators
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    rec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

    models = {
        "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label", maxIter=20),
        "Decision Tree Classifier": DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=10, maxBins=64),
        "Random Forest Classifier": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20, maxDepth=10, maxBins=64, seed=42),
    }

    results = []
    print("\n--- [2/4] Training & Evaluating Supervised Spark MLlib Classifiers ---")
    for name, clf in models.items():
        t0 = time.time()
        model = clf.fit(train_df)
        preds = model.transform(test_df)

        acc = round(acc_eval.evaluate(preds) * 100, 2)
        f1 = round(f1_eval.evaluate(preds) * 100, 2)
        prec = round(prec_eval.evaluate(preds) * 100, 2)
        rec = round(rec_eval.evaluate(preds) * 100, 2)
        elapsed = round(time.time() - t0, 2)

        print(f" -> {name}: Accuracy={acc}%, F1={f1}%, Precision={prec}%, Recall={rec}% ({elapsed}s)")
        results.append({
            "Model": name,
            "Accuracy (%)": acc,
            "F1-Score (%)": f1,
            "Precision (%)": prec,
            "Recall (%)": rec,
            "Training Time (s)": elapsed
        })

    # Unsupervised Anomaly Clustering (K-Means) on sample for fast silhouette evaluation
    print("\n--- [3/4] Running Unsupervised K-Means Clustering (k=7) ---")
    t0 = time.time()
    cluster_sample = test_df.sample(fraction=0.1, seed=42)
    kmeans = KMeans(featuresCol="features", k=7, seed=42, maxIter=15)
    km_model = kmeans.fit(cluster_sample)
    km_preds = km_model.transform(cluster_sample)
    sil_eval = ClusteringEvaluator(featuresCol="features", predictionCol="prediction", metricName="silhouette")
    silhouette = round(sil_eval.evaluate(km_preds), 4)
    km_time = round(time.time() - t0, 2)
    print(f" -> K-Means Clustering (k=7): Silhouette Score={silhouette} ({km_time}s)")

    # Save metrics & lightweight dashboard summary files locally for Streamlit
    print("\n--- [4/4] Saving Benchmark Results & Dashboard Artifacts ---")
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(results).to_csv("outputs/model_comparison.csv", index=False)

    # Save aggregated stats & stratified sample for Streamlit dashboard
    cat_summary = (
        df.groupBy("attack_cat")
        .agg(
            F.count("*").alias("flow_count"),
            F.round(F.avg("flow_duration"), 2).alias("avg_duration_us"),
            F.round(F.avg("totlen_fwd_pkts"), 2).alias("avg_fwd_bytes"),
            F.round(F.avg("flow_pkts_s"), 2).alias("avg_pkts_per_sec")
        )
        .orderBy(F.desc("flow_count"))
        .toPandas()
    )
    cat_summary.to_csv("outputs/attack_summary.csv", index=False)

    port_summary = (
        df.filter(F.col("is_attack") == 1)
        .groupBy(F.col("dst_port").cast("int").alias("target_port"), "attack_cat")
        .agg(F.count("*").alias("attack_events"))
        .orderBy(F.desc("attack_events"))
        .limit(15)
        .toPandas()
    )
    port_summary.to_csv("outputs/port_summary.csv", index=False)

    sample_pdf = (
        df.sample(fraction=0.005, seed=42)
        .limit(15000)
        .toPandas()
    )
    sample_pdf.to_csv("outputs/dashboard_sample.csv", index=False)

    meta = {
        "dataset": "CSE-CIC-IDS2018 (HDFS Distributed Storage)",
        "hdfs_raw_path": "hdfs://localhost:9000/cybersecurity/raw/",
        "hdfs_size_gb": 1.02,
        "total_records": int(df.count()),
        "train_records": int(train_count),
        "test_records": int(test_count),
        "kmeans_silhouette": silhouette
    }
    with open("outputs/run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    spark.stop()
    print("All Spark MLlib Benchmarks & Artifacts Saved Successfully!")

if __name__ == "__main__":
    main()
