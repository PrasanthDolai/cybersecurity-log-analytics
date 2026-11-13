import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, ClusteringEvaluator

def main():
    spark = SparkSession.builder \
        .appName("CybersecurityThreatML") \
        .master("local[*]") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    base_dir = os.path.abspath(".")
    parquet_path = f"file://{base_dir}/data/processed/cyber_logs.parquet"
    predictions_path = f"file://{base_dir}/data/processed/ml_predictions.parquet"

    print("\n--- [1/4] Loading Partitioned Parquet Logs ---")
    df = spark.read.parquet(parquet_path)

    # Select high-impact network traffic & behavior features
    categorical_cols = ["proto", "service", "state"]
    numeric_cols = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
        "sttl", "dttl", "sload", "dload", "sloss", "dloss",
        "sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat",
        "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm"
    ]

    # Split back into train and test sets
    train_df = df.filter(col("dataset_split") == "train")
    test_df = df.filter(col("dataset_split") == "test")

    print(f"Training Records: {train_df.count():,} | Testing Records: {test_df.count():,}")

    # 2. Feature Engineering Pipeline
    print("\n--- [2/4] Building Spark MLlib Feature Engineering Pipeline ---")
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
        for c in categorical_cols
    ]
    label_indexer = StringIndexer(inputCol="attack_cat", outputCol="target_label", handleInvalid="keep")

    feature_inputs = [f"{c}_vec" for c in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=feature_inputs, outputCol="raw_features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)

    prep_pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler, scaler])
    prep_model = prep_pipeline.fit(train_df)

    train_ready = prep_model.transform(train_df).cache()
    test_ready = prep_model.transform(test_df).cache()

    # 3. Train & Compare Multiple Classification Models
    print("\n--- [3/4] Training & Evaluating Threat Classification Models ---")
    models = {
        "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="target_label", maxIter=30),
        "Decision Tree Classifier": DecisionTreeClassifier(featuresCol="features", labelCol="target_label", maxDepth=10),
        "Random Forest Classifier": RandomForestClassifier(featuresCol="features", labelCol="target_label", numTrees=60, maxDepth=12, seed=42)
    }

    acc_eval = MulticlassClassificationEvaluator(labelCol="target_label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="target_label", predictionCol="prediction", metricName="f1")

    results = []
    best_model_name = None
    best_acc = -1.0
    best_predictions = None

    for name, clf in models.items():
        print(f"Training {name}...")
        trained_clf = clf.fit(train_ready)
        preds = trained_clf.transform(test_ready)

        acc = acc_eval.evaluate(preds)
        f1 = f1_eval.evaluate(preds)
        results.append((name, acc, f1))

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_predictions = preds

    print("\n============================================================")
    print("         SPARK MLlib THREAT CLASSIFICATION LEADERBOARD       ")
    print("============================================================")
    print(f"{'Model Name':<28} | {'Accuracy':<12} | {'F1-Score':<12}")
    print("-" * 60)
    for name, acc, f1 in results:
        print(f"{name:<28} | {acc*100:>10.2f}% | {f1*100:>10.2f}%")
    print("============================================================")
    print(f"BEST PERFORMING ML MODEL: {best_model_name} ({best_acc*100:.2f}% Accuracy)")
    print("============================================================\n")

    # 4. Unsupervised Anomaly Detection (K-Means Clustering)
    print("--- [4/4] Running K-Means Unsupervised Anomaly Clustering ---")
    kmeans = KMeans(featuresCol="features", predictionCol="anomaly_cluster", k=2, seed=42)
    km_model = kmeans.fit(train_ready)
    clustered_preds = km_model.transform(best_predictions)

    sil_eval = ClusteringEvaluator(featuresCol="features", predictionCol="anomaly_cluster", metricName="silhouette")
    silhouette = sil_eval.evaluate(clustered_preds)
    print(f"K-Means Anomaly Clustering Silhouette Score: {silhouette:.4f}")

    # Save predictions for the Streamlit Dashboard
    save_cols = [
        "id", "proto", "service", "state", "dur", "sbytes", "dbytes",
        "rate", "attack_cat", "label", "target_label", "prediction", "anomaly_cluster"
    ]
    clustered_preds.select(*save_cols).write.mode("overwrite").parquet(predictions_path)
    print(f"Saved model predictions & anomaly clusters to {predictions_path}")

    spark.stop()

if __name__ == "__main__":
    main()
