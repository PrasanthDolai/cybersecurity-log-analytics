import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, trim

def main():
    # 1. Start Apache Spark Session (explicitly set local filesystem so it doesn't look for HDFS on port 9000)
    spark = SparkSession.builder \
        .appName("CybersecurityLogETLAndAnalytics") \
        .master("local[*]") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    base_dir = os.path.abspath(".")
    train_path = f"file://{base_dir}/data/raw/UNSW_NB15_training-set.csv"
    test_path = f"file://{base_dir}/data/raw/UNSW_NB15_testing-set.csv"
    output_path = f"file://{base_dir}/data/processed/cyber_logs.parquet"

    print("\n--- [1/4] Ingesting Raw Cybersecurity Logs into Spark ---")
    train_df = spark.read.csv(train_path, header=True, inferSchema=True)
    test_df = spark.read.csv(test_path, header=True, inferSchema=True)

    train_df = train_df.withColumn("dataset_split", when(col("id").isNotNull(), "train"))
    test_df = test_df.withColumn("dataset_split", when(col("id").isNotNull(), "test"))
    full_df = train_df.unionByName(test_df)

    print(f"Total Cybersecurity Log Records Ingested: {full_df.count():,}")

    # 2. Data Cleaning & Standardization
    print("\n--- [2/4] Cleaning & Standardizing Log Fields ---")
    clean_df = full_df \
        .withColumn("service", when(col("service") == "-", "unknown").otherwise(col("service"))) \
        .withColumn("attack_cat", trim(col("attack_cat"))) \
        .withColumn("attack_cat", when(col("attack_cat").isNull() | (col("attack_cat") == ""), "Normal").otherwise(col("attack_cat"))) \
        .na.fill(0)

    # 3. Distributed Storage: Save as Partitioned Parquet
    print(f"\n--- [3/4] Writing Cleaned Logs to Distributed Parquet Format ---")
    clean_df.write.mode("overwrite").partitionBy("attack_cat").parquet(output_path)

    # 4. Distributed Querying using Spark SQL
    print("\n--- [4/4] Running Distributed Security Analytics Queries (Spark SQL) ---")
    parquet_df = spark.read.parquet(output_path)
    parquet_df.createOrReplaceTempView("cyber_logs")

    print("\n[Query 1] Attack Category Distribution & Average Bytes Transferred:")
    spark.sql("""
        SELECT 
            attack_cat AS Threat_Category,
            COUNT(*) AS Event_Count,
            ROUND(AVG(sbytes), 2) AS Avg_Source_Bytes,
            ROUND(AVG(dbytes), 2) AS Avg_Dest_Bytes,
            ROUND(AVG(rate), 2) AS Avg_Packet_Rate
        FROM cyber_logs
        GROUP BY attack_cat
        ORDER BY Event_Count DESC
    """).show(truncate=False)

    print("\n[Query 2] Top Targeted Network Services Under Attack:")
    spark.sql("""
        SELECT 
            service AS Network_Service,
            proto AS Protocol,
            COUNT(*) AS Attack_Count
        FROM cyber_logs
        WHERE label = 1
        GROUP BY service, proto
        ORDER BY Attack_Count DESC
        LIMIT 10
    """).show(truncate=False)

    print("\n[Query 3] High-Volume Traffic Anomalies (Potential DoS / Flood Bursts):")
    spark.sql("""
        SELECT 
            attack_cat AS Threat_Category,
            proto AS Protocol,
            COUNT(*) AS Burst_Events,
            MAX(sbytes) AS Max_Bytes_Spike
        FROM cyber_logs
        WHERE sbytes > 50000
        GROUP BY attack_cat, proto
        ORDER BY Burst_Events DESC
        LIMIT 10
    """).show(truncate=False)

    spark.stop()
    print("\nETL & Distributed SQL Analytics Complete!")

if __name__ == "__main__":
    main()
