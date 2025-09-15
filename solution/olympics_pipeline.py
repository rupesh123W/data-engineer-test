from pyspark.sql import SparkSession
import os
import re
import pandas as pd
from pyspark.sql.functions import trim
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

def main():
    # ✅ Build Spark session with Windows-safe configs
    spark = (
        SparkSession.builder
        .appName("OlympicsPipeline")
        .config("spark.hadoop.hadoop.native.io", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .config("mapreduce.fileoutputcommitter.algorithm.version", "2")
        .getOrCreate()
    )

    print("Spark Version:", spark.version)

    # ✅ Absolute input & output paths
    base_dir = r"C:\Users\Rupesh.shelar\data-engineer-test\datasets"
    input_dir = os.path.join(base_dir, "olympics")
    output_path = os.path.join(base_dir, "solution", "output", "olympics.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ✅ Read CSVs with pandas
    all_pdfs = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(".csv"):
            fpath = os.path.join(input_dir, file)
            print(f"Reading {fpath}")
            pdf = pd.read_csv(fpath)
            pdf["SourceFile"] = file
            match = re.search(r"(\d{4})", file)
            pdf["Year"] = int(match.group(1)) if match else None
            all_pdfs.append(pdf)

    if not all_pdfs:
        raise FileNotFoundError(f"No CSV files in {input_dir}")

    combined_pdf = pd.concat(all_pdfs, ignore_index=True)

    #  Schema definition
    schema = StructType([
        StructField("Country", StringType(), True),
        StructField("Gold", IntegerType(), True),
        StructField("Silver", IntegerType(), True),
        StructField("Bronze", IntegerType(), True),
        StructField("Total", IntegerType(), True),
        StructField("SourceFile", StringType(), True),
        StructField("Year", IntegerType(), True),
    ])

    #  Create Spark DataFrame
    df = spark.createDataFrame(combined_pdf, schema=schema)
    df = df.withColumn("Country", trim(df["Country"]))

    #  Show ALL rows in DataFrame
    print("\n=== All Data from CSVs ===")
    df.show(df.count(), truncate=False)

    #  Write to a SINGLE parquet file
    df.coalesce(1).write.mode("overwrite").parquet(output_path)
    print(f"\nOlympics dataset saved to {output_path} with {df.count()} rows (single file)")

    #  Read back Parquet and show ALL rows
    print("\n=== All Data from Parquet Output ===")
    df_parquet = spark.read.parquet(output_path)
    df_parquet.show(df_parquet.count(), truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
