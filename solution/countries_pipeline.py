from pyspark.sql import SparkSession
import os
import pandas as pd
from pyspark.sql.functions import trim, upper

def main():
    # ✅ Build Spark session with Windows-safe configs
    spark = (
        SparkSession.builder
        .appName("CountriesHealthPipeline")
        .config("spark.hadoop.hadoop.native.io", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .config("mapreduce.fileoutputcommitter.algorithm.version", "2")
        .getOrCreate()
    )

    print("Spark Version:", spark.version)

    # ✅ Absolute input & output paths
    base_dir = r"C:\Users\Rupesh.shelar\data-engineer-test\datasets"
    input_dir = os.path.join(base_dir, "countries_health")
    output_path = os.path.join(base_dir, "solution", "output", "countries_health.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ✅ Read CSVs with pandas
    all_pdfs = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(".csv"):   # ✅ only CSV files
            fpath = os.path.join(input_dir, file)
            print(f"Reading {fpath}")
            pdf = pd.read_csv(fpath)
            pdf["SourceFile"] = file
            all_pdfs.append(pdf)

    if not all_pdfs:
        raise FileNotFoundError(f"No CSV files in {input_dir}")

    combined_pdf = pd.concat(all_pdfs, ignore_index=True)

    # ✅ Normalize column names
    combined_pdf.columns = [c.strip().replace(" ", "_").lower() for c in combined_pdf.columns]

    # ✅ Create Spark DataFrame
    df = spark.createDataFrame(combined_pdf)
    if "country" in df.columns:
        df = df.withColumn("country", trim(df["country"]))
        df = df.withColumn("CountryKey", upper(trim(df["country"])))

    # ✅ Show ALL rows in DataFrame
    print("\n=== All Countries Health Data ===")
    df.show(df.count(), truncate=False)

    # ✅ Write to a SINGLE parquet file
    df.coalesce(1).write.mode("overwrite").parquet(output_path)
    print(f"\nCountries Health dataset saved to {output_path} with {df.count()} rows (single file)")

    # ✅ Read back Parquet and show ALL rows
    print("\n=== All Data from Parquet Output ===")
    df_parquet = spark.read.parquet(output_path)
    df_parquet.show(df_parquet.count(), truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
