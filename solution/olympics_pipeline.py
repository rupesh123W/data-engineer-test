import os
import re
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import trim
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

def main():
    # Start Spark session
    spark = SparkSession.builder.appName("OlympicsPipeline").getOrCreate()

    # Absolute input & output paths
    base_dir = "C:/Users/Rupesh.shelar/data-engineer-test/datasets"
    input_dir = os.path.join(base_dir, "olympics")
    output_path = os.path.join(base_dir, "solution/output/olympics.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect Excel files with pandas
    all_pdfs = []
    for file in os.listdir(input_dir):
        if file.endswith(".xlsx"):
            fpath = os.path.join(input_dir, file)
            print(f"Reading: {fpath}")
            pdf = pd.read_excel(fpath)

            # Add metadata: Source file + Year from filename
            pdf["SourceFile"] = file
            match = re.search(r"(\d{4})", file)
            pdf["Year"] = int(match.group(1)) if match else None

            all_pdfs.append(pdf)

    if not all_pdfs:
        raise FileNotFoundError(f"No Excel files found in {input_dir}")

    # Merge all DataFrames
    combined_pdf = pd.concat(all_pdfs, ignore_index=True)

    # Define schema
    schema = StructType([
        StructField("Country", StringType(), True),
        StructField("Gold", IntegerType(), True),
        StructField("Silver", IntegerType(), True),
        StructField("Bronze", IntegerType(), True),
        StructField("Total", IntegerType(), True),
        StructField("SourceFile", StringType(), True),
        StructField("Year", IntegerType(), True)
    ])

    # Convert pandas → Spark
    df = spark.createDataFrame(combined_pdf, schema=schema)

    # Clean country names
    df = df.withColumn("Country", trim(df["Country"]))

    # Preview
    df.show(20, truncate=False)

    # Save parquet
    df.write.mode("overwrite").parquet(output_path)
    print(f"Olympics dataset saved to {output_path} with {df.count()} rows.")

    spark.stop()

if __name__ == "__main__":
    main()
