from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import input_file_name, regexp_extract, trim
import os

def main():
    # Start Spark session
    spark = SparkSession.builder.appName("OlympicsPipeline").getOrCreate()

    # Input and output paths
    input_path = "datasets/olympics/*.csv"
    output_path = "solution/output/olympics.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define schema
    schema = StructType([
        StructField("Country", StringType(), True),
        StructField("Gold", IntegerType(), True),
        StructField("Silver", IntegerType(), True),
        StructField("Bronze", IntegerType(), True),
        StructField("Total", IntegerType(), True)
    ])

    # Read CSV files with schema and capture filename
    df = (spark.read
                .option("header", True)
                .schema(schema)
                .csv(input_path)
                .withColumn("filename", input_file_name()))

    # Extract year from filename
    df = df.withColumn("Year", regexp_extract("filename", r'(\d{4})', 1))

    # Clean country names
    df = df.withColumn("Country", trim(df["Country"]))

    # Show preview
    df.show(10, truncate=False)

    # Save as Parquet (overwrite mode)
    df.write.mode("overwrite").parquet(output_path)
    print(f"Olympics dataset saved to {output_path} with {df.count()} rows.")

    spark.stop()

if __name__ == "__main__":
    main()
