from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import os

def main():
    spark = SparkSession.builder.appName("CountriesPipeline").getOrCreate()

    input_path = "datasets/countries/countries of the world.csv"
    output_path = "solution/output/countries.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define schema (simplified to main columns)
    schema = StructType([
        StructField("Country", StringType(), True),
        StructField("Population", IntegerType(), True),
        StructField("Area", IntegerType(), True),
        StructField("GDP", DoubleType(), True),
        StructField("Literacy", DoubleType(), True),
        StructField("Phones", DoubleType(), True),
        StructField("Birthrate", DoubleType(), True),
        StructField("Deathrate", DoubleType(), True)
    ])

    # Read CSV
    df = spark.read.option("header", True).schema(schema).csv(input_path)

    # Clean country names
    df = df.withColumn("Country", df["Country"].trim())

    # Show preview
    df.show(10, truncate=False)

    # Save as Parquet
    df.write.mode("overwrite").parquet(output_path)
    print(f"Countries dataset saved to {output_path} with {df.count()} rows.")

    spark.stop()

if __name__ == "__main__":
    main()
