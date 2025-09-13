from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

def main():
    spark = SparkSession.builder.appName("CombinedPipeline").getOrCreate()

    # Load previous outputs
    olympics = spark.read.parquet("solution/output/olympics.parquet")
    countries = spark.read.parquet("solution/output/countries.parquet")

    # Add artificial keys for join
    olympics = olympics.withColumn("country_id", monotonically_increasing_id())
    countries = countries.withColumn("country_id", monotonically_increasing_id())

    # Join datasets
    combined = olympics.join(countries, "country_id", "inner")

    # Save combined dataset
    combined_output = "solution/output/combined.parquet"
    combined.write.mode("overwrite").parquet(combined_output)

    print(f"Combined dataset saved to {combined_output} with {combined.count()} rows.")

    spark.stop()

if __name__ == "__main__":
    main()
