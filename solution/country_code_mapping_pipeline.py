from pyspark.sql import SparkSession
from pyspark.sql.functions import trim, upper, col
import pycountry
import os

def init_spark(app_name="CountryCodeMappingPipeline"):
    """Initialize Spark session with safe configs."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.hadoop.hadoop.native.io", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .config("mapreduce.fileoutputcommitter.algorithm.version", "2")
        .getOrCreate()
    )

def build_mapping(spark):
    """Generate mapping table (countrycode → countryname) using pycountry."""
    mapping = [(c.alpha_3, c.name) for c in pycountry.countries]
    df = spark.createDataFrame(mapping, ["countrycode", "countryname"])
    df = df.withColumn("countrycode", upper(trim(col("countrycode"))))
    df = df.withColumn("countryname", trim(col("countryname")))
    return df.dropDuplicates(["countrycode"])

def save_outputs(df, base_dir):
    """Save mapping as both Parquet and CSV with consistent naming."""
    output_parquet = os.path.join(base_dir, "country_code_mapping.parquet")
    output_csv = os.path.join(base_dir, "country_code_mapping.csv")
    df.coalesce(1).write.mode("overwrite").parquet(output_parquet)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_csv)
    print(f" Country code mapping saved:\n- {output_parquet}\n- {output_csv}")

def main():
    spark = init_spark()
    base_dir = r"C:\Users\Rupesh.shelar\data-engineer-test\datasets\solution\output"
    os.makedirs(base_dir, exist_ok=True)

    df_mapping = build_mapping(spark)
    save_outputs(df_mapping, base_dir)

    df_mapping.show(20, truncate=False)
    spark.stop()

if __name__ == "__main__":
    main()
