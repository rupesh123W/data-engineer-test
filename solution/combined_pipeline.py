import os
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, regexp_replace, isnan

# ----------------------------
# Load Config
# ----------------------------
def load_config(config_file="config.json"):
    """
    Load the configuration from a JSON file.
    This helps keep configuration settings flexible and avoids hardcoding paths.
    """
    with open(config_file, 'r') as file:
        return json.load(file)

# Load configuration
config = load_config()  # Adjust the config file path if necessary
PROJECT_ROOT = config.get("PROJECT_ROOT")

if PROJECT_ROOT is None:
    raise ValueError("PROJECT_ROOT is not set in the config file")

TODAY = datetime.today().strftime("%Y%m%d")

# Directories for each layer (Bronze, Silver, Gold)
BRONZE_DIR = os.path.join(PROJECT_ROOT, "datasets", "solution", "output", TODAY, "bronze")
SILVER_DIR = os.path.join(PROJECT_ROOT, "datasets", "solution", "output", TODAY, "silver")
GOLD_DIR = os.path.join(PROJECT_ROOT, "datasets", "solution", "output", TODAY, "gold")

# Paths for Bronze Layer (CSV files only)
BRONZE_OLYMPICS = os.path.join(BRONZE_DIR, "olympics", f"olympics_{TODAY}.csv")
BRONZE_MAPPING = os.path.join(BRONZE_DIR, "mapping", f"mapping_{TODAY}.csv")
BRONZE_COUNTRIES = os.path.join(BRONZE_DIR, "countries", f"countries_{TODAY}.csv")

# ----------------------------
# Silver Paths
# ----------------------------
SILVER_OUT = os.path.join(SILVER_DIR, f"olympics_country_mapping_{TODAY}.csv")

# ----------------------------
# Init Spark
# ----------------------------
def init_spark():
    """
    Initializes and returns a SparkSession.
    This is where we configure the Spark session with necessary settings.
    """
    return (
        SparkSession.builder
        .appName("Olympics-Data-Processing")
        .config("spark.hadoop.hadoop.native.io", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .getOrCreate()
    )

# ----------------------------
# Data Quality Checks (Silver Layer)
# ----------------------------
def data_quality_checks(df):
    """
    Perform data quality checks on the DataFrame (Silver Layer):
    1. Null / blank values
    2. Duplicates
    3. Numeric validation
    """
    print("\n=== Data Quality Checks ===")

    # 1. Null / blank values
    for col_name, dtype in df.dtypes:
        if dtype == "string":
            null_count = df.filter(
                (col(col_name).isNull()) | (trim(col(col_name)) == "")
            ).count()
            if null_count > 0:
                print(f"[WARN] Column '{col_name}' has {null_count} null/blank values")
    
    # 2. Duplicate checks (based on countrycode and countryname)
    dup_count = df.count() - df.distinct().count()
    if dup_count > 0:
        print(f"[WARN] Found {dup_count} duplicate rows based on countrycode and countryname")
    
    # 3. Numeric validation (example: check for non-numeric values in 'population')
    numeric_cols = ["population"]  # Modify with actual numeric column names
    for col_name in numeric_cols:
        invalid_count = df.filter(isnan(col(col_name)) | col(col_name).isNull()).count()
        if invalid_count > 0:
            print(f"[WARN] Column '{col_name}' has {invalid_count} invalid numeric values")

# ----------------------------
# Main
# ----------------------------
def main():
    """
    Main function to run the Spark job.
    Handles loading data, applying transformations, and saving data in different layers.
    """
    spark = init_spark()

    # ----------------------------
    # BRONZE LAYER (Just loading data)
    # ----------------------------
    print("===== BRONZE LAYER =====")

    # Step 1: Load Bronze data (Olympics, Mapping, Countries)
    bronze_olympics = spark.read.option("header", "true").csv(BRONZE_OLYMPICS)
    bronze_mapping = spark.read.option("header", "true").csv(BRONZE_MAPPING)
    bronze_countries = spark.read.option("header", "true").csv(BRONZE_COUNTRIES)

    print("Olympics rows (Bronze)  ->", bronze_olympics.count())
    print("Mapping rows (Bronze)   ->", bronze_mapping.count())
    print("Countries rows (Bronze) ->", bronze_countries.count())

    # Step 2: Join Olympics with Mapping (NOC → countrycode)
    olympics_with_code = bronze_olympics.join(
        bronze_mapping,
        bronze_olympics["NOC"] == bronze_mapping["countrycode"],  # Correct join condition
        "left"
    )

    # Step 3: Join with Countries (countryname → Country)
    olympics_country_mapping = olympics_with_code.join(
        bronze_countries,
        olympics_with_code["countryname"] == bronze_countries["Country"],
        "left"
    )

    # Step 4: Select only the desired columns (countrycode and countryname)
    final_bronze = olympics_country_mapping.select("countrycode", "countryname", "year")

    # Step 5: Save Bronze output (CSV only)
    final_bronze.write.mode("overwrite").option("header", True).csv(BRONZE_DIR)

    print(f"[BRONZE] Saved country mapping data -> {BRONZE_DIR}")

    # ----------------------------
    # SILVER LAYER (Apply cleaning and quality checks)
    # ----------------------------
    print("===== SILVER LAYER =====")

    # Step 6: Clean Column Names for Silver Layer (Remove special characters)
    silver_data = final_bronze.select([regexp_replace(col(c), "[^a-zA-Z0-9_]", "_").alias(c) for c in final_bronze.columns])

    # Step 7: Trim whitespace from all columns in Silver Layer
    silver_data = silver_data.select([trim(col(c)).alias(c) for c in silver_data.columns])

    # Step 8: Sort Silver data by year
    silver_data_sorted = silver_data.orderBy("year")

    # Step 9: Perform Data Quality Checks on Silver data
    data_quality_checks(silver_data_sorted)

    # Step 10: Save Silver output (CSV only)
    silver_data_sorted.write.mode("overwrite").option("header", True).csv(SILVER_OUT)

    print(f"[SILVER] Saved country mapping data -> {SILVER_OUT}")

    # ----------------------------
    # GOLD LAYER
    # ----------------------------
    print("===== GOLD LAYER =====")

    # Step 11: Save Gold output (CSV only)
    gold_output_path = os.path.join(GOLD_DIR, f"olympics_country_mapping_gold_{TODAY}.csv")
    silver_data_sorted.write.mode("overwrite").option("header", True).csv(gold_output_path)

    print(f"[GOLD] Saved final gold data -> {gold_output_path}")

    # Step 12: Show preview of Gold data
    silver_data_sorted.show(10, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
