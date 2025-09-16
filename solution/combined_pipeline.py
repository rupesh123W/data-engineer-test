import os, re, shutil, uuid, pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import trim, upper, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# ----------------------------
# Init Spark
# ----------------------------
def init_spark(app_name="CombinedPipeline"):
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.hadoop.hadoop.native.io", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .getOrCreate()
    )

# ----------------------------
# Save helper (Parquet + CSV, with fixed naming)
# ----------------------------
def save_outputs(df, output_parquet, output_csv, filename="output.csv"):
    # Parquet
    if os.path.exists(output_parquet):
        shutil.rmtree(output_parquet, ignore_errors=True)
    df.write.mode("overwrite").parquet(output_parquet)
    print(f"[INFO] Saved Parquet: {output_parquet}")

    # CSV
    temp_csv_dir = output_csv + "_tmp_" + str(uuid.uuid4()).replace("-", "")
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_csv_dir)

    csv_file = None
    for f in os.listdir(temp_csv_dir):
        if f.endswith(".csv"):
            csv_file = os.path.join(temp_csv_dir, f)
            break
    if os.path.exists(output_csv):
        shutil.rmtree(output_csv, ignore_errors=True)
    os.makedirs(output_csv, exist_ok=True)
    if csv_file:
        final_csv = os.path.join(output_csv, filename)
        shutil.move(csv_file, final_csv)
        print(f"[INFO] Saved CSV: {final_csv}")
    shutil.rmtree(temp_csv_dir, ignore_errors=True)

# ----------------------------
# Olympics schema
# ----------------------------
def olympics_schema():
    return StructType([
        StructField("countrycode", StringType(), True),
        StructField("Gold", IntegerType(), True),
        StructField("Silver", IntegerType(), True),
        StructField("Bronze", IntegerType(), True),
        StructField("Total", IntegerType(), True),
        StructField("Year", IntegerType(), True),
    ])

# ----------------------------
# Main Combined Pipeline
# ----------------------------
def main():
    spark = init_spark()
    today = datetime.today().strftime("%Y%m%d")

    base_dir = r"C:\Users\Rupesh.shelar\data-engineer-test\datasets"
    output_dir = os.path.join(base_dir, "solution", "output", today)
    os.makedirs(output_dir, exist_ok=True)

    # -------- Bronze (Raw Layer) --------
    # Countries raw
    countries_raw = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(os.path.join(base_dir, "countries", "countries of the world.csv"))
    )
    save_outputs(countries_raw, os.path.join(output_dir, "bronze_countries_parquet"),
                               os.path.join(output_dir, "bronze_countries_csv"),
                               "countries_raw.csv")

    # Olympics raw
    pdfs = []
    input_dir = os.path.join(base_dir, "olympics")
    for file in os.listdir(input_dir):
        if file.lower().endswith(".csv"):
            pdf = pd.read_csv(os.path.join(input_dir, file))
            match = re.search(r"(\d{4})", file)
            pdf["Year"] = int(match.group(1)) if match else None
            pdfs.append(pdf)
    olympics_raw_pdf = pd.concat(pdfs, ignore_index=True)
    olympics_raw = spark.createDataFrame(olympics_raw_pdf, schema=olympics_schema())
    save_outputs(olympics_raw, os.path.join(output_dir, "bronze_olympics_parquet"),
                               os.path.join(output_dir, "bronze_olympics_csv"),
                               "olympics_raw.csv")

    # ✅ FIXED: Country code mapping raw (from CSV instead of pycountry)
    mapping_file = os.path.join(base_dir, "country_code_mapping", "country_code_mapping.csv")
    if os.path.exists(mapping_file):
        mapping_raw = spark.read.option("header", True).csv(mapping_file)
    else:
        # fallback empty DataFrame if file missing
        mapping_raw = spark.createDataFrame([], "countrycode STRING, countryname STRING")

    save_outputs(mapping_raw, os.path.join(output_dir, "bronze_mapping_parquet"),
                                os.path.join(output_dir, "bronze_mapping_csv"),
                                "mapping_raw.csv")

    # -------- Silver (Staged Layer) --------
    # Clean Olympics
    olympics_silver = olympics_raw.withColumn("countrycode", trim(col("countrycode")))
    save_outputs(olympics_silver, os.path.join(output_dir, "silver_olympics_parquet"),
                                   os.path.join(output_dir, "silver_olympics_csv"),
                                   "olympics_silver.csv")

    # Clean Mapping
    mapping_silver = mapping_raw.withColumn("countrycode", upper(trim(col("countrycode")))) \
                                .withColumn("countryname", trim(col("countryname")))
    save_outputs(mapping_silver, os.path.join(output_dir, "silver_mapping_parquet"),
                                    os.path.join(output_dir, "silver_mapping_csv"),
                                    "mapping_silver.csv")

    # Clean Countries
    countries_silver = countries_raw
    for old in countries_silver.columns:
        new = old.strip().lower().replace(" ", "_").replace("%", "percent").replace("$", "usd")
        countries_silver = countries_silver.withColumnRenamed(old, new)
    countries_silver = countries_silver.withColumn("countryname", trim(col("country")))
    save_outputs(countries_silver, os.path.join(output_dir, "silver_countries_parquet"),
                                      os.path.join(output_dir, "silver_countries_csv"),
                                      "countries_silver.csv")

    # -------- Gold (Curated Layer) --------
    # Join Olympics + Mapping (to get full country names)
    curated = olympics_silver.join(mapping_silver, "countrycode", "inner")
    # Aggregate medals per country
    curated_final = curated.groupBy("countryname") \
        .agg({"Gold": "sum", "Silver": "sum", "Bronze": "sum", "Total": "sum"}) \
        .withColumnRenamed("sum(Gold)", "total_gold") \
        .withColumnRenamed("sum(Silver)", "total_silver") \
        .withColumnRenamed("sum(Bronze)", "total_bronze") \
        .withColumnRenamed("sum(Total)", "total_medals")

    save_outputs(curated_final, os.path.join(output_dir, "gold_curated_parquet"),
                                    os.path.join(output_dir, "gold_curated_csv"),
                                    "curated_gold.csv")

    print(f"✅ Combined pipeline completed. Outputs stored under {output_dir}")
    spark.stop()

if __name__ == "__main__":
    main()
