import os, re, pandas as pd
from pyspark.sql.functions import trim
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from spark_utils import init_spark                      # reusable Spark init
from data_utils import save_outputs, data_quality_checks, ensure_output_dir  #  reusable utilities

# ----------------------------
# Define schema for Olympics data
# ----------------------------
def olympics_schema():
    """
    Returns a StructType schema for Olympics dataset.
    Ensures data types are consistent across all files.
    """
    return StructType([
        StructField("countrycode", StringType(), True),   # 3-letter country/NOC code
        StructField("Gold", IntegerType(), True),         # number of gold medals
        StructField("Silver", IntegerType(), True),       # number of silver medals
        StructField("Bronze", IntegerType(), True),       # number of bronze medals
        StructField("Total", IntegerType(), True),        # total medals
        StructField("Year", IntegerType(), True),         # extracted year from filename
    ])

# ----------------------------
# Read and combine all Olympics CSVs
# ----------------------------
def load_and_combine_csvs(input_dir):
    """
    Reads all CSV files from input_dir, adds 'Year' column
    (extracted from filename), and returns a combined pandas DataFrame.
    """
    all_csvs = []
    for file in os.listdir(input_dir):                      # iterate all files in input dir
        if file.lower().endswith(".csv"):                   # process only CSV files
            fpath = os.path.join(input_dir, file)           # build full path
            print(f"Reading {fpath}")                       # log file being read
            pdf = pd.read_csv(fpath)                        # load with pandas (fast for small CSVs)
            match = re.search(r"(\d{4})", file)             # regex to extract year from filename
            pdf["Year"] = int(match.group(1)) if match else None  # add Year column
            all_csvs.append(pdf)                            # collect into list

    if not all_csvs:                                        # no CSVs found
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    return pd.concat(all_csvs, ignore_index=True)           # combine all into single DataFrame

# ----------------------------
# Main Pipeline
# ----------------------------
def main():
    # 1. Initialize Spark (reusable function from spark_utils)
    spark = init_spark("OlympicsPipeline")

    # 2. Define input and output paths
    base_dir = r"C:\Users\Rupesh.shelar\data-engineer-test\datasets"
    input_dir = os.path.join(base_dir, "olympics")
    output_parquet = os.path.join(base_dir, "solution", "output", "olympics.parquet")
    output_csv = os.path.join(base_dir, "solution", "output", "olympics.csv")
    ensure_output_dir(output_parquet)                      # create dir if not exists

    # 3. Load and combine CSVs into pandas DataFrame
    combined_pdf = load_and_combine_csvs(input_dir)

    # 4. Convert pandas DataFrame → Spark DataFrame with schema
    df = spark.createDataFrame(combined_pdf, schema=olympics_schema())

    # 5. Clean 'countrycode' column (remove spaces, normalize)
    df = df.withColumn("countrycode", trim(df["countrycode"]))

    # 6. Show full dataset in console
    print("\n=== Olympics Data (Full) ===")
    df.show(df.count(), truncate=False)

    # 7. Run reusable data quality checks (from data_utils)
    data_quality_checks(df, key_columns=["countrycode", "Year"])

    # 8. Save outputs to both Parquet + CSV (from data_utils)
    save_outputs(df, output_parquet, output_csv)

    # 9. Validate saved outputs by reading back
    print("\n=== Data from Parquet ===")
    spark.read.parquet(output_parquet).show(truncate=False)

    print("\n=== Data from CSV ===")
    spark.read.option("header", True).csv(output_csv).show(truncate=False)

    # 10. Stop Spark session cleanly
    spark.stop()

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    main()
