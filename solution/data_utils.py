import os
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, trim, isnan, expr


def safe_col(cname: str):
    """Always wrap column names with backticks for Spark-safe usage."""
    return expr(f"`{cname}`")


# ----------------------------
# Data Quality Checks
# ----------------------------
def data_quality_checks(df: DataFrame, key_columns=None, numeric_cols=None) -> None:
    """
    Run data quality checks on a Spark DataFrame:
    - Null/blank values
    - Duplicate check
    - Numeric validation
    """
    print("\n=== Data Quality Checks ===")

    # 1. Null / blank values
    for col_name, dtype in df.dtypes:
        if dtype == "string":
            null_count = df.filter(
                safe_col(col_name).isNull() | (trim(safe_col(col_name)) == "")
            ).count()
            if null_count > 0:
                print(f"[WARN] Column '{col_name}' has {null_count} null/blank values")

    # 2. Duplicate checks
    if key_columns:
        dup_count = df.count() - df.dropDuplicates(key_columns).count()
        if dup_count > 0:
            print(f"[WARN] Found {dup_count} duplicate rows based on {key_columns}")
        else:
            print(f"[INFO] No duplicates found on keys {key_columns}")

    # 3. Numeric validation
    if numeric_cols:
        for col_name in numeric_cols:
            if col_name in df.columns:
                nulls = df.filter(
                    safe_col(col_name).isNull() | isnan(safe_col(col_name))
                ).count()
                if nulls > 0:
                    print(f"[WARN] Numeric column '{col_name}' has {nulls} NULL/NaN values")
                else:
                    print(f"[INFO] Numeric column '{col_name}' passed validation")

    print("=== Data Quality Checks Completed ===\n")


# ----------------------------
# Save Outputs
# ----------------------------
def save_outputs(df: DataFrame, output_parquet: str, output_csv: str) -> None:
    """Save DataFrame into Parquet and CSV formats."""
    df.write.mode("overwrite").parquet(output_parquet)
    print(f"[INFO] Saved DataFrame to Parquet: {output_parquet}")

    df.write.mode("overwrite").option("header", True).csv(output_csv)
    print(f"[INFO] Saved DataFrame to CSV: {output_csv}")


# ----------------------------
# Ensure Output Directory
# ----------------------------
def ensure_output_dir(path: str) -> None:
    """Ensure that output directory exists for given path."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"[INFO] Created directory: {directory}")
    else:
        print(f"[INFO] Directory already exists: {directory}")
