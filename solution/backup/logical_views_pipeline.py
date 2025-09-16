from pyspark.sql import SparkSession
import os

def main():
    # 1. Spark Session
    spark = (
        SparkSession.builder
        .appName("LogicalViewsOnParquet")
        .getOrCreate()
    )

    # 2. Paths
    base_dir = r"C:\Users\Rupesh.shelar\data-engineer-test\datasets\solution\output"
    parquet_path = os.path.join(base_dir, "combined_olympics_countries.parquet")

    # Output directories
    medal_out = os.path.join(base_dir, "vw_medal_summary_csv")
    gdp_out = os.path.join(base_dir, "vw_gdp_medals_csv")
    region_out = os.path.join(base_dir, "vw_region_summary_csv")

    # 3. Load parquet
    df = spark.read.parquet(parquet_path)

    # Register base view
    df.createOrReplaceTempView("vw_combined")

    # === Logical Views ===

    # 1. Medal summary per country
    df_medal_summary = spark.sql("""
        SELECT Country,
               COUNT(*) AS total_events,
               SUM(Gold) AS total_gold,
               SUM(Silver) AS total_silver,
               SUM(Bronze) AS total_bronze
        FROM vw_combined
        GROUP BY Country
        ORDER BY total_gold DESC
    """)
    df_medal_summary.createOrReplaceTempView("vw_medal_summary")

    # Save to CSV
    df_medal_summary.coalesce(1).write.mode("overwrite").option("header", True).csv(medal_out)

    # 2. GDP vs Medals correlation view
    df_gdp_medals = spark.sql("""
        SELECT Country,
               AVG(gdp_per_capita) AS avg_gdp,
               SUM(Gold + Silver + Bronze) AS total_medals
        FROM vw_combined
        GROUP BY Country
        ORDER BY total_medals DESC
    """)
    df_gdp_medals.createOrReplaceTempView("vw_gdp_medals")

    # Save to CSV
    df_gdp_medals.coalesce(1).write.mode("overwrite").option("header", True).csv(gdp_out)

    # 3. Region wise Olympic performance
    df_region = spark.sql("""
        SELECT region,
               COUNT(DISTINCT Country) AS num_countries,
               SUM(Gold) AS gold_medals,
               SUM(Silver) AS silver_medals,
               SUM(Bronze) AS bronze_medals
        FROM vw_combined
        GROUP BY region
        ORDER BY gold_medals DESC
    """)
    df_region.createOrReplaceTempView("vw_region_summary")

    # Save to CSV
    df_region.coalesce(1).write.mode("overwrite").option("header", True).csv(region_out)

    # === Display Samples ===
    print("=== Medal Summary (Top 10 Countries) ===")
    df_medal_summary.show(10, truncate=False)

    print("=== GDP vs Medals (Top 10 Countries) ===")
    df_gdp_medals.show(10, truncate=False)

    print("=== Region-wise Summary ===")
    df_region.show(10, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
