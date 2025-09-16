from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.appName("ReadParquet").getOrCreate()

# Path to Parquet file
parquet_path = r"C:\Users\Rupesh.shelar\data-engineer-test\datasets\solution\output\olympics.parquet"

# Read Parquet file
df = spark.read.parquet(parquet_path)

# Display results
df.show(10)
df.printSchema()
print("Total rows:", df.count())
