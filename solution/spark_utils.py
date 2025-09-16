from pyspark.sql import SparkSession

def init_spark(app_name="DefaultPipeline"):
    """
    Initialize and return a SparkSession with safe configurations.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.hadoop.hadoop.native.io", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .getOrCreate()
    )
