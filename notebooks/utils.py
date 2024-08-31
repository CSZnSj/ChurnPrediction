from pyspark.sql import SparkSession, DataFrame
import json
import os



def create_spark_session(
        app_name: str = "Basics") -> SparkSession:
    """
    Creates and returns a Spark session with the given application name.

    Args:
        app_name (str): The name of the Spark application. Defaults to "Basics".

    Returns:
        SparkSession: An active Spark session object.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_config(
        config_path: str = "../config.json") -> dict:
    """
    Loads configuration settings from a JSON file.

    Args:
        config_path (str): The path to the JSON configuration file. Defaults to "../config.json".

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def get_parquet_path(
        config: dict, 
        key: str, 
        month: int) -> str:
    """
    Constructs the file path for a Parquet file based on configuration settings.

    Args:
        config (dict): Configuration dictionary.
        key (str): The key to access the parquet path template in the configuration.
        month (str): The month to be formatted into the path.

    Returns:
        str: The constructed file path for the Parquet file.
    """
    path_template = config["paths"]["parquet"][key]
    parquet_path = os.path.join("..", path_template.format(month=month))
    return parquet_path

def custom_read_parquet(
        spark: SparkSession, 
        config: dict, 
        key: str, 
        month: int) -> DataFrame:
    """
    Reads a Parquet file into a Spark DataFrame based on configuration settings.

    Args:
        spark (SparkSession): An active Spark session object.
        config (dict): Configuration dictionary.
        key (str): The key to access the parquet path template in the configuration.
        month (str): The month to be formatted into the path.

    Returns:
        DataFrame: The loaded DataFrame from the Parquet file.
    """
    parquet_path = get_parquet_path(config, key, month)
    return spark.read.parquet(parquet_path)
