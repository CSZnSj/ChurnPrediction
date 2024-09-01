from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, to_timestamp
from pyspark.sql.types import TimestampType
from pyspark.sql.utils import AnalysisException, IllegalArgumentException
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
    try:
        spark = SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()
        return spark
    except Exception as e:
        print(f"Error creating Spark session: {e}")
        raise

def load_config(
        config_path: str = "../config.json") -> dict:
    """
    Loads configuration settings from a JSON file.

    Args:
        config_path (str): The path to the JSON configuration file. Defaults to "../config.json".

    Returns:
        dict: A dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If there is an error decoding the JSON file.
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError as e:
        print(f"Configuration file not found at path: {config_path}.")
        raise FileNotFoundError(f"File not found: {config_path}") from e
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file: {config_path}.")
        raise json.JSONDecodeError(f"JSON decode error: {config_path}") from e
    except Exception as e:
        print(f"Unexpected error loading configuration file: {config_path}.")
        raise

def get_parquet_path(config: dict, key: str, month: str) -> str:
    """
    Constructs the file path for a Parquet file based on configuration settings.

    Args:
        config (dict): Configuration dictionary.
        key (str): The key to access the parquet path template in the configuration.
        month (str): The month to be formatted into the path.

    Returns:
        str: The constructed file path for the Parquet file.

    Raises:
        KeyError: If the specified key is not found in the configuration.
    """
    try:
        # Access the path template from the config dictionary
        path_template = config["paths"]["parquet"][key]
        parquet_path = os.path.join("..", path_template.format(month=month))
        return parquet_path
    except KeyError as e:
        raise KeyError(f"Key error while accessing configuration: {e}") from e

def custom_read_parquet(
    spark: SparkSession, 
    config: dict, 
    key: str, 
    month: str
) -> DataFrame:
    """
    Reads a Parquet file into a Spark DataFrame based on configuration settings.

    Args:
        spark (SparkSession): An active Spark session object.
        config (dict): Configuration dictionary.
        key (str): The key to access the parquet path template in the configuration.
        month (str): The month to be formatted into the path.

    Returns:
        DataFrame: The loaded DataFrame from the Parquet file.

    Raises:
        ValueError: If the configuration key is invalid.
        TypeError: If the month parameter is not a string.
        FileNotFoundError: If the Parquet file does not exist.
        AnalysisException: If there is an issue with reading the Parquet file.
    """
    parquet_path = None

    try:
        parquet_path = get_parquet_path(config, key, month)
        df = spark.read.parquet(parquet_path)
        return df
    except FileNotFoundError as e:
        print(f"Parquet file not found at path: {parquet_path}.")
        raise FileNotFoundError(f"File not found: {parquet_path}") from e
    except AnalysisException as e:
        print(f"Error reading Parquet file at path: {parquet_path}.")
        raise
    except IllegalArgumentException as e:
        print(f"Illegal argument provided while reading Parquet file: {parquet_path}.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while reading Parquet file at path: {parquet_path}.")
        raise


def custom_to_timestamp(
        df: DataFrame, 
        col_name: str,
        format: str = "yyyyMMdd HH:mm:ss") -> DataFrame:
    """
    Ensures that the specified column in the DataFrame is of TimestampType.
    If the column is not already a TimestampType, it attempts to convert it 
    using the format 'yyyyMMdd HH:mm:ss'. Raises an error if conversion fails.

    Args:
        df: DataFrame containing the column to be converted.
        col_name: The name of the column to convert to TimestampType.
        format: The date format to use for conversion. Defaults to "yyyyMMdd HH:mm:ss".

    Returns:
        DataFrame: The DataFrame with the specified column converted to TimestampType.

    Raises:
        ValueError: If conversion fails.
    """
    if isinstance(df.schema[col_name].dataType, TimestampType):
        return df

    try:
        df = df.withColumn(col_name, to_timestamp(col(col_name), format))
    except Exception as e:
        print(f"Error converting column '{col_name}' to TimestampType: {e}")
        raise ValueError(f"Error converting column '{col_name}' to TimestampType: {e}")

    return df