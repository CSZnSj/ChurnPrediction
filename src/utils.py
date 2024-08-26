from pyspark.sql import SparkSession
import json


def create_spark_session(logger, app_name):
    """
    Create and return a Spark session.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    logger.info(f'{app_name} spark is initiated')
    return spark

def load_config(logger, config_path="config.json"):
    """
    Load configuration from a JSON file.

    :param config_path: Path to the JSON configuration file.
    :return: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    logger.info(f"{config_path} is read")
    return config