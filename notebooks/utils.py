from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import to_timestamp, col, max, min, sum, avg, count, hour, dayofweek, dayofmonth, when
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional


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

def count_nulls(
        df: DataFrame, 
        col_name: str) -> DataFrame:
    """
    Counts the number of null values in a specified column of the given DataFrame.

    Args:
        df (DataFrame): The input Spark DataFrame.
        col_name (str): The name of the column in which to count null values.

    Returns:
        DataFrame: A DataFrame containing a single column with the count of null values, aliased as 'null_count'.
    """
    # Compute the count of null values in the specified column
    return df.filter(F.col(col_name).isNull()).count()

def custom_count_distinct(
        df: DataFrame, 
        col_name: str) -> DataFrame:
    """
    Computes the count of distinct values for a specified column in the given DataFrame.

    Args:
        df (DataFrame): The input Spark DataFrame.
        col_name (str): The name of the column for which distinct values count is to be computed.

    Returns:
        DataFrame: A DataFrame containing a single column with the count of distinct values, aliased as 'distinct_count'.
    """
    # Compute the count of distinct values in the specified column
    return df.agg(F.countDistinct(F.col(col_name)).alias(f'{col_name}_distinct_count')).collect()[0][0]


def custom_to_timestamp(
        df: DataFrame, 
        col_name: str) -> DataFrame:
    """
    Converts a column with a string date-time format to a timestamp format in a DataFrame.

    Args:
        df (DataFrame): The input Spark DataFrame.
        col_name (str): The name of the column to convert to a timestamp.

    Returns:
        DataFrame: The DataFrame with the specified column converted to a timestamp.
    """
    return df.withColumn(col_name, to_timestamp(col(col_name), "yyyyMMdd HH:mm:ss"))

def custom_max(
        df: DataFrame, 
        col_name: str) -> DataFrame:
    """
    Computes the maximum value of a specified column in a DataFrame.

    Args:
        df (DataFrame): The input Spark DataFrame.
        col_name (str): The name of the column for which to compute the maximum value.

    Returns:
        DataFrame: A DataFrame containing the maximum value of the specified column.
    """
    return df.select(max(col(col_name))).collect()[0][0]

def custom_min(
        df: DataFrame, 
        col_name: str) -> DataFrame:
    """
    Computes the minimum value of a specified column in a DataFrame.

    Args:
        df (DataFrame): The input Spark DataFrame.
        col_name (str): The name of the column for which to compute the minimum value.

    Returns:
        DataFrame: A DataFrame containing the minimum value of the specified column.
    """
    return df.select(min(col(col_name))).collect()[0][0]


def custom_group_by(
        df: DataFrame, 
        group_by_col: str, 
        agg_col: str, 
        agg_func: str, 
        alias_name: str, 
        sort_order: str = None) -> DataFrame:
    """
    Groups the DataFrame by the specified column and applies the specified aggregation function. 
    Optionally, assigns an alias to the result column and sorts the result.

    Args:
        df (DataFrame): The input Spark DataFrame.
        group_by_col (str): The name of the column to group by.
        agg_col (str): The name of the column to aggregate.
        agg_func (str): The aggregation function to apply (e.g., 'sum', 'avg', 'count').
        alias_name (str, optional): The alias name for the aggregated column. Defaults to None.
        sort_order (str, optional): The sorting order; 'asc' for ascending, 'desc' for descending, or None for no sorting.

    Returns:
        DataFrame: A new DataFrame with the grouped, aggregated, and optionally sorted results.

    Raises:
        ValueError: If the aggregation function is not recognized.
    """
    # Define a mapping from function names to actual PySpark functions
    agg_functions = {
        'sum': F.sum,
        'avg': F.avg,
        'count': F.count,
        'count_distinct': F.countDistinct,
        'max': F.max,
        'min': F.min
    }

    # Get the aggregation function from the dictionary
    agg_function = agg_functions.get(agg_func, None)
    if agg_function is None:
        raise ValueError(f"Aggregation function '{agg_func}' is not recognized. Ensure it is a valid PySpark function.")

    # Apply the aggregation function
    aggregated_df = df.groupBy(group_by_col).agg(agg_function(agg_col).alias(alias_name))
    
    # Apply sorting based on sort_order
    if sort_order:
        ascending = True if sort_order == 'asc' else False
        aggregated_df = aggregated_df.orderBy(col(alias_name), ascending=ascending)
    
    return aggregated_df


import os
import matplotlib.pyplot as plt
from typing import Optional

def save_fig(
        output_dir: str, 
        name: str, 
        file_format: str = "png") -> None:
    """
    Saves the current figure to a specified directory with a given name and file format.

    Args:
        output_dir (str): The directory where the figure will be saved.
        name (str): The name of the file (without extension) for the saved figure.
        file_format (str, optional): The file format to use for saving the figure. Defaults to "png".

    Raises:
        ValueError: If an unsupported file format is specified.

    Returns:
        None
    """
    # Validate file format
    supported_formats = ["png", "jpg", "jpeg", "pdf", "svg"]
    if file_format not in supported_formats:
        raise ValueError(f"Unsupported file format '{file_format}'. Supported formats are: {supported_formats}.")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct file path
    file_path = os.path.join(output_dir, f"{name}.{file_format}")

    # Save the figure
    plt.savefig(file_path, format=file_format)
    plt.close()

    # Log the success message
    print(f"Figure saved to: {file_path}")

def plot_column_distribution(
        col_df: DataFrame, 
        plot_type: str,
        month: str,
        x: str, 
        y: str = None, 
        figsize: tuple = (12, 8), 
        output_dir: str = "output") -> None:
    """
    Plots the distribution of a specified column in a DataFrame using various plot types.

    Args:
        col_df (pd.DataFrame): The DataFrame containing the column to plot.
        col_name (str): The name of the column to plot.
        plot_type (str): The type of plot to create ('box', 'hist', 'violin', 'count').
        figsize (tuple, optional): Figure size for the plot. Defaults to (12, 8).

    Returns:
        None
    """
    # Check if the plot type is supported
    supported_plot_types = {
        "count": sns.countplot,
        "line": sns.lineplot,
        "box": sns.boxplot,
        "hist": sns.histplot,
        "violin": sns.violinplot,
    }
    
    if plot_type not in supported_plot_types:
        raise ValueError(f"Unsupported plot type '{plot_type}'. Supported types are: {list(supported_plot_types.keys())}.")

    # Convert the column DataFrame to Pandas
        # Convert PySpark DataFrame to Pandas DataFrame if necessary
    if isinstance(col_df, DataFrame):
        df = col_df.toPandas()
    elif isinstance(col_df, pd.DataFrame):
        df = col_df
    else:
        raise TypeError("The provided DataFrame is neither a PySpark DataFrame nor a Pandas DataFrame.")

    # Set up the plot
    plt.figure(figsize=figsize)

    plot_func = supported_plot_types[plot_type]
    if y:
        plot_func(data=df, x=x, y=y)
    else:
        plot_func(data=df, x=x)

    name = f"{plot_type.capitalize()}Plot_of_{x}__month_{month}"
    plt.xticks(rotation=90)
    plt.title(name)

    save_fig(output_dir, name)


def plot_aggregated_by_time(
        df: DataFrame, 
        timestamp_column: str, 
        agg_col : str, 
        month: str, 
        output_dir : str = "output") -> None:
    """
    Aggregates loan amounts based on different time components and plots the results.

    Args:
        dataframe (DataFrame): A PySpark DataFrame containing the data to be aggregated.
        timestamp_column (str): The name of the timestamp column to be used for aggregation.
        month (str): A string representing the month, used for naming output files.

    Returns:
        None
    """
    
    # Define keys for time aggregation and corresponding functions
    time_keys = {
        'Hour': hour,
        'DayOfWeek': dayofweek,
        'DayOfMonth': dayofmonth
    }

    # Add time-based columns to the DataFrame
    for key, time_function in time_keys.items():
        df = df.withColumn(key, time_function(col(timestamp_column)))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each key for aggregation and plotting
    for key in time_keys.keys():

        # Aggregate the loan amounts by the current time key
        aggregated_df = (
            df.groupBy(key)
            .agg(sum(agg_col).alias(f"sum_{agg_col}_{month}"))
            .toPandas()
        )

        # Prepare the DataFrame for plotting
        aggregated_df.reset_index(drop=True, inplace=True)
        name = f"SumAggregated_{agg_col}_by_{key}"
        aggregated_df[name] = aggregated_df.index

        plot_column_distribution(col_df=aggregated_df, plot_type="line", month=month, x=name, y=f"sum_{agg_col}_{month}")

def categorize_loan_amount(
    df: DataFrame,
    col_name: str,
    bins: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    new_column_name: str = "loan_category"
) -> DataFrame:
    """
    Categorizes a numeric column in a PySpark DataFrame into bins with descriptive labels.

    Args:
        df (DataFrame): The input PySpark DataFrame.
        column_name (str): The name of the column to categorize.
        bins (List[float], optional): List of bin edges. Should be sorted in ascending order and include -inf and inf.
            Defaults to predefined bins if not provided.
        labels (List[str], optional): List of labels corresponding to the bins. Should have one less element than bins.
            Defaults to predefined labels if not provided.
        new_column_name (str, optional): The name of the new column that will contain the category labels.
            Defaults to "loan_category".

    Returns:
        DataFrame: A new DataFrame with the added category column.

    Raises:
        ValueError: If the number of bins does not match the number of labels + 1, or if no bins or labels are provided.
    """
    # Default bin edges and labels if not provided
    if bins is None:
        bins = [float('-inf'), 5000, 10000, 20000, 50000, 100000, float('inf')]
    if labels is None:
        labels = ["VeryLow", "Low : 5k-10k", "Medium : 10k-20k", "High : 20k-50k", "VeryHigh : 50k-100k", "Extreme"]

    # Validate bin and label lengths
    if len(bins) != len(labels) + 1:
        raise ValueError("The number of bins must be one more than the number of labels.")
    if not bins or not labels:
        raise ValueError("Bins and labels cannot be empty.")

    # Start with a column containing the original values
    categorized_df = df.withColumn(new_column_name, col(col_name))

    # Apply categorization logic
    for i in range(1, len(bins)):
        categorized_df = categorized_df.withColumn(
            new_column_name,
            when((col(col_name) > bins[i - 1]) & (col(col_name) <= bins[i]), labels[i - 1])
            .otherwise(col(new_column_name))
        )

    return categorized_df


