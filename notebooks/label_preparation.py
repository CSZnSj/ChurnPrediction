from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, max as spark_max, min as spark_min, to_timestamp, datediff, when, greatest, least
from pyspark.sql.types import TimestampType
from utils import custom_read_parquet, custom_to_timestamp

def read_required_data(
    spark: SparkSession, 
    config: dict, 
    current_month: str, 
    next_month: str
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Reads the required parquet files for the current and next month.

    :param spark: SparkSession object.
    :param config: Configuration dictionary containing file paths or other relevant parameters.
    :param current_month: Identifier for the current month data.
    :param next_month: Identifier for the next month data.
    :return: A tuple containing DataFrames for current and next month package and recharge data.
    """
    current_package = custom_read_parquet(spark=spark, config=config, key="package", month=current_month)
    current_recharge = custom_read_parquet(spark=spark, config=config, key="recharge", month=current_month)
    next_package = custom_read_parquet(spark=spark, config=config, key="package", month=next_month)
    next_recharge = custom_read_parquet(spark=spark, config=config, key="recharge", month=next_month)
    
    return current_package, current_recharge, next_package, next_recharge

def find_time_activity_of_user(
    package_df: DataFrame, 
    recharge_df: DataFrame, 
    aggregation_func_str: str
) -> DataFrame:
    """
    Computes aggregated activity dates (maximum or minimum) for users based on package and recharge data.

    :param package_df: DataFrame containing package data with 'bib_id' and 'activation_date' columns.
    :param recharge_df: DataFrame containing recharge data with 'bib_id' and 'recharge_dt' columns.
    :param aggregation_func_str: Aggregation function to apply; should be either 'max' or 'min'.
    :return: DataFrame with 'bib_id' and the aggregated date as either 'max_date' or 'min_date'.
    :raises ValueError: If the provided aggregation function is not 'max' or 'min'.
    """
    # Convert date columns to TimestampType
    package_df = custom_to_timestamp(package_df, "activation_date")
    recharge_df = custom_to_timestamp(recharge_df, "recharge_dt")

    # Define the aggregation function mapping
    func_mapping = {"max": spark_max, "min": spark_min}
    
    if aggregation_func_str not in func_mapping:
        raise ValueError("Invalid aggregation function. Expected 'max' or 'min'.")
    
    aggregate_func = func_mapping[aggregation_func_str]
    combine_func = greatest if aggregation_func_str == "max" else least

    # Aggregate dates for packages and recharges
    package_agg = package_df.groupBy("bib_id").agg(
        aggregate_func(col("activation_date")).alias("package_date")
    )
    recharge_agg = recharge_df.groupBy("bib_id").agg(
        aggregate_func(col("recharge_dt")).alias("recharge_date")
    )

    # Join aggregated results
    combined_df = package_agg.join(recharge_agg, on="bib_id", how="outer")

    # Compute the final aggregated date
    combined_df = combined_df.withColumn(
        f"{aggregation_func_str}_date",
        combine_func(col("package_date"), col("recharge_date"))
    )
    
    return combined_df.select("bib_id", f"{aggregation_func_str}_date")

def prepare_churn_label(
    current_package_df: DataFrame, 
    current_recharge_df: DataFrame, 
    next_package_df: DataFrame, 
    next_recharge_df: DataFrame
) -> DataFrame:
    """
    Determines churn labels by comparing the latest and earliest activity dates across periods.

    :param current_package_df: DataFrame with package data for the current period.
    :param current_recharge_df: DataFrame with recharge data for the current period.
    :param next_package_df: DataFrame with package data for the next period.
    :param next_recharge_df: DataFrame with recharge data for the next period.
    :return: DataFrame with 'bib_id' and a churn label (1 for churned, 0 for not churned).
    """
    # Compute maximum dates for the current period
    latest_activity_current = find_time_activity_of_user(current_package_df, current_recharge_df, aggregation_func_str="max")
    
    # Compute minimum dates for the next period
    earliest_activity_next = find_time_activity_of_user(next_package_df, next_recharge_df, aggregation_func_str="min")

    # Join the DataFrames on 'bib_id'
    df = latest_activity_current.join(earliest_activity_next, on="bib_id", how="left")

    # Calculate date difference and determine churn label
    df = df.withColumn(
        "date_diff",
        datediff(col("min_date"), col("max_date"))
    ).withColumn(
        "label",
        when(
            (col("min_date").isNull() & col("max_date").isNotNull()) | (col("date_diff") > 30),
            1
        ).otherwise(0)
    )

    return df.select("bib_id", "label")

def generate_churn_label(
    spark: SparkSession, 
    config: dict, 
    current_month: str, 
    next_month: str
) -> DataFrame:
    """
    Generates churn labels by reading data for the current and next month and determining churn status.

    :param spark: SparkSession object.
    :param config: Configuration dictionary containing file paths or other relevant parameters.
    :param current_month: Identifier for the current month data.
    :param next_month: Identifier for the next month data.
    :return: DataFrame with 'bib_id' and churn labels.
    """
    return prepare_churn_label(*read_required_data(spark, config, current_month, next_month))