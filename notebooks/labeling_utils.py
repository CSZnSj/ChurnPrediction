from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, max as spark_max, min as spark_min, to_timestamp, datediff, when, greatest, least
from pyspark.sql.types import TimestampType

def ensure_timestamp(
        df: DataFrame, 
        column_name: str,
        format: str = "yyyyMMdd HH:mm:ss") -> DataFrame:
    """
    Ensures that the specified column in the DataFrame is of TimestampType.
    If the column is not already a TimestampType, it attempts to convert it 
    using the format 'yyyyMMdd HH:mm:ss'. Raises an error if conversion fails.

    :param df: DataFrame containing the column to be converted.
    :param column_name: The name of the column to convert to TimestampType.
    :return: DataFrame with the specified column converted to TimestampType.
    """
    # Check if the column is already of TimestampType
    if isinstance(df.schema[column_name].dataType, TimestampType):
        return df

    # Attempt to convert the column to TimestampType using the specified format
    try:
        df = df.withColumn(column_name, to_timestamp(col(column_name), format))
    except Exception as e:
        raise ValueError(f"Error converting column '{column_name}' to TimestampType: {e}")
    
    return df


def find_time_activity_of_user(
        package_df: DataFrame, 
        recharge_df: DataFrame, 
        aggregation_func: str) -> DataFrame:
    """
    Computes aggregated dates (maximum or minimum) for the package and recharge DataFrames.

    :param package_df: DataFrame containing package data with 'bib_id' and 'activation_date' columns.
    :param recharge_df: DataFrame containing recharge data with 'bib_id' and 'recharge_dt' columns.
    :param aggregation_func: Aggregation function to apply; should be either 'max' or 'min'.
    :return: DataFrame with 'bib_id' and the aggregated dates as either 'max_date' or 'min_date'.
    """
    # Convert date columns to TimestampType
    package_df = ensure_timestamp(package_df, "activation_date")
    recharge_df = ensure_timestamp(recharge_df, "recharge_dt")

    # Define a mapping between function name and the actual PySpark function
    func_mapping = {"max": spark_max, "min": spark_min}
    
    # Validate the provided aggregation function
    if aggregation_func not in func_mapping:
        raise ValueError("Invalid aggregation function. Expected 'max' or 'min'.")
    
    aggregate_func = func_mapping[aggregation_func]
    combine_func = greatest if aggregation_func == "max" else least

    # Perform aggregation on package DataFrame
    package_agg = package_df.groupBy("bib_id").agg(
        aggregate_func(col("activation_date")).alias("package_date")
    )

    # Perform aggregation on recharge DataFrame
    recharge_agg = recharge_df.groupBy("bib_id").agg(
        aggregate_func(col("recharge_dt")).alias("recharge_date")
    )

    # Combine the aggregated results using an outer join
    combined_df = package_agg.join(recharge_agg, on="bib_id", how="outer")

    # Calculate the combined aggregated date using the specified function (max or min)
    combined_df = combined_df.withColumn(
        f"{aggregation_func}_date",
        combine_func(col("package_date"), col("recharge_date"))
    )
    
    return combined_df.select("bib_id", f"{aggregation_func}_date")

def determine_churn_label(
        package_df: DataFrame, 
        recharge_df: DataFrame, 
        next_package_df: DataFrame, 
        next_recharge_df: DataFrame) -> DataFrame:
    """
    Determines churn labels by comparing maximum and minimum dates from package and recharge DataFrames.

    :param package_df: DataFrame containing package data for the current period.
    :param recharge_df: DataFrame containing recharge data for the current period.
    :param next_package_df: DataFrame containing package data for the next period.
    :param next_recharge_df: DataFrame containing recharge data for the next period.
    :return: DataFrame with 'bib_id' and a churn label (1 for churned, 0 for not churned).
    """
    # Compute maximum dates for the current period
    latest_date_of_activity_of_each_user_this_period = find_time_activity_of_user(package_df, recharge_df, aggregation_func="max")
    
    # Compute minimum dates for the next period
    earliest_date_of_activity_of_each_user_next_period = find_time_activity_of_user(next_package_df, next_recharge_df, aggregation_func="min")

    # Join the DataFrames on 'bib_id'
    df = latest_date_of_activity_of_each_user_this_period.join(earliest_date_of_activity_of_each_user_next_period, on="bib_id", how="left")

    # Calculate the date difference between min_date and max_date
    df = df.withColumn(
        "date_diff",
        datediff(col("min_date"), col("max_date"))
    )

    # Determine churn label based on conditions
    df = df.withColumn(
        "label",
        when(
            (col("min_date").isNull() & col("max_date").isNotNull()) | (col("date_diff") > 30),
            1
        ).otherwise(0)
    )

    return df.select("bib_id", "label")

if __name__ == "__main__":

    needed_for_labeling = {
        "package": "activation_date",
        "recharge": "recharge_dt"
    }

    
