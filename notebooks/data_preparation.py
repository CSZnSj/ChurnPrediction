from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, year, to_date, current_date, datediff, mean, median, count
from utils import  custom_read_parquet, custom_to_timestamp
from typing import Tuple


def read_required_data(
    spark: SparkSession, 
    config: dict, 
    month: str
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Reads the necessary Parquet files into DataFrames based on the given configuration and month.

    Args:
        spark (SparkSession): An active Spark session object.
        config (dict): Configuration dictionary.
        month (str): The month to be formatted into the path.

    Returns:
        tuple: A tuple containing DataFrames for assignment, recovery, package, recharge, and user data.

    Raises:
        ValueError: If there is an issue with reading any of the Parquet files.
    """
    
    assign_df = custom_read_parquet(spark, config, key="loan_assign", month=month)
    recovery_df = custom_read_parquet(spark, config, key="loan_recovery", month=month)
    package_df = custom_read_parquet(spark, config, key="package", month=month)
    recharge_df = custom_read_parquet(spark, config, key="recharge", month=month)
    user_df = custom_read_parquet(spark, config, key="user", month=month)
    
    return assign_df, recovery_df, package_df, recharge_df, user_df


def prepare_assign_df(assign_df: DataFrame) -> DataFrame:
    """
    Processes the assign DataFrame to compute the average and median loan amount for each 'bib_id'.
    
    Args:
    - assign_df: DataFrame containing assignment data with columns 'bib_id' and 'loan_amount'.

    Returns:
    - A DataFrame with the average and median loan amount for each 'bib_id'.
    """
    
    # Group by 'bib_id' and compute the average and median loan amount
    aggregated_assign_df = assign_df.groupBy("bib_id").agg(
        mean("loan_amount").alias("AverageAssignLoanAmount"),
        median("loan_amount").alias("MedianAssignLoanAmount"),
    )

    return aggregated_assign_df

def prepare_recovery_df(recovery_df: DataFrame) -> DataFrame:
    """
    Processes the recovery DataFrame to compute the average and median recovery loan amount for each 'bib_id'.
    
    Args:
    - recovery_df (DataFrame): The input DataFrame containing recovery data with columns 'bib_id' and 'loan_amount'.

    Returns:
    - DataFrame: A DataFrame with the average and median recovery loan amount for each 'bib_id'.
    """
    
    # Group by 'bib_id' and compute the average and median loan amount for recovery
    aggregated_recovery_df = recovery_df.groupBy("bib_id").agg(
        mean("loan_amount").alias("AverageRecoveryLoanAmount"),
        median("loan_amount").alias("MedianRecoveryLoanAmount"),
    )

    return aggregated_recovery_df

def prepare_package_df(package_df: DataFrame) -> DataFrame:
    """
    Processes the package DataFrame to compute average and median package amount and average and median duration for each 'bib_id'.
    Converts 'activation_date' and 'deactivation_date' columns to timestamp format, calculates duration,
    and computes the average and median values for each 'bib_id'.

    Args:
    - package_df: DataFrame containing package data with columns 'activation_date', 'deactivation_date', 'offer_amount', and 'bib_id'.

    Returns:
    - A DataFrame with the average and median package amount and average and median duration for each 'bib_id'.
    """

    # Convert 'activation_date' and 'deactivation_date' to timestamp format
    package_df = custom_to_timestamp(df=package_df, col_name="activation_date")
    package_df = custom_to_timestamp(df=package_df, col_name="deactivation_date")

    # Calculate the duration between 'deactivation_date' and 'activation_date'
    package_df = package_df.withColumn("duration", datediff(col("deactivation_date"), col("activation_date")))

    # Group by 'bib_id' and compute the average and median offer amount and duration
    aggregated_package_df = package_df.groupBy("bib_id").agg(
        mean("offer_amount").alias("AveragePackageAmount"),
        median("offer_amount").alias("MedianPackageAmount"),
        mean("duration").alias("AveragePackageDuration"),
        median("duration").alias("MedianPackageDuration"),
        count("*").alias("CountPackage")
    )

    return aggregated_package_df

def prepare_recharge_df(recharge_df: DataFrame) -> DataFrame:
    """
    Prepares the recharge DataFrame by aggregating recharge data for each 'bib_id'.
    It computes the average and median recharge amount and the number of recharges per 'bib_id'.

    Args:
    - recharge_df: DataFrame containing recharge data with columns 'bib_id' and 'recharge_value_amt'.

    Returns:
    - A DataFrame with aggregated recharge statistics: average and median recharge amount and count of recharges per 'bib_id'.
    """

    # Group by 'bib_id' and calculate the average, median recharge amount, and number of recharges
    aggregated_recharge_df = recharge_df.groupBy("bib_id").agg(
        mean("recharge_value_amt").alias("AverageRechargeAmount"),
        median("recharge_value_amt").alias("MedianRechargeAmount"),
        count("*").alias("CountRecharge"),
    )

    return aggregated_recharge_df

def prepare_user_df(user_df: DataFrame) -> DataFrame:
    """
    Prepares the user DataFrame by performing the following transformations:
    1. Converts 'registration_date_d' and 'date_of_birth_d' columns to timestamp format.
    2. Calculates the age of users based on 'date_of_birth_d'.
    3. Extracts the year from 'registration_date_d' for registration year.
    4. Selects relevant columns for analysis.

    Args:
    - user_df: DataFrame containing user data with columns 'registration_date_d', 'date_of_birth_d', etc.

    Returns:
    - A transformed DataFrame with selected and calculated columns.
    """

    # Convert registration and birth dates from custom string format to timestamp
    user_df = custom_to_timestamp(df=user_df, col_name="registration_date_d", format="yyyyMMdd")
    user_df = custom_to_timestamp(df=user_df, col_name="date_of_birth_d", format="yyyyMMdd")

    # Calculate age from 'date_of_birth_d' by computing the difference with the current date
    user_df = user_df.withColumn("age", datediff(current_date(), col("date_of_birth_d")) / 365.25)

    # Extract year from 'registration_date_d' for the registration year
    user_df = user_df.withColumn("registration_year", year(col("registration_date_d")))

    # Define the list of columns to be selected for further analysis
    selected_columns = ["bib_id", "age", "registration_year", "contract_type_v", "gender_v", "ability_status", "account_balance"]

    # Select only the necessary columns
    user_df = user_df.select(selected_columns)

    return user_df

from pyspark.sql import DataFrame

def prepare_data(
    assign_df: DataFrame,
    recovery_df: DataFrame,
    package_df: DataFrame,
    recharge_df: DataFrame,
    user_df: DataFrame
) -> DataFrame:
    """
    Processes and prepares DataFrames by applying respective transformation functions.

    Args:
    - assign_df: DataFrame containing assignment data.
    - recovery_df: DataFrame containing recovery data.
    - package_df: DataFrame containing package data.
    - recharge_df: DataFrame containing recharge data.
    - user_df: DataFrame containing user data.

    Returns:
    - A tuple of processed DataFrames: (prepared_assign_df, prepared_recovery_df, prepared_package_df, prepared_recharge_df, prepared_user_df).
    """
    prepared_assign_df = prepare_assign_df(assign_df)
    prepared_recovery_df = prepare_recovery_df(recovery_df)
    prepared_package_df = prepare_package_df(package_df)
    prepared_recharge_df = prepare_recharge_df(recharge_df)
    prepared_user_df = prepare_user_df(user_df)

    return (
        prepared_assign_df,
        prepared_recovery_df,
        prepared_package_df,
        prepared_recharge_df,
        prepared_user_df
    )

def merge_dataframes(
    assign_df: DataFrame,
    recovery_df: DataFrame,
    package_df: DataFrame,
    recharge_df: DataFrame,
    user_df: DataFrame,
    churn_label_df: DataFrame
) -> DataFrame:
    """
    Merges multiple DataFrames into a single DataFrame based on the 'bib_id' column.

    Args:
    - assign_df: DataFrame with assignment data.
    - recovery_df: DataFrame with recovery data.
    - package_df: DataFrame with package data.
    - recharge_df: DataFrame with recharge data.
    - user_df: DataFrame with user data.
    - churn_label_df: DataFrame containing churn labels.

    Returns:
    - A single DataFrame containing merged data from all input DataFrames.
    """
    merged_df = churn_label_df
    # merged_df = merged_df.join(assign_df, on="bib_id", how="left")
    # merged_df = merged_df.join(recovery_df, on="bib_id", how="left")
    merged_df = merged_df.join(package_df, on="bib_id", how="left")
    merged_df = merged_df.join(recharge_df, on="bib_id", how="left")
    merged_df = merged_df.join(user_df, on="bib_id", how="left")
    return merged_df

def generate_dataset(
    spark: SparkSession,
    config: dict,
    month: str,
    churn_label_df: DataFrame
) -> DataFrame:
    """
    Generates a dataset by reading required data, preparing it, and merging it with churn labels.

    Args:
        spark (SparkSession): An active Spark session object.
        config (dict): Configuration dictionary.
        month (str): The month to be used for data retrieval.
        churn_label_df (DataFrame): DataFrame containing churn labels.

    Returns:
        DataFrame: The final merged DataFrame containing all prepared and merged data and related labels
    """

    return merge_dataframes(*prepare_data(*read_required_data(spark, config, month)), churn_label_df)


