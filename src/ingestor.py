from logger import CustomLogger
from pyspark.sql.utils import AnalysisException
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from utils import create_spark_session, load_config

class CustomSchema:
    """
    Define schemas for CSV files.
    """
    loan_assign_schema = StructType([
        StructField("bib_id", StringType(), True),
        StructField("date_key", StringType(), True), 
        StructField("fake_id", StringType(), True),
        StructField("nid_hash", StringType(), True),
        StructField("loan_id", StringType(), True),
        StructField("loan_amount", DoubleType(), True),
        StructField("date_timestamp", StringType(), True) # date_timestamp
    ])

    loan_fee_schema = StructType([
        StructField("bib_id", StringType(), True),
        StructField("date_key", StringType(), True),
        StructField("fake_id", StringType(), True),
        StructField("nid_hash", StringType(), True),
        StructField("loan_id", StringType(), True),
        StructField("loan_amount", DoubleType(), True),
        StructField("hsdp_fee", DoubleType(), True),
        StructField("date_timestamp", StringType(), True) # date_timestamp
    ])

    loan_recovery_schema = StructType([
        StructField("bib_id", StringType(), True),
        StructField("date_key", StringType(), True),
        StructField("fake_id", StringType(), True),
        StructField("nid_hash", StringType(), True),
        StructField("loan_id", StringType(), True),
        StructField("loan_amount", DoubleType(), True),
        StructField("hsdp_recovery", DoubleType(), True),
        StructField("date_timestamp", StringType(), True) # date_timestamp
    ])

    package_schema = StructType([
        StructField("bib_id", StringType(), True),
        StructField("date_key", StringType(), True),
        StructField("fake_id", StringType(), True),
        StructField("nid_hash", StringType(), True),
        StructField("offering_code", StringType(), True),
        StructField("offer_amount", DoubleType(), True),
        StructField("offering_name", StringType(), True),
        StructField("activation_date", StringType(), True),
        StructField("deactivation_date", StringType(), True) # deactivation_date
    ])

    recharge_schema = StructType([
        StructField("bib_id", StringType(), True),
        StructField("date_key", StringType(), True),
        StructField("fake_id", StringType(), True),
        StructField("nid_hash", StringType(), True),
        StructField("recharge_value_amt", DoubleType(), True),
        StructField("recharge_dt", StringType(), True),
        StructField("origin_host_nm", StringType(), True),
        StructField("account_balance_before_amt", DoubleType(), True),
        StructField("account_balance_after_amt", DoubleType(), True) # account_balance_after_am
    ])

    user_schema = StructType([
        StructField("bib_id", StringType(), True),
        StructField("fake_id", StringType(), True),
        StructField("nid_hash", StringType(), True),
        StructField("contract_type_v", StringType(), True),
        StructField("gender_v", StringType(), True),
        StructField("registration_date_d", StringType(), True),
        StructField("date_of_birth_d", StringType(), True),
        StructField("ability_status", StringType(), True),
        StructField("account_balance", DoubleType(), True),
        StructField("base_station_cd", StringType(), True),
        StructField("sitei", StringType(), True)
    ])

def validate_schema(df, expected_schema):
    """
    Validate that the DataFrame conforms to the expected schema.
    
    :param df: The DataFrame to validate.
    :param expected_schema: The expected schema as a StructType object.
    :return: None. Raises ValueError if schema does not match.
    """
    expected_fields = {field.name: field.dataType for field in expected_schema.fields}
    actual_fields = {field.name: field.dataType for field in df.schema.fields}
    
    if expected_fields != actual_fields:
        mismatch_details = {
            "expected": expected_fields,
            "actual": actual_fields
        }
        raise ValueError(f"Schema mismatch found: {mismatch_details}")

def load_and_process_data(logger, config, spark):
    """
    Load raw CSV files, validate schemas, and save as Parquet files.
    
    :param logger: Logger object.
    :param config: Configuration dictionary.
    :param spark: Spark session.
    :param stop_the_program_if_mismatch: Flag to stop the program if schema mismatch is found.
    """
    logger.info("Starting to load and process CSV files ...")

    keys = config["keys"]
    months = config["months"]
    logger.info(f'keys: {keys}')
    logger.info(f"Months: {months}")

    for key in keys:
        for month in months:
            try:
                # Determine the schema and path
                schema = getattr(CustomSchema, f"{key}_schema")
                csv_path = config["ingestor.py"]["paths"]["csv"][key].format(month=month)
                
                # Read CSV with schema enforcement
                df = spark.read.csv(csv_path, header=True, schema=schema)
                logger.info(f"Successfully read CSV file: {csv_path}")

                # Validate schema
                validate_schema(df, schema)

                # Process the DataFrame (Save as Parquet)
                parquet_path = config["ingestor.py"]["paths"]["parquet"][key].format(month=month)
                df.write.parquet(parquet_path, mode="overwrite")
                logger.info(f"Successfully wrote Parquet file: {parquet_path}")

            except ValueError as ve:
                logger.error(f"Schema validation error: {ve}")
                raise
            except AnalysisException as ae:
                logger.error(f"Error processing CSV file: {csv_path}. Details: {ae}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

if __name__ == "__main__":
    logger = CustomLogger(name='IngestorLogger').get_logger()
    spark_name = "IngestorSpark"
    
    try:
        # Create Spark session
        spark = create_spark_session(logger, spark_name)
        logger.info(f"Spark session '{spark_name}' created successfully.")
        
        # Load configuration
        config = load_config(logger)
        logger.info("Configuration loaded successfully.")
        
        # Load and process data
        load_and_process_data(logger, config, spark)
    
    finally:
        spark.stop()
        logger.info(f"Spark session '{spark_name}' has been stopped.")
