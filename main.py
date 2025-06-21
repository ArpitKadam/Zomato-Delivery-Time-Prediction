from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.logger import logger  # Import your custom logger

if __name__ == "__main__":
    try:
        logger.info("Starting the Pipeline...")

        # Run the data ingestion pipeline
        logger.info("Starting data ingestion...")
        train_data_path, test_data_path = DataIngestion().run_data_ingestion_pipeline()
        logger.info("Data ingestion completed.")

        # Run the data transformation pipeline
        logger.info("Starting data transformation...")
        train_arr, test_arr, preprocessor_obj_file_path = DataTransformation().initiate_data_transformation(
            train_data_path, test_data_path
        )
        logger.info("Data transformation completed.")
        logger.info(f"Preprocessor object saved at: {preprocessor_obj_file_path}")

        logger.info("Pipeline execution completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during pipeline execution.")
