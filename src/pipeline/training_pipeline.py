from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logger
from src.exception import ZomatoDeliveryException
import sys

class Train:
    def __init__(self):
        self.c = 0
        print(f"============>  Initiated {self.c}  <============")

    def run_pipeline(self):
        try:
            logger.info("Starting the Pipeline...")

            logger.info("Starting data ingestion...")
            train_data_path, test_data_path = DataIngestion().run_data_ingestion_pipeline()
            logger.info("Data ingestion completed.")

            logger.info("Starting data transformation...")
            train_arr, test_arr, preprocessor_obj_file_path = DataTransformation().initiate_data_transformation(
                train_data_path, test_data_path
            )
            logger.info("Data transformation completed.")

            logger.info("Starting model training...")
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_training(train_arr, test_arr)
            logger.info("Model training completed.")

            logger.info("Pipeline execution completed successfully.")
        except Exception as e:
            logger.exception("An error occurred during pipeline execution.")
            logger.exception(e)
            raise ZomatoDeliveryException(e, sys)

