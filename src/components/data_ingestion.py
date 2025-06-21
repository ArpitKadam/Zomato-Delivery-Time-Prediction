from src.logger import logger
from src.exception import ZomatoDeliveryException
from src.constants import *
from src.configuration.configuration import RAW_FILE_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH
from src.utils import download_data
import os
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path: str = RAW_FILE_PATH
    train_data_path: str = TRAIN_FILE_PATH
    test_data_path: str = TEST_FILE_PATH
    data_link: str = DATA_LINK

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logger.info("Data Ingestion process started.")
            logger.debug(f"Configuration used: {self.data_ingestion_config}")

            # Create raw data directory
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            logger.debug(f"Ensured directory exists for raw data path: {self.data_ingestion_config.raw_data_path}")

            # Download the data
            logger.info(f"Downloading data from {self.data_ingestion_config.data_link}")
            download_data(self.data_ingestion_config.data_link, self.data_ingestion_config.raw_data_path)
            logger.info("Data downloaded successfully.")

            # Read the data
            logger.debug("Reading the raw CSV data...")
            raw_data = pd.read_csv(self.data_ingestion_config.raw_data_path)
            logger.info(f"Raw data read successfully with shape: {raw_data.shape}")

            # Split into train and test sets
            logger.info("Splitting raw data into training and test sets...")
            train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)
            logger.info(f"Data split: Train Shape = {train_data.shape}, Test Shape = {test_data.shape}")

            # Save train/test data
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok=True)
            logger.debug("Output directories verified.")

            train_data.to_csv(self.data_ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index=False)
            logger.info(f"Train data saved at: {self.data_ingestion_config.train_data_path}")
            logger.info(f"Test data saved at: {self.data_ingestion_config.test_data_path}")

            logger.info("Data Ingestion completed successfully.")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logger.exception("Exception occurred during data ingestion.")
            raise ZomatoDeliveryException(f"Data ingestion failed: {e}") from e

    def run_data_ingestion_pipeline(self):
        try:
            logger.info("Running the Data Ingestion Pipeline...")
            train_data_path, test_data_path = self.initiate_data_ingestion()
            logger.info("Data Ingestion Pipeline completed.")
            return train_data_path, test_data_path

        except Exception as e:
            logger.exception("Exception occurred in Data Ingestion Pipeline.")
            raise ZomatoDeliveryException(f"Failed to run Data Ingestion Pipeline: {e}") from e
