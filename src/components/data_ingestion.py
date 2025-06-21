from src.logger import logger
from src.exception import ZomatoDeliveryException
from src.constants import *
from src.configuration.configuration import *
import os
from dataclasses import dataclass
from src.utils import download_data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

@dataclass
class DataIngestionConfig:
        raw_data_path = RAW_FILE_PATH
        train_data_path = TRAIN_FILE_PATH
        test_data_path = TEST_FILE_PATH
        data_link = DATA_LINK

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            logger.info("Data Ingestion started")
            # Create directories if they do not exist
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            logger.info(f"Raw data path: {self.data_ingestion_config.raw_data_path}")

            # Download the data
            download_data(self.data_ingestion_config.data_link, self.data_ingestion_config.raw_data_path)
            logger.info("Data downloaded successfully")

            # Read the raw data
            raw_data = pd.read_csv(self.data_ingestion_config.raw_data_path)
            logger.info(f"Raw data shape: {raw_data.shape}")
            
            # Split the data into train and test sets
            train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)
            logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
            
            # Create directories for train and test data if they do not exist
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok=True)
            logger.info(f"Train data path: {self.data_ingestion_config.train_data_path}")
            logger.info(f"Test data path: {self.data_ingestion_config.test_data_path}")

            # Save the train and test data
            train_data.to_csv(self.data_ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index=False)
            logger.info(f"Train data saved to {self.data_ingestion_config.train_data_path}")
            logger.info(f"Test data saved to {self.data_ingestion_config.test_data_path}")
            
            logger.info("Data Ingestion completed successfully")
            
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise ZomatoDeliveryException(f"Data ingestion failed: {e}") from e

if __name__ == "__main__":
     obj = DataIngestion()
     train_data, test_data = obj.initiate_data_ingestion()