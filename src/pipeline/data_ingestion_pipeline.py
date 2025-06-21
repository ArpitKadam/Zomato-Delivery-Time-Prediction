from src.components.data_ingestion import DataIngestion
from src.logger import logger

def run_data_ingestion_pipeline():
    try:
        logger.info("Starting Data Ingestion Pipeline")
        
        # Create an instance of DataIngestion
        data_ingestion = DataIngestion()
        
        # Initiate data ingestion
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        logger.info(f"Data Ingestion completed successfully. ")
        
        return train_data_path, test_data_path
    
    except Exception as e:
        logger.error(f"Error in Data Ingestion Pipeline: {e}")
        raise e