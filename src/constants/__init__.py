import os
import sys
from datetime import datetime

def get_current_time():
    """Returns the current time in the format YYYY-MM-DD HH-MM-SS"""
    return str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

## DATA INGESTION CONSTANTS
CURRENT_TIME_STAME = get_current_time()
ROOT_DIR = os.getcwd()
DATA_LINK = "https://github.com/ArpitKadam/Zomato-Delivery-Time-Prediction/raw/refs/heads/main/Research/data.csv"
ARTIFACT_DIR_NAME = "Artifacts"
DATA_INGESTION_DIR_NAME = "Data_Ingestion"
DATA_INGESTION_RAW_DATA = "raw_data.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

## DATA TRANSFORMATION CONSTANTS
DATA_TRANSFORMATION_DIR_NAME = "Data_Transformation"
DATA_TRANSFORMATION_OBJ_DIR = "data tranformation models"
PREPROCESSOR_OBJ_FILE_NAME = "preprocessor.pkl"
FEATURE_ENG_OBJ_FILE_NAME = "feature_eng.pkl"
DATA_TRANSFORMED_DIR = "transformed_data"
TRAIN_TRANSFORMED_FILE_NAME = "train_transformed.csv"
TEST_TRANSFORMED_FILE_NAME = "test_transformed.csv"