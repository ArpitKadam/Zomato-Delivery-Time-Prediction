from src.constants import (ROOT_DIR, ARTIFACT_DIR_NAME, DATA_INGESTION_DIR_NAME, CURRENT_TIME_STAME, 
                           DATA_INGESTION_RAW_DATA, TRAIN_FILE_NAME, TEST_FILE_NAME, DATA_TRANSFORMATION_DIR_NAME,
                           DATA_TRANSFORMATION_OBJ_DIR, PREPROCESSOR_OBJ_FILE_NAME, FEATURE_ENG_OBJ_FILE_NAME,
                           DATA_TRANSFORMED_DIR, TRAIN_TRANSFORMED_FILE_NAME, TEST_TRANSFORMED_FILE_NAME,
                           MODEL_TRAINING_DIR_NAME, MODEL_OBJECT_NAME)
import os
import sys

RAW_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_NAME, DATA_INGESTION_DIR_NAME, CURRENT_TIME_STAME, DATA_INGESTION_RAW_DATA)
TRAIN_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_NAME, DATA_INGESTION_DIR_NAME, CURRENT_TIME_STAME, TRAIN_FILE_NAME)
TEST_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_NAME, DATA_INGESTION_DIR_NAME, CURRENT_TIME_STAME, TEST_FILE_NAME)

PROCESSOR_MODEL_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_NAME, DATA_TRANSFORMATION_DIR_NAME, CURRENT_TIME_STAME,DATA_TRANSFORMATION_OBJ_DIR , PREPROCESSOR_OBJ_FILE_NAME)
FEATURE_ENG_MODEL_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_NAME, DATA_TRANSFORMATION_DIR_NAME, CURRENT_TIME_STAME, DATA_TRANSFORMATION_OBJ_DIR, FEATURE_ENG_OBJ_FILE_NAME)
TRANSFORMED_DATA_DIR = os.path.join(ROOT_DIR, ARTIFACT_DIR_NAME, DATA_TRANSFORMATION_DIR_NAME, CURRENT_TIME_STAME, DATA_TRANSFORMED_DIR)
TRANSFORMED_TRAIN_DATA_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_NAME, DATA_TRANSFORMATION_DIR_NAME, CURRENT_TIME_STAME, DATA_TRANSFORMED_DIR, TRAIN_TRANSFORMED_FILE_NAME)
TRANSFORMED_TEST_DATA_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_NAME, DATA_TRANSFORMATION_DIR_NAME, CURRENT_TIME_STAME, DATA_TRANSFORMED_DIR, TEST_TRANSFORMED_FILE_NAME)

MODEL_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_NAME, MODEL_TRAINING_DIR_NAME, CURRENT_TIME_STAME, MODEL_OBJECT_NAME)