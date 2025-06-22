from src.constants import *
from src.configuration.configuration import *
import os, sys
import pandas as pd
import numpy as np
from src.logger import logger
from src.exception import ZomatoDeliveryException
import pickle
from src.utils import load_object
from sklearn.pipeline import Pipeline

# Folder constants
PREDICTION_FOLDER = 'batch_prediction'
FEATURE_ENG_FOLDER = 'feature_eng'
RAW_FOLDER = 'raw_input'
TRANSFORMED_FOLDER = 'transformed'
PREDICTION_CSV_FOLDER = 'prediction_csv'
PREDICTION_FILE = 'output.csv'

# Path definitions
ROOT_DIR = os.getcwd()
RAW_DIR = os.path.join(ROOT_DIR, PREDICTION_FOLDER, RAW_FOLDER)
FEATURE_ENG_DIR = os.path.join(ROOT_DIR, PREDICTION_FOLDER, FEATURE_ENG_FOLDER)
TRANSFORMED_DIR = os.path.join(ROOT_DIR, PREDICTION_FOLDER, TRANSFORMED_FOLDER)
BATCH_PREDICTION_DIR = os.path.join(ROOT_DIR, PREDICTION_FOLDER, PREDICTION_CSV_FOLDER)

class Batch_prediction:
    def __init__(self, input_file_path, model_file_path, transformer_file_path, feature_engineering_file_path):
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path

    def start_batch_prediction(self):
        try:
            logger.info("Batch prediction pipeline started.")

            # Load feature engineering object
            logger.info("Loading feature engineering pipeline...")
            with open(self.feature_engineering_file_path, 'rb') as f:
                feature_pipeline = pickle.load(f)
            logger.info(f"Loaded feature engineering object from: {self.feature_engineering_file_path}")

            # Load preprocessor
            logger.info("Loading preprocessor pipeline...")
            with open(self.transformer_file_path, 'rb') as f:
                preprocessor = pickle.load(f)
            logger.info(f"Loaded preprocessor object from: {self.transformer_file_path}")

            # Load model
            logger.info("Loading model...")
            model = load_object(file_path=self.model_file_path)
            logger.info(f"Model loaded from: {self.model_file_path}")

            # Read raw input
            df = pd.read_csv(self.input_file_path)
            os.makedirs(RAW_DIR, exist_ok=True)
            raw_path = os.path.join(RAW_DIR, 'input_data.csv')
            df.to_csv(raw_path, index=False)
            logger.info(f"Raw input saved at: {raw_path}")

            # Apply feature engineering
            logger.info("Applying feature engineering...")
            feature_engineering_pipeline = Pipeline([('feature_engineering', feature_pipeline)])
            df = feature_engineering_pipeline.transform(df)
            os.makedirs(FEATURE_ENG_DIR, exist_ok=True)
            feature_path = os.path.join(FEATURE_ENG_DIR, 'feature_engineered.csv')
            df.to_csv(feature_path, index=False)
            logger.info(f"Feature engineered data saved at: {feature_path}")

            # Drop target column
            if 'Time_taken (min)' in df.columns:
                df = df.drop('Time_taken (min)', axis=1)
                logger.info("Dropped target column 'Time_taken (min)' from features.")

            os.makedirs(TRANSFORMED_DIR, exist_ok=True)
            dropped_path = os.path.join(TRANSFORMED_DIR, 'features_dropped_target.csv')
            df.to_csv(dropped_path, index=False)
            logger.info(f"Data after dropping target saved at: {dropped_path}")

            # Apply preprocessing
            logger.info("Applying preprocessing transformation...")
            transformed_data = preprocessor.transform(df)
            logger.info(f"Preprocessing completed. Transformed shape: {transformed_data.shape}")

            # Generate predictions
            logger.info("Generating predictions...")
            predictions = model.predict(transformed_data)
            df_predictions = pd.DataFrame(predictions, columns=['prediction'])

            # Save predictions
            os.makedirs(BATCH_PREDICTION_DIR, exist_ok=True)
            output_path = os.path.join(BATCH_PREDICTION_DIR, PREDICTION_FILE)
            df_predictions.to_csv(output_path, index=False)
            logger.info(f"Batch predictions saved to: {output_path}")

            logger.info("Batch prediction pipeline completed successfully.")

        except Exception as e:
            logger.exception("An error occurred during batch prediction.")
            raise ZomatoDeliveryException(e, sys)
