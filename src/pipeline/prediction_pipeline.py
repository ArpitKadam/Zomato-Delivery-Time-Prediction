from src.logger import logger
from src.exception import ZomatoDeliveryException
from src.configuration.configuration import *
from src.utils import load_object
import sys
import pandas as pd


class Prediction_Pipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            logger.info("Starting prediction process...")

            # Load preprocessor and model
            logger.info(f"Loading preprocessor from: {PROCESSOR_MODEL_PATH}")
            preprocessor = load_object(PROCESSOR_MODEL_PATH)

            logger.info(f"Loading model from: {MODEL_FILE_PATH}")
            model = load_object(MODEL_FILE_PATH)

            # Preprocess and predict
            logger.info("Transforming input features...")
            data_scaled = preprocessor.transform(features)

            logger.info("Performing prediction...")
            predictions = model.predict(data_scaled)

            logger.info("Prediction completed successfully.")
            return predictions

        except Exception as e:
            logger.exception("Exception occurred during prediction.")
            raise ZomatoDeliveryException(e, sys)


class CustomData:
    def __init__(
        self,
        Delivery_person_Age: int,
        Delivery_person_Ratings: float,
        Weather_conditions: str,
        Road_traffic_density: str,
        Vehicle_condition: int,
        multiple_deliveries: int,
        distance: float,
        Type_of_order: str,
        Type_of_vehicle: str,
        Festival: str,
        City: str,
        Delivery_person_City: str
    ):
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.multiple_deliveries = multiple_deliveries
        self.distance = distance
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.Festival = Festival
        self.City = City
        self.Delivery_person_City = Delivery_person_City

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            logger.info("Creating DataFrame from input values...")

            data = {
                "Delivery_person_Age": [self.Delivery_person_Age],
                "Delivery_person_Ratings": [self.Delivery_person_Ratings],
                "Weather_conditions": [self.Weather_conditions],
                "Road_traffic_density": [self.Road_traffic_density],
                "Vehicle_condition": [self.Vehicle_condition],
                "multiple_deliveries": [self.multiple_deliveries],
                "distance": [self.distance],
                "Type_of_order": [self.Type_of_order],
                "Type_of_vehicle": [self.Type_of_vehicle],
                "Festival": [self.Festival],
                "City": [self.City],
                "Delivery_person_City": [self.Delivery_person_City],
            }

            df = pd.DataFrame(data)
            logger.info(f"Input DataFrame created with shape: {df.shape}")
            return df

        except Exception as e:
            logger.exception("Failed to create DataFrame from input data.")
            raise ZomatoDeliveryException(e, sys)
