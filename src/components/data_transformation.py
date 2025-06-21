from src.logger import logger
from src.exception import ZomatoDeliveryException
from src.constants import *
from src.utils import save_obj
from src.configuration.configuration import *
import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

class Feature_Engineering(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def distance_numpy(self, df, lat1, lon1, lat2, lon2):
        try:
            logger.debug("Calculating distance using Haversine formula...")
            p = np.pi / 180
            lat1_rad = df[lat1] * p
            lon1_rad = df[lon1] * p
            lat2_rad = df[lat2] * p
            lon2_rad = df[lon2] * p

            delta_lon = lon2_rad - lon1_rad
            central_angle = np.arccos(
                np.clip(
                    np.sin(lat1_rad) * np.sin(lat2_rad) +
                    np.cos(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon),
                    -1, 1
                )
            )

            R = 6371.0
            df['distance'] = R * central_angle
            logger.info("Distance column successfully calculated and added.")

        except Exception as e:
            logger.exception("Exception occurred while calculating distance.")
            raise ZomatoDeliveryException(f"Failed to calculate distance: {e}")

    def extract_time(self, x):
        try:
            return x.split(":")[0] + ":" + x.split(":")[1][:2]
        except IndexError:
            return "00:00"

    def transform(self, df):
        try:
            logger.info("Initiating feature engineering on dataset...")

            logger.debug("Dropping ID column...")
            df.drop(['ID'], axis=1, inplace=True)

            logger.debug("Cleaning latitude and longitude values...")
            df['Restaurant_latitude'] = df['Restaurant_latitude'].abs()
            df['Restaurant_longitude'] = df['Restaurant_longitude'].abs()

            logger.debug("Dropping invalid latitude and longitude rows...")
            df = df.drop(df[(df['Restaurant_latitude'] < 8) & (df['Delivery_location_latitude'] < 8)].index)

            self.distance_numpy(df, 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude')

            logger.debug("Creating 'Delivery_person_City' from ID...")
            df['Delivery_person_City'] = df['Delivery_person_ID'].str.split("RES", expand=True)[0]

            logger.debug("Filling missing values in Delivery_person_Ratings...")
            df['Delivery_person_Ratings'] = df.groupby("Delivery_person_ID")['Delivery_person_Ratings'].transform(lambda x: x.fillna(x.mean()))

            logger.debug("Dropping unnecessary columns post transformation...")
            df.drop(columns=['Delivery_person_ID', 'Time_Orderd', 'Time_Order_picked', 'Restaurant_latitude','Restaurant_longitude',
                 'Delivery_location_latitude','Delivery_location_longitude', 'Order_Date'],
                axis=1, inplace=True)

            logger.info("Feature engineering completed. Returning transformed dataframe.")
            return df

        except Exception as e:
            logger.exception("Exception occurred in transform method.")
            raise ZomatoDeliveryException(f"Failed to transform data: {e}")

@dataclass
class DataTransformationConfig:
    feature_eng_obj_file_path: str = FEATURE_ENG_MODEL_PATH
    preprocessor_obj_file_path: str = PROCESSOR_MODEL_PATH
    transformed_train_file_path: str = TRANSFORMED_TRAIN_DATA_PATH
    transformed_test_file_path: str = TRANSFORMED_TEST_DATA_PATH

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logger.info("Creating data transformation pipeline...")

            ord_col = ['Weather_conditions', 'Road_traffic_density']
            num_col = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition', 'multiple_deliveries', 'distance']
            cat_col = ['Type_of_order', 'Type_of_vehicle', 'Festival', 'City', 'Delivery_person_City']

            Weather_conditions = ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Sandstorms', 'Stormy']
            Road_traffic_density = ['Low', 'Medium', 'High', 'Jam']

            logger.debug(f"Defined column groups. Ordinal: {ord_col}, Numerical: {num_col}, Categorical: {cat_col}")

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            ord_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(categories=[Weather_conditions, Road_traffic_density], handle_unknown='use_encoded_value', unknown_value=-1)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_col),
                ('cat_pipeline', cat_pipeline, cat_col),
                ('ord_pipeline', ord_pipeline, ord_col)
            ])

            logger.info("Data transformation pipelines created successfully.")
            return preprocessor

        except Exception as e:
            logger.exception("Exception occurred in get_data_transformation_object")
            raise ZomatoDeliveryException(f"Failed to get data transformation object: {e}")

    def get_feature_engineering_object(self):
        try:
            logger.info("Creating feature engineering pipeline...")
            feature_engineering = Pipeline(steps=[('fe', Feature_Engineering())])
            return feature_engineering
        except Exception as e:
            logger.exception("Exception occurred in get_feature_engineering_object")
            raise ZomatoDeliveryException(f"Failed to get feature engineering object: {e}")

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            logger.info("Starting full data transformation process...")
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logger.debug(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            fe_obj = self.get_feature_engineering_object()
            train_df = fe_obj.fit_transform(train_df)
            test_df = fe_obj.transform(test_df)

            preproc_obj = self.get_data_transformation_object()

            target_column = 'Time_taken (min)'
            x_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            x_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logger.debug("Fitting and transforming training data...")
            x_train = preproc_obj.fit_transform(x_train)
            logger.debug("Transforming testing data...")
            x_test = preproc_obj.transform(x_test)

            train_arr = np.c_[x_train, np.array(y_train)]
            test_arr = np.c_[x_test, np.array(y_test)]

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_file_path), exist_ok=True)

            pd.DataFrame(train_arr).to_csv(self.data_transformation_config.transformed_train_file_path, index=False, header=True)
            pd.DataFrame(test_arr).to_csv(self.data_transformation_config.transformed_test_file_path, index=False, header=True)

            save_obj(file_path=self.data_transformation_config.feature_eng_obj_file_path, obj=fe_obj)
            save_obj(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preproc_obj)

            logger.info("Data transformation pipeline completed and artifacts saved.")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logger.exception("Exception occurred during data transformation.")
            raise ZomatoDeliveryException(f"Failed to perform data transformation: {e}")