from src.logger import logger
from src.exception import ZomatoDeliveryException
from src.constants import *
from src.utils import evaluate_models, save_obj
from src.configuration.configuration import *
import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import numpy as np
import json

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logger.info("Starting model training process...")
            logger.info("Splitting the train and test arrays")
            X_train, y_train, X_test, y_test = (train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1])

            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(tree_method='hist', device='cuda')
            }

            param = {
                "LinearRegression": {},
                "Lasso": {
                    "alpha": [0.01, 0.1, 1.0, 10.0],
                    "max_iter": [1000, 2000, 3000]
                },
                "Ridge": {
                    "alpha": [0.001, 0.01, 0.1, 1.0, 2.0],
                    "solver": ["auto", "svd", "cholesky"]
                },
                "ElasticNet": {
                    "alpha": [0.01, 0.1, 1.0, 5.0],
                    "l1_ratio": [0.1, 0.5, 0.9],
                    "max_iter": [1000, 2000, 3000]
                },
                "KNeighborsRegressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                },
                "DecisionTreeRegressor": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5, 10, 20]
                },
                "RandomForestRegressor": {
                    "n_estimators": [50, 100, 150, 200],
                    "max_depth": [None, 10, 20],
                    "bootstrap": [True, False]
                },
                "GradientBoostingRegressor": {
                    "n_estimators": [50, 100, 150, 200],
                    "learning_rate": [0.05, 0.1, 1.0, 2.0],
                    "max_depth": [3, 5, 9]
                },
                "XGBRegressor": {
                    "n_estimators": [50, 100, 150, 200],
                    "learning_rate": [0.001, 0.1, 1.0, 2.0],
                    "max_depth": [3, 5, 7],
                    "reg_lambda": [1, 2, 3],
                    "reg_alpha": [0, 1]
                }
            }

            logger.info("Evaluating all models using cross-validation...")
            report = evaluate_models(X_train, y_train, X_test, y_test, models, param)

            logger.info("Finding the best performing model based on test R2 score...")
            best_model_score = max([metrics['test_r2_score'] for metrics in report.values()])
            best_model_name = [name for name, metrics in report.items() if metrics['test_r2_score'] == best_model_score][0]
            best_model = models[best_model_name]
            best_model_params = param[best_model_name]

            logger.info(f"Best model: {best_model_name} with test R2 score: {best_model_score:.4f}")
            logger.info(f"Hyperparameters used: {best_model_params}")

            logger.info("Saving the best model object...")
            save_obj(self.model_trainer_config.trained_model_file_path, best_model)
            logger.info(f"Model saved successfully at {self.model_trainer_config.trained_model_file_path}")

            report_data =  {
                "best_model_name": best_model_name,
                "best_model_score": best_model_score,
                "best_model_params": best_model_params,
                "metrics_report": report
            }
            logger.info(f"{report_data}")
            print(report_data)

            report_path = os.path.join(os.path.dirname(self.model_trainer_config.trained_model_file_path), "report.json")
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=4)
            
            logger.info(f"Report saved successfully at {report_path}")

        except Exception as e:
            logger.exception("Exception occurred during model training")
            raise ZomatoDeliveryException(f"Failed to train models: {e}")
