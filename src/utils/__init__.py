import os
import sys
import urllib.request as request
from src.logger import logger
from src.exception import ZomatoDeliveryException
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

def download_data(url, file_path):
    """Downloads the data from the given URL and saves it to the specified file path."""
    try:
        request.urlretrieve(url, file_path)
        logger.info(f"Data downloaded successfully from {url} and saved to {file_path}")
    except Exception as e:
        logger.error(f"Error downloading data from {url}: {e}")
        raise ZomatoDeliveryException(f"Failed to download data: {e}")
    
def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logger.error(f"Error saving object to {file_path}: {e}")
        raise ZomatoDeliveryException(f"Failed to save object: {e}")

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            logger.info(f"Starting GridSearchCV for: {model_name}")
            params = param.get(model_name, {})
            gs = GridSearchCV(model, params, cv=5, scoring='r2', verbose=2, n_jobs=-1)
            gs.fit(X_train, y_train)

            best_params = gs.best_params_
            logger.info(f"Best params for {model_name}: {best_params}")

            model.set_params(**best_params)
            model.fit(X_train, y_train)
            logger.info(f"Model {model_name} trained with best parameters.")

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_rmse = root_mean_squared_error(y_train, y_train_pred)
            test_rmse = root_mean_squared_error(y_test, y_test_pred)

            report[model_name] = {
                "train_r2_score": train_r2,
                "test_r2_score": test_r2,
                "train_mae_score": train_mae,
                "test_mae_score": test_mae,
                "train_mse_score": train_mse,
                "test_mse_score": test_mse,
                "train_rmse_score": train_rmse,
                "test_rmse_score": test_rmse,
                "best_params": best_params
            }

            logger.info(f"Evaluation completed for {model_name} with Test R2: {test_r2:.4f}")

        logger.info("All models evaluated successfully.")
        return report

    except Exception as e:
        logger.exception("Error occurred during model evaluation.")
        raise ZomatoDeliveryException(f"Failed to evaluate models: {e}")
