from src.pipeline.training_pipeline import Train
from src.exception import ZomatoDeliveryException
import sys

if __name__ == "__main__":
    try:
        train_pipeline = Train()
        train_pipeline.run_pipeline()
    except Exception as e:
        raise ZomatoDeliveryException(e, sys)
