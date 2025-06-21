import os
import sys
import urllib.request as request
from src.logger import logger
from src.exception import ZomatoDeliveryException

def download_data(url, file_path):
    """Downloads the data from the given URL and saves it to the specified file path."""
    try:
        request.urlretrieve(url, file_path)
        logger.info(f"Data downloaded successfully from {url} and saved to {file_path}")
    except Exception as e:
        logger.error(f"Error downloading data from {url}: {e}")
        raise ZomatoDeliveryException(f"Failed to download data: {e}")
