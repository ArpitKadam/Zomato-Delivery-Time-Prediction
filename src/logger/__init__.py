import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

class Logger:
    _logger_instance = None  # class-level variable to ensure one-time setup

    @classmethod
    def get_logger(cls, name: str = "AppLogger") -> logging.Logger:
        if cls._logger_instance is None:
            # Logs directory
            logs_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(logs_dir, exist_ok=True)

            # Log filename with timestamp
            log_filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
            log_file_path = os.path.join(logs_dir, log_filename)

            # Create logger
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)

            # File handler
            file_handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=5)
            file_formatter = logging.Formatter(
                "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter("%(lineno)d - %(levelname)s - %(message)s")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            cls._logger_instance = logger

        return cls._logger_instance


logger = Logger.get_logger("ZomatoDeliveryLogger")
