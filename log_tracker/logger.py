import os
import logging
import colorlog
from utils import LoadConfig


class Logger:
    @staticmethod
    def get_logger():
        # Load config
        config = LoadConfig.load_config("configs/log_config.yaml")

        # Create logger
        logger = logging.getLogger("global_logger")
        logger.propagate = False  # Avoid duplicate logs if root logger is configured

        if not logger.handlers:  # Prevent re-adding handlers if already configured
            # Set log level
            verbose = config.get('verbose', False)
            log_level = logging.INFO if verbose else logging.WARNING
            logger.setLevel(log_level)

            # Console handler with color
            stream_handler = logging.StreamHandler()
            console_formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red'
                }
            )
            stream_handler.setFormatter(console_formatter)
            logger.addHandler(stream_handler)

            # Optional file logging
            if config.get('log_to_file', False):
                log_file_path = config.get('log_file_path', 'logs/app.log')
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                try:
                    file_handler = logging.FileHandler(log_file_path)
                    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    file_handler.setFormatter(file_formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    logger.error(f"Failed to set up file logging: {e}")

        return logger
