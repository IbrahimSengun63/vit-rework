import os
import logging
import colorlog
from utils import LoadConfig


class Logger:
    _logger = None  # Singleton-style logger

    @staticmethod
    def get_logger():
        if Logger._logger:
            return Logger._logger

        # Load config
        config = LoadConfig.load_config("configs/log_config.yaml")

        console_verbose = config.get('console_verbose', True)
        console_level = config.get('console_level', 'INFO').upper()

        write_file = config.get('write_file', False)
        file_level = config.get('file_level', 'WARN').upper()

        log_file_path = config.get('log_file_path', 'logs/app.log')

        # Create and configure logger
        logger = logging.getLogger("global_logger")
        logger.setLevel(logging.DEBUG)  # Let handlers control filtering

        if not logger.handlers:
            if console_verbose:
                stream_handler = logging.StreamHandler()
                stream_handler.setLevel(getattr(logging, console_level, logging.INFO))
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

            if write_file:
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                try:
                    file_handler = logging.FileHandler(log_file_path)
                    file_handler.setLevel(getattr(logging, file_level, logging.WARNING))
                    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    file_handler.setFormatter(file_formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    logger.error(f"Failed to set up file logging: {e}")

        Logger._logger = logger
        return logger
