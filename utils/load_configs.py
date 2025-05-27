import os
import yaml


class LoadConfig:
    @staticmethod
    def load_config(config_path):
        try:
            # Load configuration from YAML file
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            raise
