import yaml
import os
from dotenv import load_dotenv


def load_config(path="config/default.yaml"):
    """Load YAML config and merge with environment variables."""
    load_dotenv()
    with open(path) as f:
        config = yaml.safe_load(f)
    return config
