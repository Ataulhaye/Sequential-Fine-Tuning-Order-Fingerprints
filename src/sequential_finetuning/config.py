from pathlib import Path

import yaml


def load_config(path):
    """
    Load a YAML experiment configuration.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    return config
