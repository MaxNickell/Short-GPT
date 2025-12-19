"""Simple JSON lines logger for training metrics."""

import json
import os


def log_metrics(path: str, **kwargs):
    """
    Append metrics as a JSON line to a log file.

    Args:
        path: Path to the log file (will be created if doesn't exist)
        **kwargs: Metric key-value pairs to log
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(kwargs) + "\n")
