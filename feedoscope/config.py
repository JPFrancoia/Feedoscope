import os


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


DATABASE_URL = os.getenv("DATABASE_URL", "")
assert DATABASE_URL != "", "DATABASE_URL environment variable is not set"

LOGGING_CONFIG = os.getenv("LOGGING_CONFIG", "logging.conf")

ALLOW_TRAINING_WO_GPU = strtobool(os.getenv("ALLOW_TRAINING_WO_GPU", "False"))

ALLOW_INFERENCE_WO_GPU = strtobool(os.getenv("ALLOW_INFERENCE_WO_GPU", "False"))

EXCELLENT_WEIGHT = float(os.getenv("EXCELLENT_WEIGHT", "3.0"))
