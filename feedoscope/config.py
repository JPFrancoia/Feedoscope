import os
from typing import Literal, cast


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

VALIDATION_SIZE = int(os.getenv("VALIDATION_SIZE", "0"))

RELEVANCE_MODEL_NAME = os.getenv("RELEVANCE_MODEL_NAME", "google/embeddinggemma-300m")

RELEVANCE_MAX_LENGTH = int(os.getenv("RELEVANCE_MAX_LENGTH", "2048"))

_relevance_text_prep_mode = os.getenv("RELEVANCE_TEXT_PREP_MODE", "title_head")
assert _relevance_text_prep_mode in (
    "single_blob",
    "title_head",
), "RELEVANCE_TEXT_PREP_MODE must be 'single_blob' or 'title_head'"
RELEVANCE_TEXT_PREP_MODE = cast(
    Literal["single_blob", "title_head"], _relevance_text_prep_mode
)

RELEVANCE_ENCODER_BATCH_SIZE = int(os.getenv("RELEVANCE_ENCODER_BATCH_SIZE", "4"))

RELEVANCE_LINEAR_C = float(os.getenv("RELEVANCE_LINEAR_C", "5.0"))
