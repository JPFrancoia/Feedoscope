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


# PostgreSQL connection string used by every training and inference command.
DATABASE_URL = os.getenv("DATABASE_URL", "")
assert DATABASE_URL != "", "DATABASE_URL environment variable is not set"

# Logging config file path. Development commands typically override this with the
# colored console config, while production can point at the JSON logger config.
LOGGING_CONFIG = os.getenv("LOGGING_CONFIG", "logging.conf")

# Allow relevance or urgency training to fall back to CPU when CUDA is not
# available. Defaults to false because training is expected to run on a GPU.
ALLOW_TRAINING_WO_GPU = strtobool(os.getenv("ALLOW_TRAINING_WO_GPU", "False"))

# Allow inference commands to run on CPU when CUDA is not available. Defaults to
# false because production inference is expected to use a GPU.
ALLOW_INFERENCE_WO_GPU = strtobool(os.getenv("ALLOW_INFERENCE_WO_GPU", "False"))

# Extra sample weight applied to starred or upvoted articles during relevance
# training so explicitly preferred articles influence the classifier more.
EXCELLENT_WEIGHT = float(os.getenv("EXCELLENT_WEIGHT", "3.0"))

# Size of the held-out validation set used by training and eval commands.
# Production-style runs leave this at 0 to skip validation entirely.
VALIDATION_SIZE = int(os.getenv("VALIDATION_SIZE", "0"))

# Hugging Face model ID for the frozen relevance embedding encoder.
RELEVANCE_MODEL_NAME = os.getenv("RELEVANCE_MODEL_NAME", "google/embeddinggemma-300m")

# Maximum token budget used both when preparing relevance text and when encoding
# it with the frozen Gemma model.
RELEVANCE_MAX_LENGTH = int(os.getenv("RELEVANCE_MAX_LENGTH", "2048"))

# Strategy used to build the relevance text from article title and body before
# embedding. This changes the embedding output and is part of the cache key.
_relevance_text_prep_mode = os.getenv("RELEVANCE_TEXT_PREP_MODE", "title_head")
assert _relevance_text_prep_mode in (
    "single_blob",
    "title_head",
), "RELEVANCE_TEXT_PREP_MODE must be 'single_blob' or 'title_head'"
RELEVANCE_TEXT_PREP_MODE = cast(
    Literal["single_blob", "title_head"], _relevance_text_prep_mode
)

# Explicit cache-busting version for relevance text preparation. Bump this when
# changing text-cleaning or truncation logic so stale embeddings are recomputed.
RELEVANCE_PREP_VERSION = int(os.getenv("RELEVANCE_PREP_VERSION", "1"))

# Batch size for frozen relevance embedding generation. Higher values can speed
# up inference and training if enough GPU memory is available.
RELEVANCE_ENCODER_BATCH_SIZE = int(os.getenv("RELEVANCE_ENCODER_BATCH_SIZE", "4"))

# Inverse regularization strength for the logistic-regression relevance head.
# This affects only the classifier fit, not the embedding cache itself.
RELEVANCE_LINEAR_C = float(os.getenv("RELEVANCE_LINEAR_C", "5.0"))
