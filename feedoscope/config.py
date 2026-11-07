import math
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


# PostgreSQL connection string used by every training and inference command.
DATABASE_URL = os.getenv("DATABASE_URL", "")
assert DATABASE_URL != "", "DATABASE_URL environment variable is not set"

# Logging config file path. Development commands typically override this with the
# colored console config, while production can point at the JSON logger config.
LOGGING_CONFIG = os.getenv("LOGGING_CONFIG", "logging.conf")

# Allow training to fall back to CPU when CUDA is not available. Defaults to
# false because training is expected to run on a GPU.
ALLOW_TRAINING_WO_GPU = strtobool(os.getenv("ALLOW_TRAINING_WO_GPU", "False"))

# Allow inference commands to run on CPU when CUDA is not available. Defaults to
# false because production inference is expected to use a GPU.
ALLOW_INFERENCE_WO_GPU = strtobool(os.getenv("ALLOW_INFERENCE_WO_GPU", "False"))

# Fixed half-life (in days) for the age-decay backend.
AGE_DECAY_HALF_LIFE_DAYS = float(os.getenv("AGE_DECAY_HALF_LIFE_DAYS", "7"))
assert (
    math.isfinite(AGE_DECAY_HALF_LIFE_DAYS) and AGE_DECAY_HALF_LIFE_DAYS > 0
), "AGE_DECAY_HALF_LIFE_DAYS must be finite and positive"

# Size of the held-out validation set used by training and eval commands.
# Production-style runs leave this at 0 to skip validation entirely.
VALIDATION_SIZE = int(os.getenv("VALIDATION_SIZE", "0"))

# Maximum age of relevance training articles, in whole days.
TRAINING_HISTORY_DAYS = int(os.getenv("TRAINING_HISTORY_DAYS", "1095"))
assert TRAINING_HISTORY_DAYS > 0, "TRAINING_HISTORY_DAYS must be positive"

# Hugging Face model ID for the frozen relevance embedding encoder.
RELEVANCE_MODEL_NAME = os.getenv("RELEVANCE_MODEL_NAME", "google/embeddinggemma-300m")

# Stable cache key for prompted shared embeddings. This is separate from the
# Hugging Face source ID because the prompt changes vector values.
RELEVANCE_EMBEDDING_KEY = os.getenv(
    "RELEVANCE_EMBEDDING_KEY",
    "google/embeddinggemma-300m-classification-v1",
)
RELEVANCE_EMBEDDING_PROMPT = os.getenv(
    "RELEVANCE_EMBEDDING_PROMPT", "task: classification | query: "
)

# Maximum token budget for the independent relevance body embedding.
RELEVANCE_MAX_LENGTH = int(os.getenv("RELEVANCE_MAX_LENGTH", "2048"))

# Explicit cache-busting version for relevance text preparation. Bump this when
# changing text-cleaning or truncation logic so stale embeddings are recomputed.
RELEVANCE_PREP_VERSION = int(os.getenv("RELEVANCE_PREP_VERSION", "2"))

# Batch size for frozen relevance embedding generation. Higher values can speed
# up inference and training if enough GPU memory is available.
RELEVANCE_ENCODER_BATCH_SIZE = int(os.getenv("RELEVANCE_ENCODER_BATCH_SIZE", "4"))

# Prompted relevance uses deterministic weighted logistic regression. These
# values affect only the relevance classifier artifact, not the embedding cache.
RELEVANCE_LINEAR_C = float(os.getenv("RELEVANCE_LINEAR_C", "5.0"))
assert (
    math.isfinite(RELEVANCE_LINEAR_C) and RELEVANCE_LINEAR_C > 0
), "RELEVANCE_LINEAR_C must be finite and positive"
IMPORTANT_ARTICLE_WEIGHT = float(os.getenv("IMPORTANT_ARTICLE_WEIGHT", "20"))
assert (
    math.isfinite(IMPORTANT_ARTICLE_WEIGHT) and IMPORTANT_ARTICLE_WEIGHT >= 1
), "IMPORTANT_ARTICLE_WEIGHT must be finite and at least 1"
