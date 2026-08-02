import math
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

# Select the input used for final relevance-score decay. Semantic freshness is
# the default; urgency remains available for an immediate configuration rollback.
_relevance_decay_backend = os.getenv("RELEVANCE_DECAY_BACKEND", "semantic_freshness")
assert _relevance_decay_backend in (
    "semantic_freshness",
    "urgency",
), "RELEVANCE_DECAY_BACKEND must be 'semantic_freshness' or 'urgency'"
RELEVANCE_DECAY_BACKEND = cast(
    Literal["semantic_freshness", "urgency"], _relevance_decay_backend
)

# Half-life boundaries (in days) for the legacy urgency-based relevance decay.
HALF_LIFE_EVERGREEN = float(os.getenv("HALF_LIFE_EVERGREEN", "120"))
HALF_LIFE_URGENT = float(os.getenv("HALF_LIFE_URGENT", "10"))
assert (
    0 < HALF_LIFE_URGENT <= HALF_LIFE_EVERGREEN
), "HALF_LIFE_URGENT must be positive and no greater than HALF_LIFE_EVERGREEN"

# Size of the held-out validation set used by training and eval commands.
# Production-style runs leave this at 0 to skip validation entirely.
VALIDATION_SIZE = int(os.getenv("VALIDATION_SIZE", "0"))

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
RELEVANCE_PREP_VERSION = int(os.getenv("RELEVANCE_PREP_VERSION", "2"))

# Batch size for frozen relevance embedding generation. Higher values can speed
# up inference and training if enough GPU memory is available.
RELEVANCE_ENCODER_BATCH_SIZE = int(os.getenv("RELEVANCE_ENCODER_BATCH_SIZE", "4"))

# Prompted relevance uses a deterministic small MLP. These values affect only
# the relevance classifier artifact, not the shared embedding cache.
RELEVANCE_MLP_HIDDEN_LAYER_SIZE = int(
    os.getenv("RELEVANCE_MLP_HIDDEN_LAYER_SIZE", "64")
)
RELEVANCE_MLP_ALPHA = float(os.getenv("RELEVANCE_MLP_ALPHA", "0.0001"))
RELEVANCE_MLP_MAX_ITER = int(os.getenv("RELEVANCE_MLP_MAX_ITER", "300"))
# Keep this legacy value for old standalone artifact compatibility only.
RELEVANCE_LINEAR_C = float(os.getenv("RELEVANCE_LINEAR_C", "5.0"))

# Fixed preference bonus selected by the chronological ranker benchmark.
SUPER_IMPORTANT_BONUS = float(os.getenv("SUPER_IMPORTANT_BONUS", "0.0"))
assert (
    math.isfinite(SUPER_IMPORTANT_BONUS) and SUPER_IMPORTANT_BONUS >= 0
), "SUPER_IMPORTANT_BONUS must be finite and nonnegative"

# Inverse regularization strength for the logistic-regression urgency head.
# Urgency intentionally shares the same embedding config as relevance for cache
# reuse, but keeps its own classifier regularization.
URGENCY_LINEAR_C = float(os.getenv("URGENCY_LINEAR_C", "1.0"))

# The freshness model uses two cumulative logistic heads over the shared
# embedding cache. These values retain the selected simple linear model.
SEMANTIC_FRESHNESS_LINEAR_C = float(os.getenv("SEMANTIC_FRESHNESS_LINEAR_C", "20.0"))
SEMANTIC_FRESHNESS_WEIGHT_EXPONENT = float(
    os.getenv("SEMANTIC_FRESHNESS_WEIGHT_EXPONENT", "0.375")
)
assert SEMANTIC_FRESHNESS_WEIGHT_EXPONENT > 0
