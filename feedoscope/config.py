import os

DATABASE_URL = os.getenv("DATABASE_URL", "")
assert DATABASE_URL != "", "DATABASE_URL environment variable is not set"

LOGGING_CONFIG = os.getenv("LOGGING_CONFIG", "logging.conf")

# This is the model name used for generating embeddings.
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L12-v2")
assert (
    EMBEDDINGS_MODEL_NAME != ""
), "EMBEDDINGS_MODEL_NAME environment variable is empty"

INFERENCE_MODEL_NAME = os.getenv(
    "INFERENCE_MODEL_NAME", "random_forest_bagging_all-MiniLM-L12-v2.pkl"
)
