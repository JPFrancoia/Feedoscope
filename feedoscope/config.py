import os

DATABASE_URL = os.getenv("DATABASE_URL", "")


assert DATABASE_URL != "", "DATABASE_URL environment variable is not set"
