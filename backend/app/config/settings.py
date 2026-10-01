"""
Backend configuration settings
"""

import os
from pydantic import BaseModel

class Settings(BaseModel):
    APP_NAME: str = "MedClass AI API"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "Intelligent Medical Question Understanding & Multi-Task Classification"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: list = ["*"]
    MAX_QUESTION_LENGTH: int = 1000
    MIN_QUESTION_LENGTH: int = 3
    ENVIRONMENT: str = os.getenv("ENV", "development")

settings = Settings()
