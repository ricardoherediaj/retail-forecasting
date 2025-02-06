from dataclasses import dataclass
from pathlib import Path
from paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR, CACHE_DIR

@dataclass
class Config:
    """Application configuration"""
    RAW_DATA_DIR: Path = RAW_DATA_DIR
    TRANSFORMED_DATA_DIR: Path = TRANSFORMED_DATA_DIR
    CACHE_DIR: Path = CACHE_DIR
    DATABASE_NAME: str = 'supermarket.db'
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'

    # Database configuration
    DB_TYPE: str = 'sqlite'

    # Validation thresholds
    MIN_PRICE: float = 0.0
    MAX_PRICE: float = 1000.0

    @property
    def database_path(self) -> Path:
        return self.RAW_DATA_DIR / self.DATABASE_NAME