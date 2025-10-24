from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    base_path: Path = Path(__file__).parent.parent / "data"
    cache_dir: Path = base_path / "cache"

    binance_base_url: str = "https://api.binance.com"
    binance_rate_limit_per_minute: int = 1200
    binance_request_timeout: int = 30

    default_fee_rate: float = 0.001
    default_slippage_bps: float = 5.0

    max_retry_attempts: int = 5
    retry_initial_wait: float = 1.0
    retry_max_wait: float = 60.0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
