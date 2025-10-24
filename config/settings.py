from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    base_path: Path = Path(__file__).parent.parent / "data"
    cache_dir: Path = base_path / "cache"
    preset_storage_path: Path = base_path / "presets" / "presets.yaml"
    last_session_state_path: Path = base_path / "presets" / "last_state.yaml"

    binance_base_url: str = "https://api.binance.com"
    binance_rate_limit_per_minute: int = 1200
    binance_request_timeout: int = 30
    binance_klines_limit: int = 1000

    default_fee_rate: float = 0.001
    default_slippage_bps: float = 5.0

    max_retry_attempts: int = 5
    retry_initial_wait: float = 1.0
    retry_max_wait: float = 60.0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.preset_storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_session_state_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
