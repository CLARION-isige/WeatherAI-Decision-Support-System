"""Configuration management using Pydantic Settings"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Application
    app_name: str = Field(default="WeatherAI Decision Support System")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    
    # WeatherAI API
    weatherai_api_key: str = Field(default="", alias="WEATHERAI_API_KEY")
    weatherai_base_url: str = Field(
        default="https://api.weather-ai.co",
        alias="WEATHERAI_BASE_URL"
    )
    
    # Legacy support for WEATHER_API_KEY (without AI)
    @property
    def api_key(self) -> str:
        """Get API key, checking both old and new env var names"""
        import os
        return os.environ.get("WEATHERAI_API_KEY") or os.environ.get("WEATHER_API_KEY", "")
    
    # ML Configuration
    model_cache_dir: str = Field(default="./models", alias="MODEL_CACHE_DIR")
    retrain_interval_days: int = Field(default=30, alias="RETRAIN_INTERVAL_DAYS")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get singleton settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
