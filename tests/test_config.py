"""Unit tests for configuration management"""

import pytest
import os
from unittest.mock import patch
from pathlib import Path

from app.config import Settings, get_settings


class TestSettings:
    """Test Settings class"""
    
    def test_settings_initialization(self):
        """Test that Settings can be initialized"""
        settings = Settings(
            weatherai_api_key="test_api_key",
            app_name="Test App"
        )
        assert settings.weatherai_api_key == "test_api_key"
        assert settings.app_name == "Test App"
    
    def test_settings_defaults(self):
        """Test that Settings has correct defaults"""
        settings = Settings(weatherai_api_key="test_key")
        assert settings.app_name == "WeatherAI Decision Support System"
        assert settings.app_version == "1.0.0"
        assert settings.debug == True
        assert settings.log_level == "INFO"
        assert settings.weatherai_base_url == "https://api.weather-ai.co"
        assert settings.model_cache_dir == "./models"
        assert settings.retrain_interval_days == 30
    
    def test_settings_from_env(self):
        """Test that Settings loads from environment variables"""
        with patch.dict(os.environ, {
            'WEATHERAI_API_KEY': 'env_test_key',
            'APP_NAME': 'Env Test App',
            'DEBUG': 'False',
            'LOG_LEVEL': 'DEBUG'
        }):
            settings = Settings()
            assert settings.weatherai_api_key == 'env_test_key'
            assert settings.app_name == 'Env Test App'
            assert settings.debug == False
            assert settings.log_level == 'DEBUG'
    
    def test_settings_validation(self):
        """Test that Settings validates required fields"""
        import os
        # Temporarily remove the environment variable
        old_key = os.environ.pop('WEATHERAI_API_KEY', None)
        # Also need to clear the singleton
        from app.config import _settings
        import app.config
        app.config._settings = None
        try:
            # Settings should raise ValidationError when required field is missing
            # Since the conftest fixture sets the env variable, we need to ensure it's actually gone
            assert 'WEATHERAI_API_KEY' not in os.environ
            from pydantic import ValidationError
            with pytest.raises(ValidationError):
                Settings()  # Missing required weatherai_api_key
        finally:
            # Restore the environment variable
            os.environ['WEATHERAI_API_KEY'] = old_key if old_key else 'test_api_key'
            # Clear singleton again
            app.config._settings = None


class TestGetSettings:
    """Test get_settings singleton function"""
    
    def test_get_settings_returns_singleton(self):
        """Test that get_settings returns the same instance"""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
    
    def test_get_settings_initializes_once(self):
        """Test that get_settings only initializes once"""
        # Clear the singleton
        from app.config import _settings
        import app.config
        app.config._settings = None
        
        settings = get_settings()
        assert settings is not None
        assert isinstance(settings, Settings)
