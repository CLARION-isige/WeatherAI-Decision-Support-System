"""Pytest configuration and fixtures"""

import pytest
import os
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from app.core.models import (
    Location,
    WeatherData,
    ForecastData,
    WeatherCondition,
    CropType,
    PlantingDecisionRequest,
    HarvestingDecisionRequest,
    TreeAnalysis,
    TreeHealth
)


@pytest.fixture
def mock_location() -> Location:
    """Fixture for mock location"""
    return Location(
        lat=-1.2921,
        lon=36.8219,
        city="Nairobi",
        region="Nairobi County",
        country="KE",
        timezone="Africa/Nairobi"
    )


@pytest.fixture
def mock_weather_data(mock_location: Location) -> WeatherData:
    """Fixture for mock weather data"""
    return WeatherData(
        location=mock_location,
        temperature=25.5,
        humidity=65,
        wind_speed=12.3,
        condition=WeatherCondition.CLEAR,
        timestamp=datetime.now(timezone.utc),
        precipitation=0.0,
        pressure=1013.25,
        visibility=10.0
    )


@pytest.fixture
def mock_forecast_data(mock_location: Location) -> list[ForecastData]:
    """Fixture for mock forecast data (7 days)"""
    forecasts = []
    for i in range(7):
        forecast = ForecastData(
            location=mock_location,
            date=datetime.now(timezone.utc),
            temperature_high=28 + i,
            temperature_low=18 + i,
            humidity=60 + i * 2,
            precipitation_chance=30 - i * 3,
            precipitation_mm=5.0 - i * 0.5,
            wind_speed=10 + i,
            condition=WeatherCondition.CLEAR,
            ai_insights=f"Day {i+1} weather summary"
        )
        forecasts.append(forecast)
    return forecasts


@pytest.fixture
def mock_planting_request(mock_location: Location) -> PlantingDecisionRequest:
    """Fixture for mock planting decision request"""
    return PlantingDecisionRequest(
        location=mock_location,
        crop=CropType.MAIZE,
        planting_date=datetime.now(timezone.utc),
        field_size_acres=5.0
    )


@pytest.fixture
def mock_harvesting_request(mock_location: Location) -> HarvestingDecisionRequest:
    """Fixture for mock harvesting decision request"""
    return HarvestingDecisionRequest(
        location=mock_location,
        crop=CropType.MAIZE,
        expected_harvest_date=datetime.now(timezone.utc),
        current_growth_stage="flowering"
    )


@pytest.fixture
def mock_tree_analysis() -> TreeAnalysis:
    """Fixture for mock tree analysis"""
    return TreeAnalysis(
        analysis_id="test-analysis-123",
        timestamp=datetime.now(timezone.utc),
        farmer_id="F-001",
        county="Bomet",
        land_acres=2.5,
        total_tree_count=84,
        tree_density_per_acre=33.6,
        confidence_score=0.87,
        canopy_coverage_pct=41.2,
        tree_health=TreeHealth(
            healthy=68,
            needs_care=12,
            needs_replacement=4
        ),
        tree_species_guess="Tea (Camellia sinensis)",
        observations=[
            "Dense canopy in northern quadrant",
            "3 trees near water source show yellowing"
        ],
        recommendations=[
            "Consider thinning northern section",
            "Improve drainage around water source"
        ]
    )


@pytest.fixture
def mock_weather_client():
    """Fixture for mock weather client"""
    client = Mock()
    client.get_current_weather = AsyncMock()
    client.get_forecast = AsyncMock()
    client.get_insights = AsyncMock()
    client.analyze_trees = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_settings():
    """Fixture for mock settings"""
    settings = Mock()
    settings.app_name = "WeatherAI Decision Support System"
    settings.app_version = "1.0.0"
    settings.debug = True
    settings.log_level = "INFO"
    settings.weatherai_api_key = "test_api_key"
    settings.weatherai_base_url = "https://api.weather-ai.co"
    settings.firebase_project_id = None
    settings.firebase_service_account_key = None
    settings.model_cache_dir = "./models"
    settings.retrain_interval_days = 30
    return settings


@pytest.fixture
def sample_weather_api_response() -> Dict[str, Any]:
    """Fixture for sample WeatherAI API response"""
    return {
        "lat": -1.2921,
        "lon": 36.8219,
        "city": "Nairobi",
        "region": "Nairobi County",
        "country": "KE",
        "temperature": 25.5,
        "humidity": 65,
        "wind_speed": 12.3,
        "condition": "clear",
        "precipitation": 0.0,
        "pressure": 1013.25,
        "visibility": 10.0
    }


@pytest.fixture
def sample_forecast_api_response() -> Dict[str, Any]:
    """Fixture for sample forecast API response"""
    return {
        "lat": -1.2921,
        "lon": 36.8219,
        "city": "Nairobi",
        "region": "Nairobi County",
        "country": "KE",
        "forecast": [
            {
                "date": "2024-06-15T00:00:00Z",
                "temp_high": 28,
                "temp_low": 18,
                "humidity": 60,
                "precip_chance": 30,
                "precip_mm": 5.0,
                "wind_speed": 10,
                "condition": "clear",
                "ai_insights": "Good weather expected"
            }
        ]
    }


@pytest.fixture
def sample_tree_analysis_response() -> Dict[str, Any]:
    """Fixture for sample tree analysis API response"""
    return {
        "analysis_id": "test-analysis-123",
        "timestamp": "2024-06-15T09:15:00.000Z",
        "farmer_id": "F-001",
        "county": "Bomet",
        "location": "Kapkimolwa Farm, Block C",
        "land_acres": 2.5,
        "total_tree_count": 84,
        "tree_density_per_acre": 33.6,
        "confidence_score": 0.87,
        "canopy_coverage_pct": 41.2,
        "tree_health": {
            "healthy": 68,
            "needs_care": 12,
            "needs_replacement": 4
        },
        "low_confidence": False,
        "tree_species_guess": "Tea (Camellia sinensis)",
        "observations": [
            "Dense canopy in northern quadrant",
            "3 trees near water source show yellowing"
        ],
        "recommendations": [
            "Consider thinning northern section",
            "Improve drainage around water source"
        ],
        "original_image_url": "https://storage.googleapis.com/.../original.jpg",
        "overlay_image_url": "https://storage.googleapis.com/.../overlay.jpg",
        "cv_debug": {
            "orig_resolution": "4000x3000",
            "work_resolution": "1500x1125",
            "canopy_px": 412500,
            "peaks_detected": 91,
            "after_area_filter": 84
        }
    }


@pytest.fixture(autouse=True)
def set_test_env():
    """Set test environment variables"""
    os.environ["WEATHERAI_API_KEY"] = "test_api_key"
    os.environ["DEBUG"] = "True"
    yield
    # Cleanup
    os.environ.pop("WEATHERAI_API_KEY", None)
    os.environ.pop("DEBUG", None)
