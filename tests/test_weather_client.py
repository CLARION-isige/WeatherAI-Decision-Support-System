"""Unit tests for WeatherAI API client"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import httpx

from app.core.weather_client import WeatherAIClient, get_weather_client
from app.core.models import WeatherData, ForecastData, Location, WeatherCondition
from datetime import datetime


class TestWeatherAIClientSingleton:
    """Test suite for WeatherAIClient singleton pattern"""
    
    def test_singleton_pattern(self):
        """Test that WeatherAIClient implements singleton pattern"""
        client1 = WeatherAIClient()
        client2 = WeatherAIClient()
        assert client1 is client2
    
    def test_get_weather_client_singleton(self):
        """Test that get_weather_client returns singleton instance"""
        client1 = get_weather_client()
        client2 = get_weather_client()
        assert client1 is client2


class TestWeatherAIClient:
    """Test suite for WeatherAIClient functionality"""
    
    @pytest.fixture
    def client(self):
        """Fixture for weather client"""
        # Reset singleton for testing
        from app.core.weather_client import WeatherAIClient
        WeatherAIClient._instance = None
        WeatherAIClient._client = None
        return WeatherAIClient()
    
    def test_client_initialization(self, client):
        """Test client initialization"""
        assert client is not None
        assert client._client is not None
        assert client.settings is not None
    
    def test_get_headers(self, client):
        """Test header generation"""
        headers = client._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert "Content-Type" in headers
    
    @pytest.mark.asyncio
    async def test_get_current_weather_success(self, client, sample_weather_api_response):
        """Test successful current weather fetch"""
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.json.return_value = sample_weather_api_response
        mock_response.raise_for_status = Mock()
        
        client._client.get = AsyncMock(return_value=mock_response)
        
        weather = await client.get_current_weather(lat=-1.2921, lon=36.8219)
        
        assert weather is not None
        assert isinstance(weather, WeatherData)
        assert weather.location.lat == -1.2921
        assert weather.location.lon == 36.8219
        expected_url = f"{client.settings.weatherai_base_url.rstrip('/')}/v1/weather"
        assert client._client.get.call_args.args[0] == expected_url
    
    @pytest.mark.asyncio
    async def test_get_current_weather_with_units(self, client, sample_weather_api_response):
        """Test current weather fetch with imperial units"""
        mock_response = Mock()
        mock_response.json.return_value = sample_weather_api_response
        mock_response.raise_for_status = Mock()
        
        client._client.get = AsyncMock(return_value=mock_response)
        
        weather = await client.get_current_weather(lat=-1.2921, lon=36.8219, units="imperial")
        
        assert weather is not None
        client._client.get.assert_called_once()
        assert client._client.get.call_args.kwargs["params"]["units"] == "imperial"
    
    @pytest.mark.asyncio
    async def test_get_forecast_success(self, client, sample_forecast_api_response):
        """Test successful forecast fetch"""
        mock_response = Mock()
        mock_response.json.return_value = sample_forecast_api_response
        mock_response.raise_for_status = Mock()
        
        client._client.get = AsyncMock(return_value=mock_response)
        
        forecast = await client.get_forecast(lat=-1.2921, lon=36.8219, days=7)
        
        assert forecast is not None
        assert isinstance(forecast, list)
        assert len(forecast) > 0
        assert isinstance(forecast[0], ForecastData)
    
    @pytest.mark.asyncio
    async def test_get_forecast_days_limit(self, client, sample_forecast_api_response):
        """Test forecast fetch respects days limit"""
        mock_response = Mock()
        mock_response.json.return_value = sample_forecast_api_response
        mock_response.raise_for_status = Mock()
        
        client._client.get = AsyncMock(return_value=mock_response)
        
        forecast = await client.get_forecast(lat=-1.2921, lon=36.8219, days=14)
        
        assert forecast is not None
        # Check that days parameter was passed
        call_args = client._client.get.call_args
        assert call_args[1]["params"]["days"] == 14
    
    @pytest.mark.asyncio
    async def test_get_insights_success(self, client):
        """Test successful insights fetch"""
        mock_response = Mock()
        mock_response.json.return_value = {"insights": "AI-generated summary"}
        mock_response.raise_for_status = Mock()
        
        client._client.get = AsyncMock(return_value=mock_response)
        
        insights = await client.get_insights(lat=-1.2921, lon=36.8219)
        
        assert insights is not None
        assert isinstance(insights, dict)
    
    @pytest.mark.asyncio
    async def test_analyze_trees_success(self, client, sample_tree_analysis_response, tmp_path):
        """Test successful tree analysis"""
        # Create a temporary image file
        image_path = tmp_path / "test_image.jpg"
        image_path.write_bytes(b"fake image data")
        
        mock_response = Mock()
        mock_response.json.return_value = sample_tree_analysis_response
        mock_response.raise_for_status = Mock()
        
        client._client.post = AsyncMock(return_value=mock_response)
        
        analysis = await client.analyze_trees(
            image_path=str(image_path),
            farmer_id="F-001",
            county="Bomet",
            land_acres=2.5
        )
        
        assert analysis is not None
        assert analysis["analysis_id"] == "test-analysis-123"
    
    @pytest.mark.asyncio
    async def test_close_client(self, client):
        """Test client cleanup"""
        await client.close()
        assert client._client is None
    
    def test_parse_weather_data(self, client, sample_weather_api_response):
        """Test weather data parsing"""
        weather = client._parse_weather_data(sample_weather_api_response)
        
        assert weather is not None
        assert isinstance(weather, WeatherData)
        assert weather.location.lat == -1.2921
        assert weather.location.lon == 36.8219
        assert weather.temperature == 25.5
    
    def test_parse_forecast_data(self, client, sample_forecast_api_response):
        """Test forecast data parsing"""
        forecasts = client._parse_forecast_data(sample_forecast_api_response)
        
        assert forecasts is not None
        assert isinstance(forecasts, list)
        assert len(forecasts) > 0
        assert isinstance(forecasts[0], ForecastData)
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self, client):
        """Test that client retries on failure"""
        # Mock failed responses then success
        mock_response = Mock()
        mock_response.json.return_value = {"lat": -1.2921, "lon": 36.8219}
        mock_response.raise_for_status = Mock()
        
        client._client.get = AsyncMock(side_effect=[
            httpx.HTTPError("Connection error"),
            httpx.HTTPError("Connection error"),
            mock_response
        ])
        
        # This should succeed after retries
        weather = await client.get_current_weather(lat=-1.2921, lon=36.8219)
        
        assert weather is not None
        assert client._client.get.call_count == 3


class TestWeatherClientIntegration:
    """Integration tests for weather client with actual API calls"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_call_current_weather(self):
        """Test real API call for current weather (integration test)"""
        # This test requires a valid API key
        # Skip if not in integration mode
        pytest.skip("Skip integration test without API key")
        
        client = get_weather_client()
        try:
            weather = await client.get_current_weather(lat=-1.2921, lon=36.8219)
            assert weather is not None
            assert isinstance(weather, WeatherData)
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")
