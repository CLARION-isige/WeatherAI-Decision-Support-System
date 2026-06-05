"""WeatherAI API Client using Singleton Pattern"""

import httpx
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from .models import WeatherData, ForecastData, Location, WeatherCondition

logger = logging.getLogger(__name__)


class WeatherAIClient:
    """
    Singleton WeatherAI API client.
    Ensures only one instance exists throughout the application.
    """
    
    _instance: Optional['WeatherAIClient'] = None
    _client: Optional[httpx.AsyncClient] = None
    
    def __new__(cls) -> 'WeatherAIClient':
        """Implement Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the client (only once)"""
        if self._client is None:
            self.settings = get_settings()
            logger.info(f"Base URL: {self.settings.weatherai_base_url!r}")
            logger.info(f"API Key: {self.settings.weatherai_api_key!r}")
            self._client = httpx.AsyncClient(
                base_url=self.settings.weatherai_base_url,
                timeout=30.0
            )
            logger.info("WeatherAI client initialized")
    
    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        api_key = self.settings.api_key if hasattr(self.settings, 'api_key') else self.settings.weatherai_api_key
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_current_weather(
        self,
        lat: float,
        lon: float,
        units: str = "metric"
    ) -> WeatherData:
        """
        Get current weather conditions for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            units: 'metric' or 'imperial'
        
        Returns:
            WeatherData object
        """
        params = {
            "lat": lat,
            "lon": lon,
            "units": units
        }
        
        response = await self._client.get(
            "/v1/weather",
            headers=self._get_headers(),
            params=params
        )
        response.raise_for_status()
        
        data = response.json()
        return self._parse_weather_data(data)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_forecast(
        self,
        lat: float,
        lon: float,
        days: int = 7,
        units: str = "metric"
    ) -> List[ForecastData]:
        """
        Get weather forecast for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of days to forecast
            units: 'metric' or 'imperial'
        
        Returns:
            List of ForecastData objects
        """
        params = {
            "lat": lat,
            "lon": lon,
            "days": days,
            "units": units
        }
        
        response = await self._client.get(
            "/v1/forecast",
            headers=self._get_headers(),
            params=params
        )
        response.raise_for_status()
        
        data = response.json()
        return self._parse_forecast_data(data)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_insights(
        self,
        lat: float,
        lon: float,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """
        Get AI-powered weather insights.
        
        Args:
            lat: Latitude
            lon: Longitude
            units: 'metric' or 'imperial'
        
        Returns:
            Dictionary with AI insights
        """
        params = {
            "lat": lat,
            "lon": lon,
            "units": units
        }
        
        response = await self._client.get(
            "/v1/insights",
            headers=self._get_headers(),
            params=params
        )
        response.raise_for_status()
        
        return response.json()
    
    async def analyze_trees(
        self,
        image_path: str,
        farmer_id: str,
        county: str,
        land_acres: float,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze tree canopy from image.
        
        Args:
            image_path: Path to image file
            farmer_id: Farmer identifier
            county: County name
            land_acres: Land area in acres
            notes: Additional notes
        
        Returns:
            Tree analysis results
        """
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            data = {
                "farmerId": farmer_id,
                "county": county,
                "landAcres": land_acres
            }
            if notes:
                data["notes"] = notes
            
            api_key = self.settings.api_key if hasattr(self.settings, 'api_key') else self.settings.weatherai_api_key
            response = await self._client.post(
                "/v1/trees/analyze",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data
            )
            response.raise_for_status()
            
            return response.json()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_risk_assessment(
        self,
        lat: float,
        lon: float,
        days: int = 7,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """
        Get AI-powered risk assessment from WeatherAI API.
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of days to assess
            units: 'metric' or 'imperial'
        
        Returns:
            Risk assessment dictionary
        """
        params = {
            "lat": lat,
            "lon": lon,
            "days": days,
            "units": units
        }
        
        response = await self._client.get(
            "/v1/decisions/risk-assessment",
            headers=self._get_headers(),
            params=params
        )
        response.raise_for_status()
        
        return response.json()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_planting_recommendation(
        self,
        lat: float,
        lon: float,
        crop: str,
        days: int = 7,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """
        Get AI-powered planting recommendation from WeatherAI API.
        
        Args:
            lat: Latitude
            lon: Longitude
            crop: Crop type
            days: Number of days to analyze
            units: 'metric' or 'imperial'
        
        Returns:
            Planting recommendation dictionary
        """
        params = {
            "lat": lat,
            "lon": lon,
            "crop": crop,
            "days": days,
            "units": units
        }
        
        response = await self._client.get(
            "/v1/decisions/planting",
            headers=self._get_headers(),
            params=params
        )
        response.raise_for_status()
        
        return response.json()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_harvesting_recommendation(
        self,
        lat: float,
        lon: float,
        crop: str,
        days: int = 7,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """
        Get AI-powered harvesting recommendation from WeatherAI API.
        
        Args:
            lat: Latitude
            lon: Longitude
            crop: Crop type
            days: Number of days to analyze
            units: 'metric' or 'imperial'
        
        Returns:
            Harvesting recommendation dictionary
        """
        params = {
            "lat": lat,
            "lon": lon,
            "crop": crop,
            "days": days,
            "units": units
        }
        
        response = await self._client.get(
            "/v1/decisions/harvesting",
            headers=self._get_headers(),
            params=params
        )
        response.raise_for_status()
        
        return response.json()
    
    def _parse_weather_data(self, data: Dict[str, Any]) -> WeatherData:
        """Parse weather API response into WeatherData model"""
        location = Location(
            lat=data.get("lat", 0),
            lon=data.get("lon", 0),
            city=data.get("city"),
            region=data.get("region"),
            country=data.get("country")
        )
        
        return WeatherData(
            location=location,
            temperature=data.get("temperature", 0),
            humidity=data.get("humidity", 0),
            wind_speed=data.get("wind_speed", 0),
            condition=WeatherCondition(data.get("condition", "clear")),
            timestamp=datetime.now(timezone.utc),
            precipitation=data.get("precipitation", 0),
            pressure=data.get("pressure"),
            visibility=data.get("visibility")
        )
    
    def _parse_forecast_data(self, data: Dict[str, Any]) -> List[ForecastData]:
        """Parse forecast API response into ForecastData models"""
        forecasts = []
        forecast_list = data.get("forecast", [])
        
        for item in forecast_list:
            location = Location(
                lat=data.get("lat", 0),
                lon=data.get("lon", 0),
                city=data.get("city"),
                region=data.get("region"),
                country=data.get("country")
            )
            
            forecast = ForecastData(
                location=location,
                date=datetime.fromisoformat(item.get("date", "")),
                temperature_high=item.get("temp_high", 0),
                temperature_low=item.get("temp_low", 0),
                humidity=item.get("humidity", 0),
                precipitation_chance=item.get("precip_chance", 0),
                precipitation_mm=item.get("precip_mm", 0),
                wind_speed=item.get("wind_speed", 0),
                condition=WeatherCondition(item.get("condition", "clear")),
                ai_insights=item.get("ai_insights")
            )
            forecasts.append(forecast)
        
        return forecasts


# Convenience function to get the singleton instance
def get_weather_client() -> WeatherAIClient:
    """Get the singleton WeatherAI client instance"""
    return WeatherAIClient()
