"""Weather API endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging

from ..core.weather_client import get_weather_client
from ..core.models import WeatherData, ForecastData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/weather", tags=["weather"])


@router.get("/current", response_model=WeatherData)
async def get_current_weather(
    lat: float,
    lon: float,
    units: str = "metric",
    weather_client = Depends(get_weather_client)
):
    """
    Get current weather conditions for a location.
    
    Args:
        lat: Latitude
        lon: Longitude
        units: 'metric' or 'imperial'
    
    Returns:
        Current weather data
    """
    try:
        weather_data = await weather_client.get_current_weather(lat, lon, units)
        return weather_data
    except Exception as e:
        logger.error(f"Error fetching current weather: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast", response_model=List[ForecastData])
async def get_weather_forecast(
    lat: float,
    lon: float,
    days: int = 7,
    units: str = "metric",
    weather_client = Depends(get_weather_client)
):
    """
    Get weather forecast for a location.
    
    Args:
        lat: Latitude
        lon: Longitude
        days: Number of days to forecast (max 14)
        units: 'metric' or 'imperial'
    
    Returns:
        List of forecast data
    """
    try:
        if days > 14:
            days = 14
        
        forecast_data = await weather_client.get_forecast(lat, lon, days, units)
        return forecast_data
    except Exception as e:
        logger.error(f"Error fetching weather forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights")
async def get_weather_insights(
    lat: float,
    lon: float,
    units: str = "metric",
    weather_client = Depends(get_weather_client)
):
    """
    Get AI-powered weather insights.
    
    Args:
        lat: Latitude
        lon: Longitude
        units: 'metric' or 'imperial'
    
    Returns:
        AI insights dictionary
    """
    try:
        insights = await weather_client.get_insights(lat, lon, units)
        return insights
    except Exception as e:
        logger.error(f"Error fetching weather insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))
