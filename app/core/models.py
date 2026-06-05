"""Data models using Pydantic"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class WeatherCondition(str, Enum):
    """Weather condition types"""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    SNOW = "snow"
    THUNDERSTORM = "thunderstorm"
    FOG = "fog"
    WINDY = "windy"


class RiskLevel(str, Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CropType(str, Enum):
    """Supported crop types"""
    MAIZE = "maize"
    WHEAT = "wheat"
    RICE = "rice"
    TEA = "tea"
    COFFEE = "coffee"
    SORGHUM = "sorghum"
    MILLET = "millet"


class Location(BaseModel):
    """Geographic location"""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    timezone: Optional[str] = None


class WeatherData(BaseModel):
    """Weather data from API"""
    location: Location
    temperature: float
    humidity: float
    wind_speed: float
    condition: WeatherCondition
    timestamp: datetime
    precipitation: float = 0.0
    pressure: Optional[float] = None
    visibility: Optional[float] = None


class ForecastData(BaseModel):
    """Weather forecast data"""
    location: Location
    date: datetime
    temperature_high: float
    temperature_low: float
    humidity: float
    precipitation_chance: float
    precipitation_mm: float
    wind_speed: float
    condition: WeatherCondition
    ai_insights: Optional[str] = None


class TreeHealth(BaseModel):
    """Tree health metrics"""
    healthy: int
    needs_care: int
    needs_replacement: int


class TreeAnalysis(BaseModel):
    """Tree analysis results"""
    analysis_id: str
    timestamp: datetime
    farmer_id: str
    county: str
    land_acres: float
    total_tree_count: int
    tree_density_per_acre: float
    confidence_score: float
    canopy_coverage_pct: float
    tree_health: TreeHealth
    tree_species_guess: Optional[str] = None
    observations: List[str] = []
    recommendations: List[str] = []


class RiskFactor(BaseModel):
    """Individual risk factor"""
    type: str
    level: RiskLevel
    probability: float
    description: str
    mitigation: Optional[str] = None


class RiskAssessment(BaseModel):
    """Complete risk assessment"""
    overall_risk: RiskLevel
    risk_factors: List[RiskFactor]
    timestamp: datetime
    location: Location


class PlantingRecommendation(BaseModel):
    """Planting recommendation"""
    recommended: bool
    confidence: float
    optimal_date: Optional[datetime] = None
    reason: str
    risk_factors: List[str] = []
    weather_conditions: List[str] = []


class HarvestingRecommendation(BaseModel):
    """Harvesting recommendation"""
    recommended: bool
    confidence: float
    optimal_date: Optional[datetime] = None
    reason: str
    risk_factors: List[str] = []
    weather_conditions: List[str] = []


class DecisionRequest(BaseModel):
    """Base decision request"""
    location: Location
    crop: CropType
    request_date: datetime = Field(default_factory=datetime.utcnow)


class PlantingDecisionRequest(DecisionRequest):
    """Planting decision request"""
    planting_date: datetime
    field_size_acres: Optional[float] = None


class HarvestingDecisionRequest(DecisionRequest):
    """Harvesting decision request"""
    expected_harvest_date: datetime
    current_growth_stage: Optional[str] = None


class DecisionResponse(BaseModel):
    """Decision response"""
    request_id: str
    timestamp: datetime
    location: Location
    crop: CropType
    decision_type: str
    recommendation: Dict[str, Any]
    weather_data: Optional[WeatherData] = None
    forecast_data: Optional[List[ForecastData]] = None
    risk_assessment: Optional[RiskAssessment] = None
