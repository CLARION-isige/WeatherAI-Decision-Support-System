"""Unit tests for Pydantic data models"""

import pytest
from datetime import datetime,timezone
from pydantic import ValidationError

from app.core.models import (
    Location,
    WeatherData,
    ForecastData,
    WeatherCondition,
    RiskLevel,
    CropType,
    TreeHealth,
    TreeAnalysis,
    RiskFactor,
    RiskAssessment,
    PlantingRecommendation,
    HarvestingRecommendation,
    PlantingDecisionRequest,
    HarvestingDecisionRequest,
    DecisionResponse
)


class TestLocation:
    """Test suite for Location model"""
    
    def test_location_creation(self):
        """Test location model creation"""
        location = Location(
            lat=-1.2921,
            lon=36.8219,
            city="Nairobi",
            region="Nairobi County",
            country="KE"
        )
        assert location.lat == -1.2921
        assert location.lon == 36.8219
        assert location.city == "Nairobi"
    
    def test_location_validation_lat_bounds(self):
        """Test latitude validation bounds"""
        # Valid latitude
        Location(lat=0, lon=0)
        Location(lat=90, lon=0)
        Location(lat=-90, lon=0)
        
        # Invalid latitude
        with pytest.raises(ValidationError):
            Location(lat=91, lon=0)
        with pytest.raises(ValidationError):
            Location(lat=-91, lon=0)
    
    def test_location_validation_lon_bounds(self):
        """Test longitude validation bounds"""
        # Valid longitude
        Location(lat=0, lon=0)
        Location(lat=0, lon=180)
        Location(lat=0, lon=-180)
        
        # Invalid longitude
        with pytest.raises(ValidationError):
            Location(lat=0, lon=181)
        with pytest.raises(ValidationError):
            Location(lat=0, lon=-181)
    
    def test_location_optional_fields(self):
        """Test location with optional fields"""
        location = Location(lat=0, lon=0)
        assert location.city is None
        assert location.region is None
        assert location.country is None


class TestWeatherData:
    """Test suite for WeatherData model"""
    
    def test_weather_data_creation(self, mock_location):
        """Test weather data model creation"""
        weather = WeatherData(
            location=mock_location,
            temperature=25.5,
            humidity=65,
            wind_speed=12.3,
            condition=WeatherCondition.CLEAR,
            timestamp=datetime.now(timezone.utc)
        )
        assert weather.temperature == 25.5
        assert weather.humidity == 65
        assert weather.condition == WeatherCondition.CLEAR
    
    def test_weather_data_defaults(self, mock_location):
        """Test weather data default values"""
        weather = WeatherData(
            location=mock_location,
            temperature=25.5,
            humidity=65,
            wind_speed=12.3,
            condition=WeatherCondition.CLEAR,
            timestamp=datetime.now(timezone.utc)
        )
        assert weather.precipitation == 0.0
        assert weather.pressure is None
        assert weather.visibility is None


class TestForecastData:
    """Test suite for ForecastData model"""
    
    def test_forecast_data_creation(self, mock_location):
        """Test forecast data model creation"""
        forecast = ForecastData(
            location=mock_location,
            date=datetime.now(timezone.utc),
            temperature_high=28,
            temperature_low=18,
            humidity=60,
            precipitation_chance=30,
            precipitation_mm=5.0,
            wind_speed=10,
            condition=WeatherCondition.CLEAR
        )
        assert forecast.temperature_high == 28
        assert forecast.temperature_low == 18
        assert forecast.precipitation_chance == 30
    
    def test_forecast_data_with_ai_insights(self, mock_location):
        """Test forecast data with AI insights"""
        forecast = ForecastData(
            location=mock_location,
            date=datetime.now(timezone.utc),
            temperature_high=28,
            temperature_low=18,
            humidity=60,
            precipitation_chance=30,
            precipitation_mm=5.0,
            wind_speed=10,
            condition=WeatherCondition.CLEAR,
            ai_insights="Good weather expected"
        )
        assert forecast.ai_insights == "Good weather expected"


class TestTreeHealth:
    """Test suite for TreeHealth model"""
    
    def test_tree_health_creation(self):
        """Test tree health model creation"""
        health = TreeHealth(
            healthy=68,
            needs_care=12,
            needs_replacement=4
        )
        assert health.healthy == 68
        assert health.needs_care == 12
        assert health.needs_replacement == 4
    
    def test_tree_health_all_healthy(self):
        """Test tree health with all healthy"""
        health = TreeHealth(healthy=100, needs_care=0, needs_replacement=0)
        assert health.healthy == 100


class TestTreeAnalysis:
    """Test suite for TreeAnalysis model"""
    
    def test_tree_analysis_creation(self, mock_location):
        """Test tree analysis model creation"""
        analysis = TreeAnalysis(
            analysis_id="test-123",
            timestamp=datetime.now(timezone.utc),
            farmer_id="F-001",
            county="Bomet",
            land_acres=2.5,
            total_tree_count=84,
            tree_density_per_acre=33.6,
            confidence_score=0.87,
            canopy_coverage_pct=41.2,
            tree_health=TreeHealth(healthy=68, needs_care=12, needs_replacement=4)
        )
        assert analysis.analysis_id == "test-123"
        assert analysis.total_tree_count == 84
        assert analysis.confidence_score == 0.87
    
    def test_tree_analysis_with_optional_fields(self, mock_location):
        """Test tree analysis with optional fields"""
        analysis = TreeAnalysis(
            analysis_id="test-123",
            timestamp=datetime.now(timezone.utc),
            farmer_id="F-001",
            county="Bomet",
            land_acres=2.5,
            total_tree_count=84,
            tree_density_per_acre=33.6,
            confidence_score=0.87,
            canopy_coverage_pct=41.2,
            tree_health=TreeHealth(healthy=68, needs_care=12, needs_replacement=4),
            tree_species_guess="Tea",
            observations=["Good canopy"],
            recommendations=["Maintain current practices"]
        )
        assert analysis.tree_species_guess == "Tea"
        assert len(analysis.observations) == 1
        assert len(analysis.recommendations) == 1


class TestRiskFactor:
    """Test suite for RiskFactor model"""
    
    def test_risk_factor_creation(self):
        """Test risk factor model creation"""
        factor = RiskFactor(
            type="frost",
            level=RiskLevel.HIGH,
            probability=0.7,
            description="High frost risk expected"
        )
        assert factor.type == "frost"
        assert factor.level == RiskLevel.HIGH
        assert factor.probability == 0.7
    
    def test_risk_factor_with_mitigation(self):
        """Test risk factor with mitigation"""
        factor = RiskFactor(
            type="drought",
            level=RiskLevel.MEDIUM,
            probability=0.5,
            description="Moderate drought risk",
            mitigation="Ensure irrigation systems are ready"
        )
        assert factor.mitigation == "Ensure irrigation systems are ready"


class TestRiskAssessment:
    """Test suite for RiskAssessment model"""
    
    def test_risk_assessment_creation(self, mock_location):
        """Test risk assessment model creation"""
        assessment = RiskAssessment(
            overall_risk=RiskLevel.HIGH,
            risk_factors=[],
            timestamp=datetime.now(timezone.utc),
            location=mock_location
        )
        assert assessment.overall_risk == RiskLevel.HIGH
        assert assessment.location == mock_location
    
    def test_risk_assessment_with_factors(self, mock_location):
        """Test risk assessment with risk factors"""
        factors = [
            RiskFactor(
                type="frost",
                level=RiskLevel.HIGH,
                probability=0.7,
                description="Frost risk"
            )
        ]
        assessment = RiskAssessment(
            overall_risk=RiskLevel.HIGH,
            risk_factors=factors,
            timestamp=datetime.now(timezone.utc),
            location=mock_location
        )
        assert len(assessment.risk_factors) == 1


class TestPlantingRecommendation:
    """Test suite for PlantingRecommendation model"""
    
    def test_planting_recommendation_creation(self):
        """Test planting recommendation model creation"""
        recommendation = PlantingRecommendation(
            recommended=True,
            confidence=0.85,
            optimal_date=datetime.now(timezone.utc),
            reason="Conditions are favorable"
        )
        assert recommendation.recommended == True
        assert recommendation.confidence == 0.85
    
    def test_planting_recommendation_not_recommended(self):
        """Test planting recommendation when not recommended"""
        recommendation = PlantingRecommendation(
            recommended=False,
            confidence=0.3,
            reason="Conditions not suitable"
        )
        assert recommendation.recommended == False
        assert recommendation.confidence == 0.3
        assert recommendation.optimal_date is None


class TestHarvestingRecommendation:
    """Test suite for HarvestingRecommendation model"""
    
    def test_harvesting_recommendation_creation(self):
        """Test harvesting recommendation model creation"""
        recommendation = HarvestingRecommendation(
            recommended=True,
            confidence=0.9,
            optimal_date=datetime.now(timezone.utc),
            reason="Optimal harvesting conditions"
        )
        assert recommendation.recommended == True
        assert recommendation.confidence == 0.9


class TestPlantingDecisionRequest:
    """Test suite for PlantingDecisionRequest model"""
    
    def test_planting_decision_request_creation(self, mock_location):
        """Test planting decision request creation"""
        request = PlantingDecisionRequest(
            location=mock_location,
            crop=CropType.MAIZE,
            planting_date=datetime.now(timezone.utc),
            field_size_acres=5.0
        )
        assert request.crop == CropType.MAIZE
        assert request.field_size_acres == 5.0
    
    def test_planting_decision_request_defaults(self, mock_location):
        """Test planting decision request with defaults"""
        request = PlantingDecisionRequest(
            location=mock_location,
            crop=CropType.MAIZE,
            planting_date=datetime.now(timezone.utc)
        )
        assert request.field_size_acres is None


class TestHarvestingDecisionRequest:
    """Test suite for HarvestingDecisionRequest model"""
    
    def test_harvesting_decision_request_creation(self, mock_location):
        """Test harvesting decision request creation"""
        request = HarvestingDecisionRequest(
            location=mock_location,
            crop=CropType.WHEAT,
            expected_harvest_date=datetime.now(timezone.utc),
            current_growth_stage="flowering"
        )
        assert request.crop == CropType.WHEAT
        assert request.current_growth_stage == "flowering"


class TestDecisionResponse:
    """Test suite for DecisionResponse model"""
    
    def test_decision_response_creation(self, mock_location):
        """Test decision response creation"""
        response = DecisionResponse(
            request_id="req-123",
            timestamp=datetime.now(timezone.utc),
            location=mock_location,
            crop=CropType.MAIZE,
            decision_type="planting",
            recommendation={}
        )
        assert response.request_id == "req-123"
        assert response.decision_type == "planting"
    
    def test_decision_response_with_full_data(self, mock_location, mock_weather_data):
        """Test decision response with full data"""
        response = DecisionResponse(
            request_id="req-123",
            timestamp=datetime.now(timezone.utc),
            location=mock_location,
            crop=CropType.MAIZE,
            decision_type="planting",
            recommendation={"recommended": True},
            weather_data=mock_weather_data,
            forecast_data=[],
            risk_assessment=None
        )
        assert response.weather_data is not None
        assert response.decision_type == "planting"


class TestEnums:
    """Test suite for enum values"""
    
    def test_weather_condition_values(self):
        """Test weather condition enum values"""
        assert WeatherCondition.CLEAR.value == "clear"
        assert WeatherCondition.CLOUDY.value == "cloudy"
        assert WeatherCondition.RAIN.value == "rain"
        assert WeatherCondition.SNOW.value == "snow"
        assert WeatherCondition.THUNDERSTORM.value == "thunderstorm"
        assert WeatherCondition.FOG.value == "fog"
        assert WeatherCondition.WINDY.value == "windy"
    
    def test_risk_level_values(self):
        """Test risk level enum values"""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"
    
    def test_crop_type_values(self):
        """Test crop type enum values"""
        assert CropType.MAIZE.value == "maize"
        assert CropType.WHEAT.value == "wheat"
        assert CropType.RICE.value == "rice"
        assert CropType.TEA.value == "tea"
        assert CropType.COFFEE.value == "coffee"
        assert CropType.SORGHUM.value == "sorghum"
        assert CropType.MILLET.value == "millet"
