"""Unit tests for Decision Engine"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timezone

from app.core.decision_engine import (
    DecisionEngine,
    WeatherDataHandler,
    ForecastDataHandler,
    RiskAssessmentHandler,
    PlantingDecisionHandler,
    HarvestingDecisionHandler,
    ResponseBuilderHandler
)
from app.core.models import (
    PlantingDecisionRequest,
    HarvestingDecisionRequest,
    DecisionResponse,
    Location,
    CropType,
    ForecastData,
    WeatherData,
    WeatherCondition,
    RiskAssessment,
    RiskLevel
)
from app.ml.planting_model import PlantingPredictorModel
from app.ml.risk_model import RiskAssessmentModel


class TestDecisionHandlers:
    """Test suite for individual decision handlers"""
    
    @pytest.fixture
    def mock_weather_client(self):
        """Fixture for mock weather client"""
        client = Mock()
        client.get_current_weather = AsyncMock()
        client.get_forecast = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_planting_model(self):
        """Fixture for mock planting model"""
        model = Mock(spec=PlantingPredictorModel)
        return model
    
    @pytest.fixture
    def mock_risk_model(self):
        """Fixture for mock risk model"""
        model = Mock(spec=RiskAssessmentModel)
        return model
    
    @pytest.fixture
    def mock_location(self):
        """Fixture for mock location"""
        return Location(lat=-1.2921, lon=36.8219, region="Nairobi")
    
    @pytest.fixture
    def mock_request(self, mock_location):
        """Fixture for mock decision request"""
        return PlantingDecisionRequest(
            location=mock_location,
            crop=CropType.MAIZE,
            planting_date=datetime.now(timezone.utc)
        )
    
    def test_weather_data_handler(self, mock_weather_client, mock_request):
        """Test WeatherDataHandler"""
        handler = WeatherDataHandler(mock_weather_client)
        
        # Mock weather data
        mock_weather = WeatherData(
            location=mock_request.location,
            temperature=25.5,
            humidity=65,
            wind_speed=12.3,
            condition=WeatherCondition.CLEAR,
            timestamp=datetime.now(timezone.utc)
        )
        mock_weather_client.get_current_weather = AsyncMock(return_value=mock_weather)
        
        context = {}
        result = handler.handle(mock_request, context)
        
        # Since it's async, we need to await it
        import asyncio
        result = asyncio.run(handler.handle(mock_request, context))
        
        assert "weather_data" in result
        assert result["weather_data"] == mock_weather
    
    def test_forecast_data_handler(self, mock_weather_client, mock_request):
        """Test ForecastDataHandler"""
        handler = ForecastDataHandler(mock_weather_client)
        
        # Mock forecast data
        mock_forecast = [
            ForecastData(
                location=mock_request.location,
                date=datetime.now(timezone.utc),
                temperature_high=28,
                temperature_low=18,
                humidity=60,
                precipitation_chance=30,
                precipitation_mm=5.0,
                wind_speed=10,
                condition=WeatherCondition.CLEAR
            )
        ]
        mock_weather_client.get_forecast = AsyncMock(return_value=mock_forecast)
        
        import asyncio
        result = asyncio.run(handler.handle(mock_request, {}))
        
        assert "forecast_data" in result
        assert len(result["forecast_data"]) == 1
    
    def test_risk_assessment_handler(self, mock_risk_model, mock_request, mock_location):
        """Test RiskAssessmentHandler"""
        handler = RiskAssessmentHandler(mock_risk_model)
        
        # Mock risk assessment
        mock_risk = RiskAssessment(
            overall_risk=RiskLevel.LOW,
            risk_factors=[],
            timestamp=datetime.now(timezone.utc),
            location=mock_location
        )
        mock_risk_model.assess_risks = Mock(return_value=mock_risk)
        
        # Provide non-empty forecast data
        from app.core.models import ForecastData, WeatherCondition
        forecast_data = [
            ForecastData(
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
        ]
        context = {"forecast_data": forecast_data}
        
        import asyncio
        result = asyncio.run(handler.handle(mock_request, context))
        
        assert "risk_assessment" in result
        assert result["risk_assessment"] == mock_risk
    
    def test_handler_chain_of_responsibility(self, mock_weather_client):
        """Test chain of responsibility pattern"""
        handler1 = WeatherDataHandler(mock_weather_client)
        handler2 = ForecastDataHandler(mock_weather_client)
        
        # Set up chain
        handler1.set_next(handler2)
        
        assert handler1._next_handler == handler2
    
    def test_handler_pass_to_next(self, mock_weather_client, mock_request):
        """Test that handlers pass to next in chain"""
        handler1 = WeatherDataHandler(mock_weather_client)
        handler2 = Mock()
        handler2.handle = AsyncMock(return_value={"next": True})
        
        handler1.set_next(handler2)
        
        import asyncio
        result = asyncio.run(handler1.handle(mock_request, {}))
        
        assert handler2.handle.called


class TestDecisionEngine:
    """Test suite for DecisionEngine"""
    
    @pytest.fixture
    def mock_weather_client(self):
        """Fixture for mock weather client"""
        client = Mock()
        client.get_current_weather = AsyncMock()
        client.get_forecast = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_planting_model(self):
        """Fixture for mock planting model"""
        model = PlantingPredictorModel()
        return model
    
    @pytest.fixture
    def mock_risk_model(self):
        """Fixture for mock risk model"""
        model = RiskAssessmentModel()
        return model
    
    @pytest.fixture
    def engine(self, mock_weather_client, mock_planting_model, mock_risk_model):
        """Fixture for decision engine"""
        return DecisionEngine(
            weather_client=mock_weather_client,
            planting_model=mock_planting_model,
            risk_model=mock_risk_model
        )
    
    @pytest.fixture
    def mock_location(self):
        """Fixture for mock location"""
        return Location(lat=-1.2921, lon=36.8219, region="Nairobi")
    
    @pytest.fixture
    def planting_request(self, mock_location):
        """Fixture for planting decision request"""
        return PlantingDecisionRequest(
            location=mock_location,
            crop=CropType.MAIZE,
            planting_date=datetime.now(timezone.utc),
            field_size_acres=5.0
        )
    
    @pytest.fixture
    def harvesting_request(self, mock_location):
        """Fixture for harvesting decision request"""
        return HarvestingDecisionRequest(
            location=mock_location,
            crop=CropType.MAIZE,
            expected_harvest_date=datetime.now(timezone.utc),
            current_growth_stage="flowering"
        )
    
    def test_engine_initialization(self, engine):
        """Test decision engine initialization"""
        assert engine is not None
        assert engine.weather_client is not None
        assert engine.planting_model is not None
        assert engine.risk_model is not None
        assert engine.chain is not None
    
    def test_engine_chain_building(self, engine):
        """Test that engine builds handler chain correctly"""
        assert engine.chain is not None
        assert engine.planting_chain is not None
        assert engine.harvesting_chain is not None
    
    @pytest.mark.asyncio
    async def test_process_planting_decision(self, engine, planting_request, mock_weather_client):
        """Test processing planting decision"""
        # Mock weather data
        mock_weather = WeatherData(
            location=planting_request.location,
            temperature=25.5,
            humidity=65,
            wind_speed=12.3,
            condition=WeatherCondition.CLEAR,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Mock forecast data
        mock_forecast = [
            ForecastData(
                location=planting_request.location,
                date=datetime.now(timezone.utc),
                temperature_high=28,
                temperature_low=18,
                humidity=60,
                precipitation_chance=30,
                precipitation_mm=5.0,
                wind_speed=10,
                condition=WeatherCondition.CLEAR
            ) for _ in range(7)
        ]
        
        mock_weather_client.get_current_weather = AsyncMock(return_value=mock_weather)
        mock_weather_client.get_forecast = AsyncMock(return_value=mock_forecast)
        
        response = await engine.process_planting_decision(planting_request)
        
        assert response is not None
        assert isinstance(response, DecisionResponse)
        assert response.decision_type == "planting"
        assert response.crop == CropType.MAIZE
    
    @pytest.mark.asyncio
    async def test_process_harvesting_decision(self, engine, harvesting_request, mock_weather_client):
        """Test processing harvesting decision"""
        # Mock weather data
        mock_weather = WeatherData(
            location=harvesting_request.location,
            temperature=25.5,
            humidity=65,
            wind_speed=12.3,
            condition=WeatherCondition.CLEAR,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Mock forecast data
        mock_forecast = [
            ForecastData(
                location=harvesting_request.location,
                date=datetime.now(timezone.utc),
                temperature_high=28,
                temperature_low=18,
                humidity=60,
                precipitation_chance=30,
                precipitation_mm=5.0,
                wind_speed=10,
                condition=WeatherCondition.CLEAR
            ) for _ in range(7)
        ]
        
        mock_weather_client.get_current_weather = AsyncMock(return_value=mock_weather)
        mock_weather_client.get_forecast = AsyncMock(return_value=mock_forecast)
        
        response = await engine.process_harvesting_decision(harvesting_request)
        
        assert response is not None
        assert isinstance(response, DecisionResponse)
        assert response.decision_type == "harvesting"
        assert response.crop == CropType.MAIZE
    
    @pytest.mark.asyncio
    async def test_process_planting_decision_different_crops(self, engine, mock_location, mock_weather_client):
        """Test planting decision for different crops"""
        crops = [CropType.MAIZE, CropType.WHEAT, CropType.TEA]
        
        for crop in crops:
            request = PlantingDecisionRequest(
                location=mock_location,
                crop=crop,
                planting_date=datetime.now(timezone.utc)
            )
            
            # Mock weather data
            mock_weather = WeatherData(
                location=mock_location,
                temperature=25.5,
                humidity=65,
                wind_speed=12.3,
                condition=WeatherCondition.CLEAR,
                timestamp=datetime.now(timezone.utc)
            )
            
            mock_forecast = [
                ForecastData(
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
            ]
            
            mock_weather_client.get_current_weather = AsyncMock(return_value=mock_weather)
            mock_weather_client.get_forecast = AsyncMock(return_value=mock_forecast)
            
            response = await engine.process_planting_decision(request)
            
            assert response is not None
            assert response.crop == crop
