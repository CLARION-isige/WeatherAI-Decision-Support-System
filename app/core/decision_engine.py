"""Decision Engine using Chain of Responsibility Pattern"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import logging

from .models import (
    DecisionRequest,
    PlantingDecisionRequest,
    HarvestingDecisionRequest,
    DecisionResponse,
    ForecastData,
    WeatherData,
    Location,
    CropType
)

logger = logging.getLogger(__name__)


class DecisionHandler(ABC):
    """
    Abstract base class for decision handlers in Chain of Responsibility.
    Each handler can process a decision request and pass it to the next handler.
    """
    
    def __init__(self):
        self._next_handler: Optional['DecisionHandler'] = None
    
    def set_next(self, handler: 'DecisionHandler') -> 'DecisionHandler':
        """
        Set the next handler in the chain.
        
        Args:
            handler: Next handler to process the request
        
        Returns:
            The handler that was set (for chaining)
        """
        self._next_handler = handler
        return handler
    
    @abstractmethod
    async def handle(self, request: DecisionRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle the decision request.
        
        Args:
            request: Decision request
            context: Shared context between handlers
        
        Returns:
            Updated context with handler's results
        """
        pass
    
    async def _pass_to_next(self, request: DecisionRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pass request to next handler in chain"""
        if self._next_handler:
            return await self._next_handler.handle(request, context)
        return context


class WeatherDataHandler(DecisionHandler):
    """
    Fetches weather data for the location.
    First handler in the chain.
    """
    
    def __init__(self, weather_client):
        super().__init__()
        self.weather_client = weather_client
    
    async def handle(self, request: DecisionRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch current weather data"""
        logger.info(f"Fetching weather data for {request.location.lat}, {request.location.lon}")
        
        try:
            weather_data = await self.weather_client.get_current_weather(
                lat=request.location.lat,
                lon=request.location.lon
            )
            context["weather_data"] = weather_data
            logger.info("Weather data fetched successfully")
        except Exception as e:
            logger.error(f"Failed to fetch weather data: {e}")
            context["weather_data"] = None
        
        return await self._pass_to_next(request, context)


class ForecastDataHandler(DecisionHandler):
    """
    Fetches forecast data for the location.
    Second handler in the chain.
    """
    
    def __init__(self, weather_client):
        super().__init__()
        self.weather_client = weather_client
    
    async def handle(self, request: DecisionRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch forecast data"""
        logger.info(f"Fetching forecast data for {request.location.lat}, {request.location.lon}")
        
        try:
            forecast_data = await self.weather_client.get_forecast(
                lat=request.location.lat,
                lon=request.location.lon,
                days=7
            )
            context["forecast_data"] = forecast_data
            logger.info(f"Forecast data fetched successfully ({len(forecast_data)} days)")
        except Exception as e:
            logger.error(f"Failed to fetch forecast data: {e}")
            context["forecast_data"] = []
        
        return await self._pass_to_next(request, context)


class RiskAssessmentHandler(DecisionHandler):
    """
    Performs risk assessment using WeatherAI API.
    Third handler in the chain.
    """
    
    def __init__(self, weather_client):
        super().__init__()
        self.weather_client = weather_client
    
    async def handle(self, request: DecisionRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk assessment using WeatherAI API"""
        logger.info("Performing risk assessment via WeatherAI API")
        
        try:
            risk_assessment_data = await self.weather_client.get_risk_assessment(
                lat=request.location.lat,
                lon=request.location.lon,
                days=7
            )
            context["risk_assessment"] = risk_assessment_data
            logger.info(f"Risk assessment completed via WeatherAI API")
        except Exception as e:
            logger.error(f"Failed to perform risk assessment via WeatherAI API: {e}")
            context["risk_assessment"] = None
        
        return await self._pass_to_next(request, context)


class PlantingDecisionHandler(DecisionHandler):
    """
    Generates planting recommendations using WeatherAI API.
    Fourth handler in the chain (for planting decisions).
    """
    
    def __init__(self, weather_client):
        super().__init__()
        self.weather_client = weather_client
    
    async def handle(self, request: DecisionRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate planting recommendation using WeatherAI API"""
        if not isinstance(request, PlantingDecisionRequest):
            return await self._pass_to_next(request, context)
        
        logger.info(f"Generating planting recommendation for {request.crop} via WeatherAI API")
        
        try:
            recommendation_data = await self.weather_client.get_planting_recommendation(
                lat=request.location.lat,
                lon=request.location.lon,
                crop=request.crop.value,
                days=7
            )
            context["recommendation"] = recommendation_data
            logger.info(f"Planting recommendation completed via WeatherAI API")
        except Exception as e:
            logger.error(f"Failed to generate planting recommendation via WeatherAI API: {e}")
            context["recommendation"] = None
        
        return await self._pass_to_next(request, context)


class HarvestingDecisionHandler(DecisionHandler):
    """
    Generates harvesting recommendations using WeatherAI API.
    Fourth handler in the chain (for harvesting decisions).
    """
    
    def __init__(self, weather_client):
        super().__init__()
        self.weather_client = weather_client
    
    async def handle(self, request: DecisionRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate harvesting recommendation using WeatherAI API"""
        if not isinstance(request, HarvestingDecisionRequest):
            return await self._pass_to_next(request, context)
        
        logger.info(f"Generating harvesting recommendation for {request.crop} via WeatherAI API")
        
        try:
            recommendation_data = await self.weather_client.get_harvesting_recommendation(
                lat=request.location.lat,
                lon=request.location.lon,
                crop=request.crop.value,
                days=7
            )
            context["recommendation"] = recommendation_data
            logger.info(f"Harvesting recommendation completed via WeatherAI API")
        except Exception as e:
            logger.error(f"Failed to generate harvesting recommendation via WeatherAI API: {e}")
            context["recommendation"] = None
        
        return await self._pass_to_next(request, context)


class ResponseBuilderHandler(DecisionHandler):
    """
    Builds the final response object.
    Last handler in the chain.
    """
    
    async def handle(self, request: DecisionRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build final response"""
        logger.info("Building final response")
        
        decision_type = "planting" if isinstance(request, PlantingDecisionRequest) else "harvesting"
        
        response = DecisionResponse(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            location=request.location,
            crop=request.crop,
            decision_type=decision_type,
            recommendation=context.get("recommendation").dict() if context.get("recommendation") else {},
            weather_data=context.get("weather_data"),
            forecast_data=context.get("forecast_data"),
            risk_assessment=context.get("risk_assessment")
        )
        
        context["response"] = response
        return context


class DecisionEngine:
    """
    Main decision engine that orchestrates the Chain of Responsibility.
    """
    
    def __init__(self, weather_client):
        """
        Initialize the decision engine with required components.
        
        Args:
            weather_client: WeatherAI API client
        """
        self.weather_client = weather_client
        
        # Build the chain of responsibility
        self._build_chain()
    
    def _build_chain(self):
        """Build the handler chain"""
        # Start with weather data fetcher
        self.chain = WeatherDataHandler(self.weather_client)
        
        # Add forecast data fetcher
        self.chain.set_next(ForecastDataHandler(self.weather_client))
        
        # Add risk assessment using WeatherAI API
        self.chain.set_next(RiskAssessmentHandler(self.weather_client))
        
        # Add decision-specific handlers using WeatherAI API
        planting_handler = PlantingDecisionHandler(self.weather_client)
        harvesting_handler = HarvestingDecisionHandler(self.weather_client)
        
        # Both decision handlers point to response builder
        planting_handler.set_next(ResponseBuilderHandler())
        harvesting_handler.set_next(ResponseBuilderHandler())
        
        # Risk assessment passes to both decision handlers
        # We'll handle this dynamically in the process method
        self.planting_chain = planting_handler
        self.harvesting_chain = harvesting_handler
    
    async def process_planting_decision(self, request: PlantingDecisionRequest) -> DecisionResponse:
        """
        Process a planting decision request.
        
        Args:
            request: Planting decision request
        
        Returns:
            Decision response with recommendations
        """
        logger.info(f"Processing planting decision for {request.crop}")
        
        # Build context
        context = {}
        
        # Run through common handlers (weather, forecast, risk)
        context = await self.chain.handle(request, context)
        
        # Run through planting-specific handler
        context = await self.planting_chain.handle(request, context)
        
        return context.get("response")
    
    async def process_harvesting_decision(self, request: HarvestingDecisionRequest) -> DecisionResponse:
        """
        Process a harvesting decision request.
        
        Args:
            request: Harvesting decision request
        
        Returns:
            Decision response with recommendations
        """
        logger.info(f"Processing harvesting decision for {request.crop}")
        
        # Build context
        context = {}
        
        # Run through common handlers (weather, forecast, risk)
        context = await self.chain.handle(request, context)
        
        # Run through harvesting-specific handler
        context = await self.harvesting_chain.handle(request, context)
        
        return context.get("response")
