"""Decision Engine using Chain of Responsibility Pattern"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import logging

from ..ml.planting_model import PlantingPredictorModel
from ..ml.risk_model import RiskAssessmentModel
from ..ml.canopy_model import CanopyHealthModel
from .models import (
    DecisionRequest,
    PlantingDecisionRequest,
    HarvestingDecisionRequest,
    DecisionResponse,
    PlantingRecommendation,
    HarvestingRecommendation,
    RiskAssessment,
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
    Performs risk assessment using ML model.
    Third handler in the chain.
    """
    
    def __init__(self, risk_model: RiskAssessmentModel):
        super().__init__()
        self.risk_model = risk_model
    
    async def handle(self, request: DecisionRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk assessment"""
        logger.info("Performing risk assessment")
        
        forecast_data = context.get("forecast_data", [])
        
        if forecast_data:
            try:
                risk_assessment = self.risk_model.assess_risks(
                    forecast_data=forecast_data,
                    location=request.location
                )
                context["risk_assessment"] = risk_assessment
                logger.info(f"Risk assessment completed: {risk_assessment.overall_risk}")
            except Exception as e:
                logger.error(f"Failed to perform risk assessment: {e}")
                context["risk_assessment"] = None
        else:
            logger.warning("No forecast data available for risk assessment")
            context["risk_assessment"] = None
        
        return await self._pass_to_next(request, context)


class PlantingDecisionHandler(DecisionHandler):
    """
    Generates planting recommendations.
    Fourth handler in the chain (for planting decisions).
    """
    
    def __init__(self, planting_model: PlantingPredictorModel):
        super().__init__()
        self.planting_model = planting_model
    
    async def handle(self, request: DecisionRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate planting recommendation"""
        if not isinstance(request, PlantingDecisionRequest):
            return await self._pass_to_next(request, context)
        
        logger.info(f"Generating planting recommendation for {request.crop}")
        
        forecast_data = context.get("forecast_data", [])
        
        if forecast_data:
            try:
                recommendation = self.planting_model.evaluate_planting_suitability(
                    forecast_data=forecast_data,
                    crop=request.crop
                )
                context["recommendation"] = recommendation
                logger.info(f"Planting recommendation: {recommendation.recommended}")
            except Exception as e:
                logger.error(f"Failed to generate planting recommendation: {e}")
                context["recommendation"] = None
        else:
            logger.warning("No forecast data available for planting recommendation")
            context["recommendation"] = None
        
        return await self._pass_to_next(request, context)


class HarvestingDecisionHandler(DecisionHandler):
    """
    Generates harvesting recommendations.
    Fourth handler in the chain (for harvesting decisions).
    """
    
    async def handle(self, request: DecisionRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate harvesting recommendation"""
        if not isinstance(request, HarvestingDecisionRequest):
            return await self._pass_to_next(request, context)
        
        logger.info(f"Generating harvesting recommendation for {request.crop}")
        
        forecast_data = context.get("forecast_data", [])
        risk_assessment = context.get("risk_assessment")
        
        if forecast_data and risk_assessment:
            try:
                # Simple harvesting logic based on risk assessment
                recommended = risk_assessment.overall_risk.value in ["low", "medium"]
                confidence = 0.8 if recommended else 0.4
                
                recommendation = HarvestingRecommendation(
                    recommended=recommended,
                    confidence=confidence,
                    reason=f"Risk level is {risk_assessment.overall_risk.value}",
                    risk_factors=[rf.type for rf in risk_assessment.risk_factors],
                    weather_conditions=[f.condition.value for f in forecast_data[:3]]
                )
                context["recommendation"] = recommendation
                logger.info(f"Harvesting recommendation: {recommendation.recommended}")
            except Exception as e:
                logger.error(f"Failed to generate harvesting recommendation: {e}")
                context["recommendation"] = None
        else:
            logger.warning("Insufficient data for harvesting recommendation")
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
    
    def __init__(self, weather_client, planting_model: PlantingPredictorModel, risk_model: RiskAssessmentModel):
        """
        Initialize the decision engine with required components.
        
        Args:
            weather_client: WeatherAI API client
            planting_model: Planting prediction model
            risk_model: Risk assessment model
        """
        self.weather_client = weather_client
        self.planting_model = planting_model
        self.risk_model = risk_model
        
        # Build the chain of responsibility
        self._build_chain()
    
    def _build_chain(self):
        """Build the handler chain"""
        # Start with weather data fetcher
        self.chain = WeatherDataHandler(self.weather_client)
        
        # Add forecast data fetcher
        self.chain.set_next(ForecastDataHandler(self.weather_client))
        
        # Add risk assessment
        self.chain.set_next(RiskAssessmentHandler(self.risk_model))
        
        # Add decision-specific handlers
        planting_handler = PlantingDecisionHandler(self.planting_model)
        harvesting_handler = HarvestingDecisionHandler()
        
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
