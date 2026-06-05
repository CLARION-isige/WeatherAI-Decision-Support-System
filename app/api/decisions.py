"""Decision Support API endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from ..core.weather_client import get_weather_client
from ..core.decision_engine import DecisionEngine
from ..core.models import (
    PlantingDecisionRequest,
    HarvestingDecisionRequest,
    DecisionResponse
)
from ..ml.planting_model import PlantingPredictorModel
from ..ml.risk_model import RiskAssessmentModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/decisions", tags=["decisions"])

# Initialize ML models
planting_model = PlantingPredictorModel()
risk_model = RiskAssessmentModel()

# Decision engine will be initialized on first request
_decision_engine: DecisionEngine = None


def get_decision_engine(weather_client) -> DecisionEngine:
    """Get or create decision engine instance"""
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = DecisionEngine(
            weather_client=weather_client,
            planting_model=planting_model,
            risk_model=risk_model
        )
    return _decision_engine


@router.post("/planting", response_model=DecisionResponse)
async def get_planting_recommendation(
    request: PlantingDecisionRequest,
    weather_client = Depends(get_weather_client)
):
    """
    Get AI-powered planting recommendations.
    
    Args:
        request: Planting decision request with location, crop, and dates
    
    Returns:
        Decision response with planting recommendation
    """
    try:
        engine = get_decision_engine(weather_client)
        response = await engine.process_planting_decision(request)
        
        return response
    except Exception as e:
        logger.error(f"Error processing planting decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/harvesting", response_model=DecisionResponse)
async def get_harvesting_recommendation(
    request: HarvestingDecisionRequest,
    weather_client = Depends(get_weather_client)
):
    """
    Get AI-powered harvesting recommendations.
    
    Args:
        request: Harvesting decision request with location, crop, and dates
    
    Returns:
        Decision response with harvesting recommendation
    """
    try:
        engine = get_decision_engine(weather_client)
        response = await engine.process_harvesting_decision(request)
        
        return response
    except Exception as e:
        logger.error(f"Error processing harvesting decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk-assessment")
async def get_risk_assessment(
    request: PlantingDecisionRequest,
    weather_client = Depends(get_weather_client)
):
    """
    Get risk assessment for agricultural activities.
    
    Args:
        request: Decision request with location and crop
    
    Returns:
        Risk assessment results
    """
    try:
        # Get forecast data
        forecast_data = await weather_client.get_forecast(
            lat=request.location.lat,
            lon=request.location.lon,
            days=7
        )
        
        # Perform risk assessment
        risk_assessment = risk_model.assess_risks(
            forecast_data=forecast_data,
            location=request.location
        )
        
        return risk_assessment
    except Exception as e:
        logger.error(f"Error performing risk assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


