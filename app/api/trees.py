"""Trees & Forestry API endpoints"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List, Optional
from datetime import datetime, timezone
import logging

from ..core.weather_client import get_weather_client
from ..core.models import TreeAnalysis
from ..ml.canopy_model import CanopyHealthModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/trees", tags=["trees"])

# Initialize canopy health model
canopy_model = CanopyHealthModel()


@router.post("/analyze", response_model=TreeAnalysis)
async def analyze_tree_canopy(
    image: UploadFile = File(...),
    farmer_id: str = Form(...),
    county: str = Form(...),
    land_acres: float = Form(...),
    notes: Optional[str] = Form(None),
    weather_client = Depends(get_weather_client)
):
    """
    Analyze tree canopy from uploaded image.
    
    Args:
        image: Image file of the farm/trees
        farmer_id: Farmer identifier
        county: County name
        land_acres: Land area in acres
        notes: Additional notes about the farm
    
    Returns:
        Tree analysis results
    """
    try:
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Call WeatherAI trees analyze API
            analysis_data = await weather_client.analyze_trees(
                image_path=temp_path,
                farmer_id=farmer_id,
                county=county,
                land_acres=land_acres,
                notes=notes
            )
            
            # Convert to TreeAnalysis model
            from datetime import datetime
            tree_analysis = TreeAnalysis(
                analysis_id=analysis_data.get("analysis_id", ""),
                timestamp=datetime.fromisoformat(analysis_data.get("timestamp", datetime.now(timezone.utc).isoformat())),
                farmer_id=analysis_data.get("farmer_id", farmer_id),
                county=analysis_data.get("county", county),
                land_acres=analysis_data.get("land_acres", land_acres),
                total_tree_count=analysis_data.get("total_tree_count", 0),
                tree_density_per_acre=analysis_data.get("tree_density_per_acre", 0),
                confidence_score=analysis_data.get("confidence_score", 0),
                canopy_coverage_pct=analysis_data.get("canopy_coverage_pct", 0),
                tree_health=analysis_data.get("tree_health", {"healthy": 0, "needs_care": 0, "needs_replacement": 0}),
                tree_species_guess=analysis_data.get("tree_species_guess"),
                observations=analysis_data.get("observations", []),
                recommendations=analysis_data.get("recommendations", [])
            )
            
            return tree_analysis
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Error analyzing tree canopy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-trend")
async def analyze_health_trend(
    farmer_id: str,
    benchmark_density: float = 30.0
):
    """
    Analyze tree health trend over time.
    
    Args:
        farmer_id: Farmer identifier
        benchmark_density: Benchmark tree density per acre
    
    Returns:
        Health trend analysis
    """
    try:
        # This would typically fetch historical data from repository
        # For now, return a mock response
        return {
            "trend": "insufficient_data",
            "health_change": 0,
            "coverage_change": 0,
            "recommendation": "Need more historical data points for trend analysis"
        }
    except Exception as e:
        logger.error(f"Error analyzing health trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-score")
async def calculate_health_score(
    total_tree_count: int,
    healthy: int,
    needs_care: int,
    needs_replacement: int,
    confidence_score: float,
    canopy_coverage_pct: float
):
    """
    Calculate overall health score from tree metrics.
    
    Args:
        total_tree_count: Total number of trees
        healthy: Number of healthy trees
        needs_care: Number of trees needing care
        needs_replacement: Number of trees needing replacement
        confidence_score: Analysis confidence score
        canopy_coverage_pct: Canopy coverage percentage
    
    Returns:
        Health score (0-100)
    """
    try:
        from ..core.models import TreeAnalysis, TreeHealth
        
        # Create mock analysis for calculation
        tree_health = TreeHealth(
            healthy=healthy,
            needs_care=needs_care,
            needs_replacement=needs_replacement
        )
        
        mock_analysis = TreeAnalysis(
            analysis_id="mock",
            timestamp=datetime.now(timezone.utc),
            farmer_id="mock",
            county="mock",
            land_acres=1.0,
            total_tree_count=total_tree_count,
            tree_density_per_acre=total_tree_count,
            confidence_score=confidence_score,
            canopy_coverage_pct=canopy_coverage_pct,
            tree_health=tree_health
        )
        
        score = canopy_model.calculate_health_score(mock_analysis)
        return {"health_score": score}
        
    except Exception as e:
        logger.error(f"Error calculating health score: {e}")
        raise HTTPException(status_code=500, detail=str(e))
