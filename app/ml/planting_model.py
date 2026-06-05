"""Planting Window Predictor Model using Strategy Pattern"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List
import logging

from .base import MLModelStrategy
from ..core.models import ForecastData, PlantingRecommendation, CropType

logger = logging.getLogger(__name__)


class PlantingPredictorModel(MLModelStrategy):
    """
    Predicts optimal planting windows based on weather patterns.
    Uses Random Forest for classification of planting suitability.
    """
    
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        self.scaler = StandardScaler()
        self.crop_requirements = self._load_crop_requirements()
    
    def _load_crop_requirements(self) -> Dict[CropType, Dict[str, Any]]:
        """
        Load crop-specific planting requirements.
        These are generic baseline values that can be customized per region.
        
        Returns:
            Dictionary mapping crop types to their requirements
        """
        # Generic baseline requirements - can be overridden via configuration
        return {
            CropType.MAIZE: {
                "min_temp": 15,
                "max_temp": 35,
                "optimal_temp": 25,
                "min_soil_moisture": 40,
                "frost_sensitive": True,
                "drought_sensitive": True
            },
            CropType.WHEAT: {
                "min_temp": 10,
                "max_temp": 30,
                "optimal_temp": 20,
                "min_soil_moisture": 35,
                "frost_sensitive": False,
                "drought_sensitive": True
            },
            CropType.TEA: {
                "min_temp": 12,
                "max_temp": 28,
                "optimal_temp": 22,
                "min_soil_moisture": 60,
                "frost_sensitive": False,
                "drought_sensitive": True
            },
            CropType.RICE: {
                "min_temp": 20,
                "max_temp": 35,
                "optimal_temp": 28,
                "min_soil_moisture": 70,
                "frost_sensitive": True,
                "drought_sensitive": True
            },
            CropType.COFFEE: {
                "min_temp": 15,
                "max_temp": 28,
                "optimal_temp": 22,
                "min_soil_moisture": 55,
                "frost_sensitive": True,
                "drought_sensitive": True
            },
            CropType.SORGHUM: {
                "min_temp": 18,
                "max_temp": 38,
                "optimal_temp": 30,
                "min_soil_moisture": 30,
                "frost_sensitive": False,
                "drought_sensitive": False
            },
            CropType.MILLET: {
                "min_temp": 20,
                "max_temp": 35,
                "optimal_temp": 28,
                "min_soil_moisture": 35,
                "frost_sensitive": False,
                "drought_sensitive": False
            }
        }
    
    def update_crop_requirements(self, crop: CropType, requirements: Dict[str, Any]) -> None:
        """
        Update crop-specific requirements dynamically.
        Allows for region-specific customization.
        
        Args:
            crop: Crop type to update
            requirements: New requirements dictionary
        """
        self.crop_requirements[crop] = requirements
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the Random Forest classifier.
        
        Args:
            X: Feature matrix
            y: Target variable (1 = suitable for planting, 0 = not suitable)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info("Planting predictor model trained successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict planting suitability.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions (1 = suitable, 0 = not suitable)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        feature_names = [
            "temp_high", "temp_low", "humidity", "precip_chance",
            "precip_mm", "wind_speed", "day_of_year"
        ]
        
        importances = dict(zip(feature_names, self.model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    def evaluate_planting_suitability(
        self,
        forecast_data: List[ForecastData],
        crop: CropType
    ) -> PlantingRecommendation:
        """
        Evaluate if conditions are suitable for planting a specific crop.
        
        Args:
            forecast_data: Weather forecast data
            crop: Crop type to evaluate
        
        Returns:
            Planting recommendation
        """
        if not forecast_data:
            return PlantingRecommendation(
                recommended=False,
                confidence=0.0,
                reason="No forecast data available"
            )
        
        # Get crop requirements
        req = self.crop_requirements.get(crop, self.crop_requirements[CropType.MAIZE])
        
        # Analyze forecast for the next 7 days
        suitable_days = 0
        risk_factors = []
        weather_conditions = []
        
        for forecast in forecast_data[:7]:
            avg_temp = (forecast.temperature_high + forecast.temperature_low) / 2
            
            # Check temperature suitability
            if req["min_temp"] <= avg_temp <= req["max_temp"]:
                suitable_days += 1
            elif avg_temp < req["min_temp"]:
                risk_factors.append(f"Low temperature on {forecast.date.strftime('%Y-%m-%d')}")
            else:
                risk_factors.append(f"High temperature on {forecast.date.strftime('%Y-%m-%d')}")
            
            # Check frost risk
            if req["frost_sensitive"] and forecast.temperature_low < 5:
                risk_factors.append(f"Frost risk on {forecast.date.strftime('%Y-%m-%d')}")
            
            # Check drought risk
            if req["drought_sensitive"] and forecast.precipitation_mm < 5:
                risk_factors.append(f"Drought risk on {forecast.date.strftime('%Y-%m-%d')}")
            
            weather_conditions.append(f"{forecast.condition.value} on {forecast.date.strftime('%Y-%m-%d')}")
        
        # Calculate confidence based on suitable days
        confidence = suitable_days / min(len(forecast_data[:7]), 7)
        
        # Determine recommendation
        recommended = confidence >= 0.6 and len(risk_factors) <= 2
        
        # Generate reason
        if recommended:
            reason = f"Conditions are favorable for {crop.value} planting. {suitable_days} out of 7 days meet requirements."
        else:
            reason = f"Conditions are not optimal for {crop.value} planting due to: {', '.join(risk_factors[:3])}"
        
        # Find optimal date (first suitable day)
        optimal_date = None
        if recommended:
            for forecast in forecast_data[:7]:
                avg_temp = (forecast.temperature_high + forecast.temperature_low) / 2
                if req["min_temp"] <= avg_temp <= req["max_temp"]:
                    optimal_date = forecast.date
                    break
        
        return PlantingRecommendation(
            recommended=recommended,
            confidence=confidence,
            optimal_date=optimal_date,
            reason=reason,
            risk_factors=risk_factors,
            weather_conditions=weather_conditions
        )
