"""Risk Assessment Model using Strategy Pattern"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, List
from datetime import datetime, timezone
import logging

from .base import MLModelStrategy
from ..core.models import ForecastData, RiskAssessment, RiskFactor, RiskLevel, Location

logger = logging.getLogger(__name__)


class RiskAssessmentModel(MLModelStrategy):
    """
    Assesses agricultural risks using anomaly detection.
    Uses Isolation Forest for detecting unusual weather patterns.
    """
    
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        self.risk_thresholds = self._load_risk_thresholds()
    
    def _load_risk_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """
        Load risk threshold definitions.
        These are generic baseline values that can be customized per region.
        
        Returns:
            Dictionary mapping risk types to thresholds
        """
        # Generic baseline thresholds - can be overridden via configuration
        return {
            "frost": {
                "critical": -2,
                "high": 0,
                "medium": 5,
                "low": 10
            },
            "drought": {
                "critical": 0,
                "high": 5,
                "medium": 15,
                "low": 30
            },
            "extreme_wind": {
                "critical": 50,
                "high": 40,
                "medium": 30,
                "low": 20
            },
            "heat_stress": {
                "critical": 40,
                "high": 35,
                "medium": 32,
                "low": 30
            },
            "heavy_rain": {
                "critical": 100,
                "high": 50,
                "medium": 30,
                "low": 20
            }
        }
    
    def update_risk_thresholds(self, risk_type: str, thresholds: Dict[str, Any]) -> None:
        """
        Update risk thresholds dynamically.
        Allows for region-specific customization.
        
        Args:
            risk_type: Type of risk (frost, drought, etc.)
            thresholds: New threshold values
        """
        self.risk_thresholds[risk_type] = thresholds
    
    def train(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Train the Isolation Forest for anomaly detection.
        
        Args:
            X: Feature matrix (weather data)
            y: Not used for unsupervised learning
        """
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X)
        self.is_trained = True
        logger.info("Risk assessment model trained successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Detect anomalies in weather data.
        
        Args:
            X: Feature matrix
        
        Returns:
            Anomaly predictions (-1 = anomaly, 1 = normal)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Isolation Forest doesn't provide feature importance.
        
        Returns:
            Empty dictionary
        """
        return {}
    
    def assess_risks(
        self,
        forecast_data: List[ForecastData],
        location: Location
    ) -> RiskAssessment:
        """
        Assess multiple risk factors from forecast data.
        
        Args:
            forecast_data: Weather forecast data
            location: Geographic location
        
        Returns:
            Complete risk assessment
        """
        if not forecast_data:
            return RiskAssessment(
                overall_risk=RiskLevel.LOW,
                risk_factors=[],
                timestamp=datetime.now(timezone.utc),
                location=location
            )
        
        risk_factors = []
        
        # Analyze each risk type
        for forecast in forecast_data[:7]:
            # Frost risk
            frost_risk = self._assess_frost_risk(forecast)
            if frost_risk["level"] != RiskLevel.LOW:
                risk_factors.append(RiskFactor(
                    type="frost",
                    level=frost_risk["level"],
                    probability=frost_risk["probability"],
                    description=frost_risk["description"],
                    mitigation="Consider frost protection measures or delay planting"
                ))
            
            # Drought risk
            drought_risk = self._assess_drought_risk(forecast)
            if drought_risk["level"] != RiskLevel.LOW:
                risk_factors.append(RiskFactor(
                    type="drought",
                    level=drought_risk["level"],
                    probability=drought_risk["probability"],
                    description=drought_risk["description"],
                    mitigation="Ensure irrigation systems are ready, consider drought-resistant varieties"
                ))
            
            # Extreme wind risk
            wind_risk = self._assess_wind_risk(forecast)
            if wind_risk["level"] != RiskLevel.LOW:
                risk_factors.append(RiskFactor(
                    type="extreme_wind",
                    level=wind_risk["level"],
                    probability=wind_risk["probability"],
                    description=wind_risk["description"],
                    mitigation="Secure loose items, consider windbreaks for young plants"
                ))
            
            # Heat stress risk
            heat_risk = self._assess_heat_risk(forecast)
            if heat_risk["level"] != RiskLevel.LOW:
                risk_factors.append(RiskFactor(
                    type="heat_stress",
                    level=heat_risk["level"],
                    probability=heat_risk["probability"],
                    description=heat_risk["description"],
                    mitigation="Increase irrigation, provide shade if possible"
                ))
            
            # Heavy rain risk
            rain_risk = self._assess_rain_risk(forecast)
            if rain_risk["level"] != RiskLevel.LOW:
                risk_factors.append(RiskFactor(
                    type="heavy_rain",
                    level=rain_risk["level"],
                    probability=rain_risk["probability"],
                    description=rain_risk["description"],
                    mitigation="Ensure proper drainage, avoid planting in low-lying areas"
                ))
        
        # Calculate overall risk
        overall_risk = self._calculate_overall_risk(risk_factors)
        
        return RiskAssessment(
            overall_risk=overall_risk,
            risk_factors=risk_factors,
            timestamp=datetime.now(timezone.utc),
            location=location
        )
    
    def _assess_frost_risk(self, forecast: ForecastData) -> Dict[str, Any]:
        """Assess frost risk from forecast data"""
        thresholds = self.risk_thresholds["frost"]
        temp = forecast.temperature_low
        
        if temp <= thresholds["critical"]:
            return {
                "level": RiskLevel.CRITICAL,
                "probability": 0.9,
                "description": f"Critical frost risk: Temperature {temp}°C on {forecast.date.strftime('%Y-%m-%d')}"
            }
        elif temp <= thresholds["high"]:
            return {
                "level": RiskLevel.HIGH,
                "probability": 0.7,
                "description": f"High frost risk: Temperature {temp}°C on {forecast.date.strftime('%Y-%m-%d')}"
            }
        elif temp <= thresholds["medium"]:
            return {
                "level": RiskLevel.MEDIUM,
                "probability": 0.5,
                "description": f"Moderate frost risk: Temperature {temp}°C on {forecast.date.strftime('%Y-%m-%d')}"
            }
        else:
            return {
                "level": RiskLevel.LOW,
                "probability": 0.1,
                "description": "Low frost risk"
            }
    
    def _assess_drought_risk(self, forecast: ForecastData) -> Dict[str, Any]:
        """Assess drought risk from forecast data"""
        thresholds = self.risk_thresholds["drought"]
        precip = forecast.precipitation_mm
        humidity = forecast.humidity
        
        # Combined drought indicator
        drought_score = (100 - humidity) / 2 + (10 - precip) / 2
        drought_score = max(0, min(100, drought_score))
        
        if drought_score >= thresholds["critical"]:
            return {
                "level": RiskLevel.CRITICAL,
                "probability": 0.85,
                "description": f"Critical drought risk: Low precipitation ({precip}mm) and humidity ({humidity}%)"
            }
        elif drought_score >= thresholds["high"]:
            return {
                "level": RiskLevel.HIGH,
                "probability": 0.65,
                "description": f"High drought risk: Low precipitation ({precip}mm) and humidity ({humidity}%)"
            }
        elif drought_score >= thresholds["medium"]:
            return {
                "level": RiskLevel.MEDIUM,
                "probability": 0.45,
                "description": f"Moderate drought risk: Precipitation {precip}mm, humidity {humidity}%"
            }
        else:
            return {
                "level": RiskLevel.LOW,
                "probability": 0.15,
                "description": "Low drought risk"
            }
    
    def _assess_wind_risk(self, forecast: ForecastData) -> Dict[str, Any]:
        """Assess extreme wind risk from forecast data"""
        thresholds = self.risk_thresholds["extreme_wind"]
        wind = forecast.wind_speed
        
        if wind >= thresholds["critical"]:
            return {
                "level": RiskLevel.CRITICAL,
                "probability": 0.8,
                "description": f"Critical wind risk: {wind} km/h on {forecast.date.strftime('%Y-%m-%d')}"
            }
        elif wind >= thresholds["high"]:
            return {
                "level": RiskLevel.HIGH,
                "probability": 0.6,
                "description": f"High wind risk: {wind} km/h on {forecast.date.strftime('%Y-%m-%d')}"
            }
        elif wind >= thresholds["medium"]:
            return {
                "level": RiskLevel.MEDIUM,
                "probability": 0.4,
                "description": f"Moderate wind risk: {wind} km/h on {forecast.date.strftime('%Y-%m-%d')}"
            }
        else:
            return {
                "level": RiskLevel.LOW,
                "probability": 0.1,
                "description": "Low wind risk"
            }
    
    def _assess_heat_risk(self, forecast: ForecastData) -> Dict[str, Any]:
        """Assess heat stress risk from forecast data"""
        thresholds = self.risk_thresholds["heat_stress"]
        temp = forecast.temperature_high
        
        if temp >= thresholds["critical"]:
            return {
                "level": RiskLevel.CRITICAL,
                "probability": 0.85,
                "description": f"Critical heat stress: {temp}°C on {forecast.date.strftime('%Y-%m-%d')}"
            }
        elif temp >= thresholds["high"]:
            return {
                "level": RiskLevel.HIGH,
                "probability": 0.65,
                "description": f"High heat stress: {temp}°C on {forecast.date.strftime('%Y-%m-%d')}"
            }
        elif temp >= thresholds["medium"]:
            return {
                "level": RiskLevel.MEDIUM,
                "probability": 0.45,
                "description": f"Moderate heat stress: {temp}°C on {forecast.date.strftime('%Y-%m-%d')}"
            }
        else:
            return {
                "level": RiskLevel.LOW,
                "probability": 0.1,
                "description": "Low heat stress risk"
            }
    
    def _assess_rain_risk(self, forecast: ForecastData) -> Dict[str, Any]:
        """Assess heavy rain risk from forecast data"""
        thresholds = self.risk_thresholds["heavy_rain"]
        precip = forecast.precipitation_mm
        precip_chance = forecast.precipitation_chance
        
        if precip >= thresholds["critical"] and precip_chance > 80:
            return {
                "level": RiskLevel.CRITICAL,
                "probability": 0.9,
                "description": f"Critical heavy rain risk: {precip}mm expected on {forecast.date.strftime('%Y-%m-%d')}"
            }
        elif precip >= thresholds["high"] and precip_chance > 60:
            return {
                "level": RiskLevel.HIGH,
                "probability": 0.7,
                "description": f"High heavy rain risk: {precip}mm expected on {forecast.date.strftime('%Y-%m-%d')}"
            }
        elif precip >= thresholds["medium"] and precip_chance > 40:
            return {
                "level": RiskLevel.MEDIUM,
                "probability": 0.5,
                "description": f"Moderate rain risk: {precip}mm expected on {forecast.date.strftime('%Y-%m-%d')}"
            }
        else:
            return {
                "level": RiskLevel.LOW,
                "probability": 0.1,
                "description": "Low heavy rain risk"
            }
    
    def _calculate_overall_risk(self, risk_factors: List[RiskFactor]) -> RiskLevel:
        """
        Calculate overall risk level from individual risk factors.
        
        Args:
            risk_factors: List of individual risk factors
        
        Returns:
            Overall risk level
        """
        if not risk_factors:
            return RiskLevel.LOW
        
        # Count risks by level
        critical_count = sum(1 for r in risk_factors if r.level == RiskLevel.CRITICAL)
        high_count = sum(1 for r in risk_factors if r.level == RiskLevel.HIGH)
        medium_count = sum(1 for r in risk_factors if r.level == RiskLevel.MEDIUM)
        
        # Determine overall risk
        if critical_count > 0:
            return RiskLevel.CRITICAL
        elif high_count >= 2:
            return RiskLevel.CRITICAL
        elif high_count == 1:
            return RiskLevel.HIGH
        elif medium_count >= 3:
            return RiskLevel.HIGH
        elif medium_count >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
