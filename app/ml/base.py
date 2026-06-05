"""Base ML Model Strategy Pattern"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path

from ..core.models import ForecastData, RiskAssessment, RiskLevel

logger = logging.getLogger(__name__)


class MLModelStrategy(ABC):
    """
    Abstract base class for ML models using Strategy Pattern.
    Allows different ML algorithms to be used interchangeably.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the ML model.
        
        Args:
            model_path: Path to load/save trained model
        """
        self.model = None
        self.model_path = model_path
        self.is_trained = False
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model on given data.
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    def save_model(self, path: str = None) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save model (uses model_path if not provided)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        save_path = path or self.model_path
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, save_path)
            logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str = None) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load model from (uses model_path if not provided)
        """
        load_path = path or self.model_path
        if load_path and Path(load_path).exists():
            self.model = joblib.load(load_path)
            self.is_trained = True
            logger.info(f"Model loaded from {load_path}")
        else:
            logger.warning(f"No model found at {load_path}")
    
    def extract_features(self, forecast_data: List[ForecastData]) -> pd.DataFrame:
        """
        Extract features from forecast data for ML prediction.
        
        Args:
            forecast_data: List of forecast data points
        
        Returns:
            Feature matrix as DataFrame
        """
        features = []
        
        for forecast in forecast_data:
            feature_dict = {
                "temp_high": forecast.temperature_high,
                "temp_low": forecast.temperature_low,
                "humidity": forecast.humidity,
                "precip_chance": forecast.precipitation_chance,
                "precip_mm": forecast.precipitation_mm,
                "wind_speed": forecast.wind_speed,
                "day_of_year": forecast.date.timetuple().tm_yday
            }
            features.append(feature_dict)
        
        return pd.DataFrame(features)
