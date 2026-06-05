"""ML model unit tests"""

import pytest
import pandas as pd
import numpy as np
from app.ml.planting_model import PlantingPredictorModel
from app.ml.risk_model import RiskAssessmentModel
from app.ml.canopy_model import CanopyHealthModel
from app.ml.base import MLModelStrategy
from app.core.models import CropType, ForecastData, Location, WeatherCondition, TreeAnalysis, TreeHealth
from datetime import datetime, timezone


class TestPlantingPredictorModel:
    """Test suite for PlantingPredictorModel"""
    
    @pytest.fixture
    def model(self):
        """Fixture for planting predictor model"""
        return PlantingPredictorModel()
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert model is not None
        assert isinstance(model, MLModelStrategy)
        assert model.crop_requirements is not None
        assert CropType.MAIZE in model.crop_requirements
    
    def test_crop_requirements_loaded(self, model):
        """Test that crop requirements are loaded correctly"""
        maize_req = model.crop_requirements[CropType.MAIZE]
        assert "min_temp" in maize_req
        assert "max_temp" in maize_req
        assert "optimal_temp" in maize_req
        assert "min_soil_moisture" in maize_req
        assert "frost_sensitive" in maize_req
        assert "drought_sensitive" in maize_req
    
    def test_evaluate_planting_suitability_with_good_conditions(self, model, mock_forecast_data):
        """Test planting suitability evaluation with good conditions"""
        # Create favorable forecast data
        favorable_forecast = [
            ForecastData(
                location=Location(lat=-1.2921, lon=36.8219),
                date=datetime.now(timezone.utc),
                temperature_high=25,
                temperature_low=18,
                humidity=60,
                precipitation_chance=30,
                precipitation_mm=10,
                wind_speed=10,
                condition=WeatherCondition.CLEAR
            ) for _ in range(7)
        ]
        
        recommendation = model.evaluate_planting_suitability(favorable_forecast, CropType.MAIZE)
        
        assert recommendation is not None
        assert hasattr(recommendation, 'recommended')
        assert hasattr(recommendation, 'confidence')
        assert 0 <= recommendation.confidence <= 1
        assert isinstance(recommendation.reason, str)
    
    def test_evaluate_planting_suitability_with_frost_risk(self, model):
        """Test planting suitability evaluation with frost risk"""
        frost_forecast = [
            ForecastData(
                location=Location(lat=-1.2921, lon=36.8219),
                date=datetime.now(timezone.utc),
                temperature_high=15,
                temperature_low=2,  # Frost risk
                humidity=60,
                precipitation_chance=30,
                precipitation_mm=5,
                wind_speed=10,
                condition=WeatherCondition.CLEAR
            )
        ]
        
        recommendation = model.evaluate_planting_suitability(frost_forecast, CropType.MAIZE)
        
        assert recommendation is not None
        assert len(recommendation.risk_factors) > 0
        assert any("frost" in factor.lower() for factor in recommendation.risk_factors)
    
    def test_evaluate_planting_suitability_empty_forecast(self, model):
        """Test planting suitability with empty forecast data"""
        recommendation = model.evaluate_planting_suitability([], CropType.MAIZE)
        
        assert recommendation is not None
        assert recommendation.recommended == False
        assert recommendation.confidence == 0.0
    
    def test_evaluate_planting_suitability_different_crops(self, model, mock_forecast_data):
        """Test planting suitability for different crop types"""
        crops = [CropType.MAIZE, CropType.WHEAT, CropType.TEA, CropType.RICE]
        
        for crop in crops:
            recommendation = model.evaluate_planting_suitability(mock_forecast_data, crop)
            assert recommendation is not None
            assert 0 <= recommendation.confidence <= 1
    
    def test_train_model(self, model):
        """Test model training"""
        # Create dummy training data
        X = pd.DataFrame({
            "temp_high": [25, 26, 24, 27, 25],
            "temp_low": [18, 19, 17, 20, 18],
            "humidity": [60, 65, 55, 70, 60],
            "precip_chance": [30, 40, 20, 50, 30],
            "precip_mm": [5, 8, 3, 10, 5],
            "wind_speed": [10, 12, 8, 15, 10],
            "day_of_year": [165, 166, 167, 168, 169]
        })
        y = pd.Series([1, 1, 0, 1, 0])
        
        model.train(X, y)
        assert model.is_trained == True
        assert model.model is not None
    
    def test_predict_without_training(self, model):
        """Test prediction without training raises error"""
        X = pd.DataFrame({"temp_high": [25]})
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(X)
    
    def test_get_feature_importance(self, model):
        """Test getting feature importance"""
        # Train model first
        X = pd.DataFrame({
            "temp_high": [25, 26, 24],
            "temp_low": [18, 19, 17],
            "humidity": [60, 65, 55],
            "precip_chance": [30, 40, 20],
            "precip_mm": [5, 8, 3],
            "wind_speed": [10, 12, 8],
            "day_of_year": [165, 166, 167]
        })
        y = pd.Series([1, 1, 0])
        model.train(X, y)
        
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == 7


class TestRiskAssessmentModel:
    """Test suite for RiskAssessmentModel"""
    
    @pytest.fixture
    def model(self):
        """Fixture for risk assessment model"""
        return RiskAssessmentModel()
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert model is not None
        assert isinstance(model, MLModelStrategy)
        assert model.risk_thresholds is not None
    
    def test_risk_thresholds_loaded(self, model):
        """Test that risk thresholds are loaded correctly"""
        assert "frost" in model.risk_thresholds
        assert "drought" in model.risk_thresholds
        assert "extreme_wind" in model.risk_thresholds
        assert "heat_stress" in model.risk_thresholds
        assert "heavy_rain" in model.risk_thresholds
    
    def test_assess_risks_with_normal_conditions(self, model, mock_forecast_data, mock_location):
        """Test risk assessment with normal conditions"""
        risk_assessment = model.assess_risks(mock_forecast_data, mock_location)
        
        assert risk_assessment is not None
        assert hasattr(risk_assessment, 'overall_risk')
        assert hasattr(risk_assessment, 'risk_factors')
        assert hasattr(risk_assessment, 'timestamp')
        assert risk_assessment.location == mock_location
    
    def test_assess_risks_with_frost(self, model, mock_location):
        """Test frost risk assessment"""
        frost_forecast = [
            ForecastData(
                location=mock_location,
                date=datetime.now(timezone.utc),
                temperature_high=5,
                temperature_low=-5,
                humidity=80,
                precipitation_chance=50,
                precipitation_mm=2,
                wind_speed=15,
                condition=WeatherCondition.SNOW
            )
        ]
        
        risk_assessment = model.assess_risks(frost_forecast, mock_location)
        
        assert risk_assessment is not None
        frost_risks = [rf for rf in risk_assessment.risk_factors if rf.type == "frost"]
        assert len(frost_risks) > 0
    
    def test_assess_risks_with_drought(self, model, mock_location):
        """Test drought risk assessment"""
        drought_forecast = [
            ForecastData(
                location=mock_location,
                date=datetime.now(timezone.utc),
                temperature_high=35,
                temperature_low=25,
                humidity=20,
                precipitation_chance=5,
                precipitation_mm=0,
                wind_speed=5,
                condition=WeatherCondition.CLEAR
            )
        ]
        
        risk_assessment = model.assess_risks(drought_forecast, mock_location)
        
        assert risk_assessment is not None
        drought_risks = [rf for rf in risk_assessment.risk_factors if rf.type == "drought"]
        assert len(drought_risks) > 0
    
    def test_assess_risks_with_extreme_wind(self, model, mock_location):
        """Test extreme wind risk assessment"""
        wind_forecast = [
            ForecastData(
                location=mock_location,
                date=datetime.now(timezone.utc),
                temperature_high=25,
                temperature_low=18,
                humidity=60,
                precipitation_chance=30,
                precipitation_mm=5,
                wind_speed=55,
                condition=WeatherCondition.WINDY
            )
        ]
        
        risk_assessment = model.assess_risks(wind_forecast, mock_location)
        
        assert risk_assessment is not None
        wind_risks = [rf for rf in risk_assessment.risk_factors if rf.type == "extreme_wind"]
        assert len(wind_risks) > 0
    
    def test_assess_risks_empty_forecast(self, model, mock_location):
        """Test risk assessment with empty forecast"""
        risk_assessment = model.assess_risks([], mock_location)
        
        assert risk_assessment is not None
        assert risk_assessment.overall_risk.value == "low"
        assert len(risk_assessment.risk_factors) == 0
    
    def test_calculate_overall_risk_critical(self, model):
        """Test overall risk calculation with critical risks"""
        from app.core.models import RiskFactor, RiskLevel
        
        risk_factors = [
            RiskFactor(
                type="frost",
                level=RiskLevel.CRITICAL,
                probability=0.9,
                description="Critical frost"
            )
        ]
        
        overall_risk = model._calculate_overall_risk(risk_factors)
        assert overall_risk == RiskLevel.CRITICAL
    
    def test_calculate_overall_risk_high(self, model):
        """Test overall risk calculation with high risks"""
        from app.core.models import RiskFactor, RiskLevel
        
        risk_factors = [
            RiskFactor(
                type="drought",
                level=RiskLevel.HIGH,
                probability=0.7,
                description="High drought"
            ),
            RiskFactor(
                type="wind",
                level=RiskLevel.HIGH,
                probability=0.6,
                description="High wind"
            )
        ]
        
        overall_risk = model._calculate_overall_risk(risk_factors)
        assert overall_risk == RiskLevel.CRITICAL
    
    def test_train_model(self, model):
        """Test model training with Isolation Forest"""
        X = pd.DataFrame({
            "temp_high": [25, 26, 24, 27, 25],
            "temp_low": [18, 19, 17, 20, 18],
            "humidity": [60, 65, 55, 70, 60],
            "precip_chance": [30, 40, 20, 50, 30],
            "precip_mm": [5, 8, 3, 10, 5],
            "wind_speed": [10, 12, 8, 15, 10]
        })
        
        model.train(X)
        assert model.is_trained == True
        assert model.model is not None


class TestCanopyHealthModel:
    """Test suite for CanopyHealthModel"""
    
    @pytest.fixture
    def model(self):
        """Fixture for canopy health model"""
        return CanopyHealthModel()
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert model is not None
        assert isinstance(model, MLModelStrategy)
        assert model.scaler is not None
    
    def test_calculate_health_score(self, model, mock_tree_analysis):
        """Test health score calculation"""
        score = model.calculate_health_score(mock_tree_analysis)
        
        assert score is not None
        assert 0 <= score <= 100
        assert isinstance(score, (int, float))
    
    def test_calculate_health_score_all_healthy(self, model):
        """Test health score with all healthy trees"""
        analysis = TreeAnalysis(
            analysis_id="test",
            timestamp=datetime.now(timezone.utc),
            farmer_id="F-001",
            county="Bomet",
            land_acres=2.5,
            total_tree_count=100,
            tree_density_per_acre=40,
            confidence_score=1.0,
            canopy_coverage_pct=100,
            tree_health=TreeHealth(healthy=100, needs_care=0, needs_replacement=0)
        )
        
        score = model.calculate_health_score(analysis)
        assert score == 100
    
    def test_calculate_health_score_all_needs_replacement(self, model):
        """Test health score with all trees needing replacement"""
        analysis = TreeAnalysis(
            analysis_id="test",
            timestamp=datetime.now(timezone.utc),
            farmer_id="F-001",
            county="Bomet",
            land_acres=2.5,
            total_tree_count=100,
            tree_density_per_acre=40,
            confidence_score=1.0,
            canopy_coverage_pct=100,
            tree_health=TreeHealth(healthy=0, needs_care=0, needs_replacement=100)
        )
        
        score = model.calculate_health_score(analysis)
        assert score == 0
    
    def test_analyze_health_trend_insufficient_data(self, model, mock_tree_analysis):
        """Test health trend analysis with insufficient data"""
        trend = model.analyze_health_trend([mock_tree_analysis])
        
        assert trend is not None
        assert trend["trend"] == "insufficient_data"
    
    def test_analyze_health_trend_improving(self, model, mock_tree_analysis):
        """Test health trend analysis showing improvement"""
        # Create two analyses showing improvement
        old_analysis = TreeAnalysis(
            analysis_id="old",
            timestamp=datetime.now(timezone.utc),
            farmer_id="F-001",
            county="Bomet",
            land_acres=2.5,
            total_tree_count=84,
            tree_density_per_acre=33.6,
            confidence_score=0.87,
            canopy_coverage_pct=30.0,
            tree_health=TreeHealth(healthy=50, needs_care=20, needs_replacement=14)
        )
        
        new_analysis = TreeAnalysis(
            analysis_id="new",
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
        
        trend = model.analyze_health_trend([old_analysis, new_analysis])
        
        assert trend is not None
        assert trend["trend"] == "improving"
        assert trend["health_change"] > 0
    
    def test_analyze_health_trend_declining(self, model, mock_tree_analysis):
        """Test health trend analysis showing decline"""
        old_analysis = TreeAnalysis(
            analysis_id="old",
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
        
        new_analysis = TreeAnalysis(
            analysis_id="new",
            timestamp=datetime.now(timezone.utc),
            farmer_id="F-001",
            county="Bomet",
            land_acres=2.5,
            total_tree_count=84,
            tree_density_per_acre=33.6,
            confidence_score=0.87,
            canopy_coverage_pct=30.0,
            tree_health=TreeHealth(healthy=50, needs_care=20, needs_replacement=14)
        )
        
        trend = model.analyze_health_trend([old_analysis, new_analysis])
        
        assert trend is not None
        assert trend["trend"] == "declining"
        assert trend["health_change"] < 0
    
    def test_compare_with_benchmark_above(self, model, mock_tree_analysis):
        """Test benchmark comparison when above benchmark"""
        result = model.compare_with_benchmark(mock_tree_analysis, benchmark_density=25.0)
        
        assert result is not None
        assert result["status"] == "above_benchmark"
        assert result["current_density"] > result["benchmark_density"]
    
    def test_compare_with_benchmark_below(self, model, mock_tree_analysis):
        """Test benchmark comparison when below benchmark"""
        result = model.compare_with_benchmark(mock_tree_analysis, benchmark_density=45.0)
        
        assert result is not None
        assert result["status"] == "below_benchmark"
        assert result["current_density"] < result["benchmark_density"]
    
    def test_compare_with_benchmark_within(self, model, mock_tree_analysis):
        """Test benchmark comparison when within range"""
        result = model.compare_with_benchmark(mock_tree_analysis, benchmark_density=33.6)
        
        assert result is not None
        assert result["status"] == "within_benchmark"
    
    def test_train_model(self, model):
        """Test model training with KMeans"""
        X = pd.DataFrame({
            "total_tree_count": [84, 90, 80, 85, 82],
            "canopy_coverage_pct": [41.2, 45.0, 38.0, 42.0, 40.0],
            "confidence_score": [0.87, 0.90, 0.85, 0.88, 0.86]
        })
        
        model.train(X)
        assert model.is_trained == True
        assert model.model is not None
